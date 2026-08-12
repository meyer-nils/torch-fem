"""Thin ctypes binding for AmgX (github.com/NVIDIA/AMGX), an optional GPU AMG backend.

AmgX ships no wheels; using it requires building `amgxsh.dll`/`libamgxsh.so`
from source against a matching CUDA toolkit and pointing the `AMGX_DLL`
environment variable at it. Importing this module raises `ImportError` when
that library is missing, so `sparse.py` can guard it with the same
try/except that guards CuPy and PyPardiso. Binding the symbols touches no
GPU; `AMGX_initialize()` is deferred to the first solver, so importing
torchfem never creates a CUDA context.

Only the scalar "dDDI" mode is supported (device-resident, double matrix,
double vector, int32 index), matching the CSR matrices `_solve_gpu` already
builds. Matrix/vector uploads and downloads are pointed directly at CuPy
device pointers (`.data.ptr`); AmgX resolves these via `cudaMemcpyDefault`
(confirmed in `src/amgx_c.cu`), so there is no host round-trip.

Handle lifetimes are managed explicitly, never via `__del__` or
`weakref.finalize`: destruction order matters (`AMGX_finalize` must be last),
and getting it wrong segfaults rather than raising, especially once CuPy's
memory pool and Python's GC are both in the picture. `AmgXSolver.close()`
frees a solver's own handles; nothing calls `AMGX_finalize()` at all, so a
solver that is simply dropped without `close()` leaks its GPU resources for
the remainder of the process rather than risking a bad teardown order.

The C API has no equivalent of pyamg's `B` argument -- no near-null-space basis
can be handed to AmgX. Its answer for vector problems is instead to aggregate
over `block_size x block_size` nodal blocks, which keeps the translational
rigid-body modes in the coarse space implicitly. That is worth doing: on linear
elasticity, scalar mode diverged on three of six test systems, while 3x3 blocks
converged on all of them, with about 25% fewer iterations and less wall time
than scalar wherever both converged. `_solve_gpu` therefore derives the block
size from `B` and passes it in.

`resolve_method` never selects `amgx` on its own -- it is opt-in per call.
Measured against the CuPy Jacobi path at `stol=1e-10` on an RTX 4060 Ti, it
takes 3.9-6.4x fewer iterations across 24k-648k dof on both linear elasticity
(blocked) and scalar heat conduction. A V-cycle costs far more than a diagonal
scaling, though, so that only pays for itself in wall time on heat conduction
and the smaller elasticity systems; past ~200k dof of elasticity it is about
30% slower than Jacobi despite needing a fifth of the iterations. Solves that
fail to converge raise, so callers can fall back to `cg` or `minres`.

One load-order caveat, which is why `sparse.py` imports torch first: AmgX's
`cublasDdot` returns `CUBLAS_STATUS_NOT_SUPPORTED` unless torch is imported
before the first solve, presumably because torch loads a cuBLAS this build is
happy with. Importing torchfem at all satisfies this.
"""

from __future__ import annotations

import ctypes as C
import json
import os
from copy import deepcopy
from typing import Any

ERR_AMGX_MISSING = (
    "AmgX is not available.\n\n"
    "AmgX ships no wheels: build it from source (github.com/NVIDIA/AMGX) "
    "against a matching CUDA toolkit, then point AMGX_DLL at the resulting "
    "shared library, e.g.\n"
    "> export AMGX_DLL=/path/to/libamgxsh.so   # amgxsh.dll on Windows"
)

# AMGX_mode_dDDI: device-resident, double matrix, double vector, int32 index.
# Hardcoded from the enum value documented in amgx_c.h/amgx_config.h ("mode ==
# 8193"); "dDDI" is the only mode this binding ever asks AmgX to build.
_MODE_dDDI = 8193

_AMGX_RC_OK = 0
_AMGX_SOLVE_SUCCESS = 0
_AMGX_SOLVE_STATUS_NAMES = {1: "FAILED", 2: "DIVERGED", 3: "NOT_CONVERGED"}

# Aggregation AMG with a Jacobi smoother, mirroring the CPU default (pyamg
# smoothed_aggregation_solver, smooth="jacobi"). `tolerance` is overwritten per
# solve with the caller's `stol`.
#
# The V-cycle sweeps asymmetrically, so the preconditioner is not symmetric and
# PCG loses orthogonality: it stalls short of a tight `stol` on the larger
# elasticity systems, while symmetric sweeps cost an order of magnitude more
# iterations. BiCGStab does not assume a symmetric preconditioner and converged
# on every system tested, with the lowest iteration counts of the methods tried.
_DEFAULT_CONFIG: dict[str, Any] = {
    "config_version": 2,
    "determinism_flag": 1,
    "solver": {
        "solver": "PBICGSTAB",
        "convergence": "RELATIVE_INI",
        "norm": "L2",
        "max_iters": 1000,
        "monitor_residual": 1,
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "selector": "SIZE_2",
            "interpolator": "D2",
            "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": 0.8},
            "presweeps": 0,
            "postsweeps": 3,
            "cycle": "V",
            "max_iters": 1,
            "coarse_solver": "NOSOLVER",
            "max_levels": 50,
        },
    },
}

_SIGNATURES: dict[str, tuple[list[Any], Any]] = {
    "AMGX_initialize": ([], C.c_int),
    "AMGX_get_error_string": ([C.c_int, C.c_char_p, C.c_int], C.c_int),
    "AMGX_config_create": ([C.POINTER(C.c_void_p), C.c_char_p], C.c_int),
    "AMGX_config_destroy": ([C.c_void_p], C.c_int),
    "AMGX_resources_create_simple": ([C.POINTER(C.c_void_p), C.c_void_p], C.c_int),
    "AMGX_resources_destroy": ([C.c_void_p], C.c_int),
    "AMGX_matrix_create": ([C.POINTER(C.c_void_p), C.c_void_p, C.c_int], C.c_int),
    "AMGX_matrix_destroy": ([C.c_void_p], C.c_int),
    "AMGX_matrix_upload_all": (
        [
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
        ],
        C.c_int,
    ),
    "AMGX_matrix_replace_coefficients": (
        [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p],
        C.c_int,
    ),
    "AMGX_vector_create": ([C.POINTER(C.c_void_p), C.c_void_p, C.c_int], C.c_int),
    "AMGX_vector_destroy": ([C.c_void_p], C.c_int),
    "AMGX_vector_upload": ([C.c_void_p, C.c_int, C.c_int, C.c_void_p], C.c_int),
    "AMGX_vector_download": ([C.c_void_p, C.c_void_p], C.c_int),
    "AMGX_solver_create": (
        [C.POINTER(C.c_void_p), C.c_void_p, C.c_int, C.c_void_p],
        C.c_int,
    ),
    "AMGX_solver_destroy": ([C.c_void_p], C.c_int),
    "AMGX_solver_setup": ([C.c_void_p, C.c_void_p], C.c_int),
    "AMGX_solver_resetup": ([C.c_void_p, C.c_void_p], C.c_int),
    "AMGX_solver_solve": ([C.c_void_p, C.c_void_p, C.c_void_p], C.c_int),
    "AMGX_solver_get_status": ([C.c_void_p, C.POINTER(C.c_int)], C.c_int),
    "AMGX_solver_get_iterations_number": ([C.c_void_p, C.POINTER(C.c_int)], C.c_int),
}


def _bind_library() -> C.CDLL:
    """Load the AmgX shared library and apply the signatures, at import time."""
    name = os.environ.get("AMGX_DLL") or (
        "amgxsh.dll" if os.name == "nt" else "libamgxsh.so"
    )
    try:
        lib = C.CDLL(name)
    except OSError as err:
        raise ImportError(ERR_AMGX_MISSING) from err

    for fname, (argtypes, restype) in _SIGNATURES.items():
        func = getattr(lib, fname)
        func.argtypes = argtypes
        func.restype = restype
    return lib


_lib = _bind_library()
_initialized = False


def _initialize() -> None:
    """Call `AMGX_initialize` once per process, on the first solver."""
    global _initialized
    if _initialized:
        return
    if _lib.AMGX_initialize() != _AMGX_RC_OK:
        raise RuntimeError("AMGX_initialize() failed.")
    _initialized = True


def _check(lib: C.CDLL, rc: int) -> None:
    if rc == _AMGX_RC_OK:
        return
    buf = C.create_string_buffer(4096)
    lib.AMGX_get_error_string(rc, buf, len(buf))
    raise RuntimeError(f"AmgX error {rc}: {buf.value.decode(errors='replace')}")


class AmgXSolver:
    """Owns one AmgX config/resources/matrix/vector/solver handle set.

    `setup()` builds the AMG hierarchy from a CSR matrix; `resetup()` reuses it
    across a Newton loop where only the coefficients change, via
    `AMGX_matrix_replace_coefficients` + `AMGX_solver_resetup`. `A_cp` must be a
    `cupyx.scipy.sparse.csr_matrix` with float64 data and int32 indices/indptr.

    `block_size` is the number of degrees of freedom per node, which AmgX
    aggregates as a unit; see the module docstring for why that matters.
    """

    def __init__(self, n: int, stol: float, block_size: int = 1) -> None:
        _initialize()
        lib = _lib
        self._lib = lib
        self._closed = False

        config = deepcopy(_DEFAULT_CONFIG)
        config["solver"]["tolerance"] = stol
        cfg = C.c_void_p()
        _check(lib, lib.AMGX_config_create(C.byref(cfg), json.dumps(config).encode()))
        self._cfg = cfg

        rsc = C.c_void_p()
        _check(lib, lib.AMGX_resources_create_simple(C.byref(rsc), cfg))
        self._rsc = rsc

        mtx = C.c_void_p()
        _check(lib, lib.AMGX_matrix_create(C.byref(mtx), rsc, _MODE_dDDI))
        self._mtx = mtx

        rhs = C.c_void_p()
        _check(lib, lib.AMGX_vector_create(C.byref(rhs), rsc, _MODE_dDDI))
        self._rhs = rhs

        sol = C.c_void_p()
        _check(lib, lib.AMGX_vector_create(C.byref(sol), rsc, _MODE_dDDI))
        self._sol = sol

        slv = C.c_void_p()
        _check(lib, lib.AMGX_solver_create(C.byref(slv), rsc, _MODE_dDDI, cfg))
        self._slv = slv

        self.n = n
        self.block_size = block_size
        self.n_rows = n // block_size

    def setup(self, A_cp: Any) -> None:
        """Upload `A_cp` and build the AMG hierarchy from scratch."""
        lib = self._lib
        indptr, indices, data, n_blocks = self._blocks(A_cp)
        self._n_blocks = n_blocks
        _check(
            lib,
            lib.AMGX_matrix_upload_all(
                self._mtx,
                self.n_rows,
                n_blocks,
                self.block_size,
                self.block_size,
                indptr.data.ptr,
                indices.data.ptr,
                data.data.ptr,
                None,
            ),
        )
        _check(lib, lib.AMGX_solver_setup(self._slv, self._mtx))

    def resetup(self, A_cp: Any) -> None:
        """Refresh coefficients in place and rebuild only what changed."""
        lib = self._lib
        _, _, data, n_blocks = self._blocks(A_cp)
        _check(
            lib,
            lib.AMGX_matrix_replace_coefficients(
                self._mtx, self.n_rows, n_blocks, data.data.ptr, None
            ),
        )
        _check(lib, lib.AMGX_solver_resetup(self._slv, self._mtx))

    def _blocks(self, A_cp: Any) -> tuple[Any, Any, Any, int]:
        """Return `A_cp` as `(indptr, indices, values, count)` over `block_size` blocks.

        A scalar problem passes the CSR arrays straight through. Otherwise the
        entries are gathered into dense row-major blocks, which is what AmgX
        aggregates over. The block sparsity follows from the entry pattern
        alone, so a `resetup` on a matrix with unchanged structure reproduces
        this ordering and may replace the values in place.
        """
        if self.block_size == 1:
            return A_cp.indptr, A_cp.indices, A_cp.data, A_cp.nnz

        import cupy

        bs = self.block_size
        coo = A_cp.tocoo()
        # int64, since the key runs to n_rows**2 and overflows the int32 column
        # indices at about 46k block rows.
        rows = (coo.row // bs).astype(cupy.int64)
        key = rows * self.n_rows + (coo.col // bs)
        uniq, inverse = cupy.unique(key, return_inverse=True)  # type: ignore
        values = cupy.zeros(len(uniq) * bs * bs)
        values[inverse * bs * bs + (coo.row % bs) * bs + (coo.col % bs)] = coo.data
        indices = (uniq % self.n_rows).astype(cupy.int32)
        indptr = cupy.zeros(self.n_rows + 1, dtype=cupy.int32)
        indptr[1:] = cupy.cumsum(
            cupy.bincount(uniq // self.n_rows, minlength=self.n_rows)
        )
        return indptr, indices, values, len(uniq)

    def solve(self, b_cp: Any, x0_cp: Any | None) -> Any:
        """Solve for `b_cp`, warm-started from `x0_cp` (zeros if None).

        Returns the solution as a CuPy array.
        """
        import cupy

        lib = self._lib
        x_cp = cupy.zeros(self.n) if x0_cp is None else cupy.asarray(x0_cp).copy()
        rows, bs = self.n_rows, self.block_size

        _check(lib, lib.AMGX_vector_upload(self._rhs, rows, bs, b_cp.data.ptr))
        _check(lib, lib.AMGX_vector_upload(self._sol, rows, bs, x_cp.data.ptr))
        _check(lib, lib.AMGX_solver_solve(self._slv, self._rhs, self._sol))

        status = C.c_int()
        _check(lib, lib.AMGX_solver_get_status(self._slv, C.byref(status)))
        if status.value != _AMGX_SOLVE_SUCCESS:
            name = _AMGX_SOLVE_STATUS_NAMES.get(status.value, str(status.value))
            raise RuntimeError(
                f"AmgX solve did not converge ({name}) in {self.iterations} "
                "iterations. Try 'cg' or 'minres' instead, or tune "
                "_DEFAULT_CONFIG for this problem."
            )

        _check(lib, lib.AMGX_vector_download(self._sol, x_cp.data.ptr))
        return x_cp

    @property
    def iterations(self) -> int:
        n = C.c_int()
        _check(
            self._lib,
            self._lib.AMGX_solver_get_iterations_number(self._slv, C.byref(n)),
        )
        return n.value

    def close(self) -> None:
        """Destroy this solver's handles. Not automatic -- see module docstring."""
        if self._closed:
            return
        self._closed = True
        lib = self._lib
        lib.AMGX_solver_destroy(self._slv)
        lib.AMGX_vector_destroy(self._sol)
        lib.AMGX_vector_destroy(self._rhs)
        lib.AMGX_matrix_destroy(self._mtx)
        lib.AMGX_resources_destroy(self._rsc)
        lib.AMGX_config_destroy(self._cfg)
