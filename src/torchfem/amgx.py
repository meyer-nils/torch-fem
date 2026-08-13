"""Thin ctypes binding for AmgX (github.com/NVIDIA/AMGX), an optional GPU AMG backend.

AmgX ships no wheels: build it from source and point `AMGX_DLL` at the shared
library. Importing raises `ImportError` when it is missing, so `sparse.py` can
guard it like CuPy and PyPardiso. Binding touches no GPU -- `AMGX_initialize()`
waits for the first solver, so importing torchfem creates no CUDA context.

Only mode "dDDI" is used (device, double matrix, double vector, int32 index),
matching the CSR that `_solve_gpu` already builds. Uploads point straight at
CuPy device pointers, which AmgX resolves with `cudaMemcpyDefault`, so nothing
round-trips through the host.

Handles are freed explicitly by `close()`, never from `__del__`: destruction is
order-dependent and a stale handle segfaults rather than raises. Nothing calls
`AMGX_finalize()`, so a dropped solver leaks until the process exits, which
beats risking a bad teardown order.

The C API takes no near-null-space basis, pyamg's `B`. Vector problems instead
aggregate over `block_size x block_size` nodal blocks, which holds the
translational rigid-body modes implicitly; `_solve_gpu` reads that size off `B`.
Scalar mode diverged on half the elasticity systems tried, 3x3 blocks on none.

`resolve_method` never picks `amgx`, since whether it wins depends on the matrix
rather than on anything the selection can cheaply inspect. Against the CuPy
Jacobi path at `stol=1e-10` on an RTX 4060 Ti it is 1.7-2.7x quicker on scalar
heat conduction and up to 2.1x on elasticity of varying stiffness, but 1.5-1.8x
slower on uniform elasticity, where a smooth solution leaves multigrid little to
remove and a V-cycle costs some seven Jacobi iterations. Aggregation also wants
a definite operator: an indefinite tangent costs every AMG an order of magnitude
more iterations, so `max_iters` stays low and a solve that exhausts it raises,
on the grounds that such a tangent is worth reformulating instead.

`sparse.py` imports torch first because AmgX's `cublasDdot` returns
CUBLAS_STATUS_NOT_SUPPORTED otherwise; importing torchfem is enough.
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

# Aggregation AMG shaped after the CPU default (pyamg
# smoothed_aggregation_solver, smooth="jacobi"): a symmetric Gauss-Seidel sweep
# either side of the cycle and an exact coarse solve, which roughly halves the
# iterations AmgX's own defaults need. SIZE_8 then shortens the hierarchy by
# more than the iterations it costs. BiCGStab wraps it because the sweeps are
# not symmetric, so PCG stalls short of a tight `stol`; FGMRES matches it on
# elasticity but is slower on heat conduction and stores restart vectors.
# `tolerance` is overwritten per solve with `stol`.
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
            "selector": "SIZE_8",
            "interpolator": "D2",
            "smoother": {"solver": "MULTICOLOR_GS"},
            "symmetric_GS": 1,
            "matrix_coloring_scheme": "MIN_MAX",
            "max_uncolored_percentage": 0.15,
            "presweeps": 1,
            "postsweeps": 1,
            "cycle": "V",
            "max_iters": 1,
            "coarse_solver": "DENSE_LU_SOLVER",
            "min_coarse_rows": 16,
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

    `setup()` builds the hierarchy from a `cupyx.scipy.sparse.csr_matrix` with
    float64 data and int32 indices; `resetup()` reuses it across a Newton loop
    where only the coefficients change. `block_size` is the degrees of freedom
    per node, which AmgX aggregates as a unit.
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
        """Return `A_cp` as `(indptr, indices, values, count)` over nodal blocks.

        A scalar problem passes the CSR arrays straight through. The ordering
        follows from the entry pattern alone, so `resetup` on an unchanged
        structure reproduces it and can replace the values in place.
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
        """Solve for `b_cp` as a CuPy array, warm-started from `x0_cp`."""
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
