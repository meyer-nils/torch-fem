"""Thin ctypes binding for AmgX (github.com/NVIDIA/AMGX), an optional GPU AMG backend.

AmgX ships no wheels: build it from source and point `AMGX_DLL` at the shared
library. Importing raises `ImportError` when it is missing, so `sparse.py` can
guard it like PyPardiso, and touches no GPU: `AMGX_initialize()` waits for the
first solver, so importing torchfem creates no CUDA context. `sparse.py` imports
torch first, without which `cublasDdot` returns CUBLAS_STATUS_NOT_SUPPORTED.

Only mode "dDDI" is used (device, double matrix, double vector, int32 index),
matching the CSR a model assembles. Uploads point straight at torch device
pointers, which AmgX resolves with `cudaMemcpyDefault`, so no array is copied
into another library and nothing round-trips through the host.

Handles are freed explicitly by `close()`, never from `__del__`: destruction is
order-dependent and a stale handle segfaults rather than raises. A dropped
solver would therefore hold its hierarchy until the process exits, so `sparse.py`
closes each one at the end of the solve that built it, and no solver is handed
from a forward pass to its adjoint. A solve that fails closes itself, since it
is never reused and a cutback would otherwise leave one behind on every retry.
`AMGX_finalize()` is never called, which beats risking a bad teardown order.

The C API takes no near-null-space basis, pyamg's `B`, which costs dearly on
elasticity. Two things recover most of it. Aggregating over
`block_size x block_size` nodal blocks holds the translational rigid-body modes
implicitly, and passing nodal coordinates to `AMGX_matrix_attach_geometry` lets
the `GEO` selector aggregate geometrically, which is where the rotational modes
hide. `GEO` needs one coordinate per block row, so a shell, whose node spans
two blocks, keeps the algebraic `SIZE_8` selector, as does a scalar problem that
has no coordinates to read. AmgX reads the problem dimension from whether a
third coordinate is attached, so a planar model attaches two rather than a flat
z, which it would call degenerate.

`resolve_preconditioner` prefers AMG for iterative solves on CUDA once AmgX is
installed, and `resolve_library` then routes the whole solve here. Aggregation
wants a definite operator, which an indefinite tangent costs dearly, so
`max_iters` stays low and a solve that exhausts it raises: such a tangent is
worth reformulating instead.
"""

from __future__ import annotations

import ctypes as C
import json
import os
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

# AMGX_mode_dDDI: device-resident, double matrix, double vector, int32 index.
# Hardcoded from the enum value documented in amgx_c.h/amgx_config.h ("mode ==
# 8193"); "dDDI" is the only mode this binding ever asks AmgX to build.
_MODE_dDDI = 8193

_AMGX_RC_OK = 0
_AMGX_SOLVE_SUCCESS = 0
_AMGX_SOLVE_STATUS_NAMES = {1: "FAILED", 2: "DIVERGED", 3: "NOT_CONVERGED"}

# Aggregation AMG shaped after the CPU default (pyamg
# smoothed_aggregation_solver, smooth="jacobi"): two damped Jacobi sweeps either
# side of the cycle and an exact coarse solve. SIZE_8 shortens the hierarchy by
# more than the iterations it costs. The smoother has to be symmetric, because
# PCG preconditions with the whole cycle, and damped enough to hold a
# hyperelastic tangent as the deformation grows. `solver` and `tolerance` are
# overwritten per solve with the Krylov method and `stol`.
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
            "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": 0.5},
            "presweeps": 2,
            "postsweeps": 2,
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
    "AMGX_matrix_attach_geometry": (
        [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int],
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
        # sparse.py turns this into ERR_AMGX_MISSING.
        raise ImportError(f"Could not load the AmgX shared library {name!r}.") from err

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


def _check(rc: int) -> None:
    """Raise with AmgX's own message unless `rc` reports success."""
    if rc == _AMGX_RC_OK:
        return
    buf = C.create_string_buffer(4096)
    _lib.AMGX_get_error_string(rc, buf, len(buf))
    raise RuntimeError(f"AmgX error {rc}: {buf.value.decode(errors='replace')}")


def _create(entry_point: Any, *args: Any) -> C.c_void_p:
    """Return a new handle from one of AmgX's `*_create` entry points."""
    handle = C.c_void_p()
    _check(entry_point(C.byref(handle), *args))
    return handle


class AmgXSolver:
    """Owns one AmgX config/resources/matrix/vector/solver handle set.

    `setup()` builds the hierarchy from a torch sparse CSR tensor with float64
    values and int32 indices; `resetup()` reuses it across a Newton loop
    where only the coefficients change. `block_size` is the degrees of freedom
    per node, which AmgX aggregates as a unit. `coords` are host arrays of nodal
    x and y, and z in three dimensions, which switch aggregation to `GEO`; see
    the module docstring.
    `krylov` is the solver AMG preconditions, `PCG` for a symmetric matrix and
    `PBICGSTAB` otherwise.
    """

    def __init__(
        self,
        n: int,
        stol: float,
        block_size: int = 1,
        coords: tuple[Any, ...] | None = None,
        krylov: str = "PBICGSTAB",
    ) -> None:
        _initialize()
        self._closed = False
        self._coords = coords

        config = deepcopy(_DEFAULT_CONFIG)
        config["solver"]["solver"] = krylov
        config["solver"]["tolerance"] = stol
        if coords is not None:
            config["solver"]["preconditioner"]["selector"] = "GEO"

        self._cfg = _create(_lib.AMGX_config_create, json.dumps(config).encode())
        self._rsc = _create(_lib.AMGX_resources_create_simple, self._cfg)
        self._mtx = _create(_lib.AMGX_matrix_create, self._rsc, _MODE_dDDI)
        self._rhs = _create(_lib.AMGX_vector_create, self._rsc, _MODE_dDDI)
        self._sol = _create(_lib.AMGX_vector_create, self._rsc, _MODE_dDDI)
        self._slv = _create(_lib.AMGX_solver_create, self._rsc, _MODE_dDDI, self._cfg)

        self.n = n
        self.block_size = block_size
        self.n_rows = n // block_size

    def setup(self, A: Tensor) -> None:
        """Upload `A` and build the AMG hierarchy from scratch."""
        indptr, indices, data, n_blocks = self._blocks(A)
        _check(
            _lib.AMGX_matrix_upload_all(
                self._mtx,
                self.n_rows,
                n_blocks,
                self.block_size,
                self.block_size,
                indptr.data_ptr(),
                indices.data_ptr(),
                data.data_ptr(),
                None,
            )
        )
        # The block arrays are torch's, and AmgX allocates its hierarchy from
        # what torch is not holding, so they go back to the driver here.
        del indptr, indices, data
        torch.cuda.empty_cache()
        if self._coords is not None:
            # Host pointers: AmgX copies these element by element on the CPU,
            # unlike the uploads above.
            x, y, *rest = self._coords
            # A null third coordinate is how AmgX is told the problem is planar.
            z = rest[0].ctypes.data if rest else None
            _check(
                _lib.AMGX_matrix_attach_geometry(
                    self._mtx, x.ctypes.data, y.ctypes.data, z, self.n_rows
                )
            )
        _check(_lib.AMGX_solver_setup(self._slv, self._mtx))

    def resetup(self, A: Tensor) -> None:
        """Refresh coefficients in place and rebuild only what changed.

        Geometry is the exception: AmgX consumes the attached coordinates during
        the first setup, and a hierarchy rebuilt without them needs far more
        iterations, so the handles are replaced and the hierarchy built again.
        """
        if self._coords is not None:
            _lib.AMGX_solver_destroy(self._slv)
            _lib.AMGX_matrix_destroy(self._mtx)
            self._mtx = _create(_lib.AMGX_matrix_create, self._rsc, _MODE_dDDI)
            self._slv = _create(
                _lib.AMGX_solver_create, self._rsc, _MODE_dDDI, self._cfg
            )
            self.setup(A)
            return

        _, _, data, n_blocks = self._blocks(A)
        _check(
            _lib.AMGX_matrix_replace_coefficients(
                self._mtx, self.n_rows, n_blocks, data.data_ptr(), None
            )
        )
        del data
        torch.cuda.empty_cache()
        _check(_lib.AMGX_solver_resetup(self._slv, self._mtx))

    def _blocks(self, A: Tensor) -> tuple[Tensor, Tensor, Tensor, int]:
        """Return `A` as `(indptr, indices, values, count)` over nodal blocks.

        A scalar problem passes the CSR arrays straight through. The ordering
        follows from the entry pattern alone, so `resetup` on an unchanged
        structure reproduces it and can replace the values in place.
        """
        # Mode dDDI indexes in int32; a model that assembles one passes through.
        crow = A.crow_indices().to(torch.int32)
        col = A.col_indices().to(torch.int32)
        data = A.values()
        if self.block_size == 1:
            return crow, col, data, data.numel()

        bs = self.block_size
        row = torch.repeat_interleave(
            torch.arange(self.n, dtype=torch.int64, device=col.device), crow.diff()
        )
        # The key runs to n_rows**2, which int32 cannot hold. It and the
        # scatter index are built in place and dropped as soon as they are
        # spent, since either is as long as the matrix has entries.
        key = row // bs
        key *= self.n_rows
        key += col // bs
        uniq = torch.unique(key)
        idx = torch.searchsorted(uniq, key)
        del key
        idx *= bs * bs
        idx += (row % bs) * bs + col % bs
        del row
        values = torch.zeros(len(uniq) * bs * bs, dtype=data.dtype, device=data.device)
        values[idx] = data
        del idx
        indices = (uniq % self.n_rows).to(torch.int32)
        indptr = torch.zeros(self.n_rows + 1, dtype=torch.int32, device=col.device)
        indptr[1:] = torch.cumsum(
            torch.bincount(uniq // self.n_rows, minlength=self.n_rows), 0
        )
        return indptr, indices, values, len(uniq)

    def solve(self, b: Tensor) -> Tensor:
        """Solve for `b` as a torch tensor, returning one."""
        x = torch.zeros(self.n, dtype=b.dtype, device=b.device)
        rows, bs = self.n_rows, self.block_size

        _check(_lib.AMGX_vector_upload(self._rhs, rows, bs, b.data_ptr()))
        _check(_lib.AMGX_vector_upload(self._sol, rows, bs, x.data_ptr()))
        _check(_lib.AMGX_solver_solve(self._slv, self._rhs, self._sol))

        status = C.c_int()
        _check(_lib.AMGX_solver_get_status(self._slv, C.byref(status)))
        if status.value != _AMGX_SOLVE_SUCCESS:
            name = _AMGX_SOLVE_STATUS_NAMES.get(status.value, str(status.value))
            # Read the count before closing: it queries the solver handle. A
            # failed solver is never reused, and only `close()` frees it, so it
            # goes here rather than leaving one behind at every cutback.
            iterations = self.iterations
            self.close()
            raise RuntimeError(
                f"AmgX solve did not converge ({name}) in {iterations} "
                "iterations. Try preconditioner='jacobi' instead, or tune "
                "_DEFAULT_CONFIG for this problem."
            )

        _check(_lib.AMGX_vector_download(self._sol, x.data_ptr()))
        return x

    @property
    def iterations(self) -> int:
        n = C.c_int()
        _check(_lib.AMGX_solver_get_iterations_number(self._slv, C.byref(n)))
        return n.value

    def close(self) -> None:
        """Destroy this solver's handles. Not automatic -- see module docstring."""
        if self._closed:
            return
        self._closed = True
        _lib.AMGX_solver_destroy(self._slv)
        _lib.AMGX_vector_destroy(self._sol)
        _lib.AMGX_vector_destroy(self._rhs)
        _lib.AMGX_matrix_destroy(self._mtx)
        _lib.AMGX_resources_destroy(self._rsc)
        _lib.AMGX_config_destroy(self._cfg)
