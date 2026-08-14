from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pyamg
import torch
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg as scipy_cg
from scipy.sparse.linalg import eigsh as scipy_eigsh
from scipy.sparse.linalg import minres as scipy_minres
from scipy.sparse.linalg import spsolve as scipy_spsolve
from torch import Tensor
from torch.autograd import Function

if TYPE_CHECKING:
    from .amgx import AmgXSolver
    from .report import SolveReport

ERR_CUPY_MISSING = (
    "CuPy is not available.\n\n"
    "Please install CuPy to use GPU acceleration:\n"
    "> pip install cupy-cuda12x # v12.x\n"
    "> pip install cupy-cuda13x # v13.x"
)

ERR_PYPARDISO_MISSING = (
    "PyPardiso is not available.\n\n"
    "Please install Pypardiso separately:\n"
    "> pip install pypardiso"
)

ERR_AMGX_MISSING = (
    "AmgX is not available.\n\n"
    "AmgX ships no wheels, so build it from source and point AMGX_DLL at the "
    "shared library:\n"
    "> export AMGX_DLL=/path/to/libamgxsh.so # amgxsh.dll on Windows"
)

available_backends = ["scipy"]

try:
    import cupy
    from cupyx.scipy.sparse import coo_matrix as cupy_coo_matrix
    from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
    from cupyx.scipy.sparse import diags as cupy_diags
    from cupyx.scipy.sparse.linalg import cg as cupy_cg
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh
    from cupyx.scipy.sparse.linalg import minres as cupy_minres
    from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve

    available_backends.append("cupy")
except ImportError:
    pass

try:
    import pypardiso

    available_backends.append("pypardiso")
except ImportError:
    pass

try:
    from .amgx import AmgXSolver

    available_backends.append("amgx")
except ImportError:
    pass


def resolve_method(n_dofs: int, device: str, method: str | None) -> str:
    """Return the backend that `sparse_solve` uses for a system of this size."""
    if method is not None:
        return method
    if n_dofs < 10000:
        if device == "cpu" and "pypardiso" in available_backends:
            return "pardiso"
        return "spsolve"
    if device == "cuda" and "amgx" in available_backends:
        return "amgx"
    return "minres"


def describe_method(n_dofs: int, device: str, method: str | None) -> str:
    """Describe the backend that `resolve_method` picks, for verbose output."""
    resolved = resolve_method(n_dofs, device, method)
    kind = "direct"
    if resolved in ("minres", "cg"):
        kind = "iterative | " + ("jacobi" if device == "cuda" else "AMG")
    elif resolved == "amgx":
        kind = "iterative | AMG"
    library = "cupy" if device == "cuda" else "scipy"
    if resolved == "pardiso":
        library = "pypardiso"
    elif resolved == "amgx":
        library = "amgx"
    return f"{resolved} | {kind} | {library} | {device}"


class CachedSolve:
    """Cache of the previous solution and gradient, used to warm-start solvers.

    Written only when the caller passes `update_cache=True`. Direct solvers
    ignore it.

    Args:
        previous_x (Tensor | None): Previous forward solution.
            *Shape:* `(n_dofs,)`.
        previous_grad (Tensor | None): Previous adjoint solution.
            *Shape:* `(n_dofs,)`.
    """

    def __init__(
        self, previous_x: Tensor | None = None, previous_grad: Tensor | None = None
    ) -> None:
        self.previous_x = previous_x
        self.previous_grad = previous_grad

    def update_grad(self, grad: Tensor | None) -> None:
        """Stores a detached copy of `grad` as the next backward warm start."""
        self.previous_grad = grad.detach().clone() if grad is not None else None

    def update_x(self, x: Tensor | None) -> None:
        """Stores a detached copy of `x` as the next forward warm start."""
        self.previous_x = x.detach().clone() if x is not None else None


class Solve(Function):
    """
    Inspired by
    - https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
    - https://github.com/pytorch/pytorch/issues/69538
    - https://github.com/cai4cai/torchsparsegradutils
    - https://doi.org/10.48550/arXiv.2601.13994
    """

    @staticmethod
    def forward(
        A: Tensor,
        b: Tensor,
        B: Tensor | None = None,
        stol: float = 1e-10,
        device: str | None = None,
        method: str | None = None,
        M: LinearOperator | AmgXSolver | None = None,
        cached_solve: CachedSolve | None = None,
        update_cache: bool = False,
    ) -> tuple[Tensor, LinearOperator | AmgXSolver | None]:
        """Solve `A x = b`, warm-starting from `cached_solve`.

        See `sparse_solve` for the arguments.

        Returns:
            x (Tensor): Solution vector.
                *Shape:* `(n_dofs,)`.
            M (LinearOperator | AmgXSolver | None): Preconditioner or AmgX
                solver built or reused by the solve, passed on to `backward`
                for the adjoint system.
        """
        x0 = None
        if cached_solve is not None and cached_solve.previous_x is not None:
            x0 = cached_solve.previous_x

        x, M = sparse_solve(A, b, B, stol, device, method, M, x0)

        if update_cache and cached_solve is not None:
            cached_solve.update_x(x)

        return x, M

    @staticmethod
    def backward(ctx, *grad_outputs) -> tuple:
        """Backpropagate through the solve via the adjoint system `A^T λ = grad_x`.

        The matrix gradient `dL/dA_ij = -λ_i x_j` is evaluated only at the
        sparsity pattern of `A`, so no dense matrix is formed.

        Returns:
            Gradients for `A` and `b`, then `None` for the other arguments.
        """
        # Upstream gradient for the solution x
        grad_x = grad_outputs[0]

        # Access the saved variables
        A, x = ctx.saved_tensors

        x0 = None
        if ctx.cached_solve is not None and ctx.cached_solve.previous_grad is not None:
            x0 = ctx.cached_solve.previous_grad

        # Adjoint solve: A^T lambda = grad_x
        gradb, _ = sparse_solve(
            A.T, grad_x, ctx.B, ctx.stol, ctx.device, ctx.method, ctx.M, x0=x0
        )

        # Backprop rule: gradA = -gradb @ x^T, sparse version
        indices = A._indices()
        row = indices[0, :]
        col = indices[1, :]
        val = -gradb[row] * x[col]
        gradA = torch.sparse_coo_tensor(indices, val, A.shape, is_coalesced=True)

        # Update storage for next iteration
        if ctx.update_cache and ctx.cached_solve is not None:
            ctx.cached_solve.update_grad(gradb.detach().clone())

        return gradA, gradb, None, None, None, None, None, None, None

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        """Save `A`, `x` and the solver settings for `backward`.

        Stores the preconditioner *returned* by the forward pass, not the one
        passed in, so the adjoint solve reuses the AMG hierarchy built there.
        """
        A, b, B, stol, device, method, M, cached_solve, update_cache = inputs
        x, M_computed = output
        ctx.save_for_backward(A, x)

        # Save the parameters for backward pass (including the preconditioner)
        ctx.B = B
        ctx.stol = stol
        ctx.device = device
        ctx.method = method
        ctx.M = M_computed
        ctx.cached_solve = cached_solve
        ctx.update_cache = update_cache


def differentiable_sparse_solve(
    A: Tensor,
    b: Tensor,
    B: Tensor | None = None,
    stol: float = 1e-10,
    device: str | None = None,
    method: str | None = None,
    M: LinearOperator | AmgXSolver | None = None,
    cached_solve: CachedSolve | None = None,
    update_cache: bool = False,
) -> Tensor:
    """Solve `A x = b` with custom sparse adjoint autograd support.

    The forward pass may use non-differentiable sparse backends (SciPy, CuPy,
    Pardiso). The backward pass solves the adjoint system and returns gradients
    with respect to both `A` and `b`.
    """
    result, _ = Solve.apply(
        A, b, B, stol, device, method, M, cached_solve, update_cache
    )  # type: ignore
    if result is None:
        raise RuntimeError("Solve.apply returned None, expected a Tensor.")
    return result


def sparse_solve(
    A: Tensor,
    b: Tensor,
    B: Tensor | None = None,
    stol: float = 1e-10,
    device: str | None = None,
    method: str | None = None,
    M: LinearOperator | AmgXSolver | None = None,
    x0: Tensor | None = None,
) -> tuple[Tensor, LinearOperator | AmgXSolver | None]:
    """
    Solve the linear system Ax = b.

    Args:
        A (sparse_coo_tensor): Sparse matrix A.
        b (Tensor): Right-hand side vector b.
        B (Tensor, optional): Null space rigid body modes for AMG preconditioner.
        stol (float, optional): Relative solver tolerance for the iterative solver.
            Defaults to 1e-10.
        device (str, optional): Device to run the computation on ('cpu' or 'cuda').
            Defaults to None, which uses the current device.
        method (str, optional): Method to use for solving ('spsolve', 'minres',
            'cg', 'pardiso', 'amgx'). Defaults to None for automatic selection based
            on the input size and available backends. 'amgx' is never chosen
            automatically and requires an AmgX build (see `torchfem.amgx`).
        M (Tensor, optional): Preconditioner matrix for iterative methods, or an
            `AmgXSolver` to reuse for method='amgx'. Defaults to None.
        x0 (Tensor, optional): Initial guess for iterative solvers. Defaults to None.

    Returns:
        x (Tensor): Solution vector.
            *Shape:* `(n_dofs,)`.
        M (LinearOperator | AmgXSolver | None): Preconditioner or AmgX solver
            built or reused by the solve, `None` for direct methods.
    """
    # Check the input shape
    if A.ndim != 2 or (A.shape[0] != A.shape[1]):
        raise ValueError("A should be a square 2D matrix.")
    shape = A.size()
    out_device = b.device

    # Check the input method
    if method is not None and method not in [
        "spsolve",
        "minres",
        "cg",
        "pardiso",
        "amgx",
    ]:
        raise ValueError(
            f"Method {method} is not supported. "
            "Choose from 'spsolve', 'minres', 'cg', 'pardiso', or 'amgx'."
        )

    # Move to requested device, if available
    if device is not None:
        A = A.to(device)
        b = b.to(device)
        if B is not None:
            B = B.to(device)
        if x0 is not None:
            x0 = x0.to(device)

    # Make default solver choice based on shape and available backends
    method = resolve_method(shape[0], A.device.type, method)

    # Solve either on CPU or GPU
    if A.device.type == "cuda":
        x_xp, M_xp = _solve_gpu(A, b, B, method, stol, M, shape, x0)
    else:
        x_xp, M_xp = _solve_cpu(A, b, B, method, stol, M, shape, x0)

    # Convert back to torch
    x = torch.tensor(x_xp, dtype=b.dtype, device=out_device)

    return x, M_xp


def _solve_gpu(
    A: Tensor,
    b: Tensor,
    B: Tensor | None,
    method: str,
    stol: float,
    M: LinearOperator | AmgXSolver | None,
    shape: torch.Size,
    x0: Tensor | None,
) -> tuple[Any, Any]:
    """Solve `A x = b` on the GPU via CuPy. See `sparse_solve` for arguments.

    Iterative methods build a Jacobi preconditioner from `A` unless `M` is
    supplied, mirroring `_solve_cpu`. `amgx` reuses `M` as an `AmgXSolver` the
    same way, refreshing coefficients instead of a fresh setup, and reads the
    nodal block size off `B`. `pardiso` is CPU only and raises.
    """
    if "cupy" not in available_backends:
        raise RuntimeError(ERR_CUPY_MISSING)

    # Torch's pool holds GBs of freed blocks that CuPy cannot allocate from
    cupy.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

    # Copy tensors to CuPy. An adjoint solve passes `K.T`, whose entries are
    # sorted by column: reading them as `K` and letting cuSPARSE flip that back
    # costs about half of what coalescing the transpose does. Anything else
    # uncoalesced, such as a sum of two matrices, is coalesced as before.
    flip = False
    if not A.is_coalesced():
        row, col = A._indices()
        ascending = (col[1:] == col[:-1]) & (row[1:] > row[:-1])
        flip = bool(((col[1:] > col[:-1]) | ascending).all())
        if not flip:
            A = A.coalesce()
    idx = A._indices()
    row, col = (idx[1], idx[0]) if flip else (idx[0], idx[1])
    indices = cupy.asarray(col).astype(cupy.int32)
    indptr = cupy.zeros(shape[0] + 1, dtype=cupy.int32)
    indptr[1:] = cupy.cumsum(cupy.bincount(cupy.asarray(row), minlength=shape[0]))
    A_cp = cupy_csr_matrix((cupy.asarray(A._values()), indices, indptr), shape=shape)
    A_cp.has_sorted_indices = True
    if flip:
        A_cp = A_cp.T.tocsr()
    b_cp = cupy.asarray(b.data)

    if x0 is not None:
        x0_cp = cupy.asarray(x0)
    else:
        x0_cp = None

    if method == "pardiso":
        raise RuntimeError("Pardiso backend is not available on GPU.")
    elif method == "spsolve":
        x_xp = cupy_spsolve(A_cp, b_cp)
        M = None
    elif method == "minres":
        # Jacobi preconditioner, unless one was already built and passed in
        if M is None:
            M = cupy_diags(1.0 / A_cp.diagonal())
        # Solve with minres
        x_xp, exit_code = cupy_minres(A_cp, b_cp, M=M, tol=stol, x0=x0_cp)
        if exit_code != 0:
            raise RuntimeError(f"minres failed with exit code {exit_code}")
    elif method == "cg":
        # Jacobi preconditioner, unless one was already built and passed in
        if M is None:
            M = cupy_diags(1.0 / A_cp.diagonal())
        # Solve with conjugate gradients
        x_xp, exit_code = cupy_cg(A_cp, b_cp, M=M, rtol=stol, x0=x0_cp)
        if exit_code != 0:
            raise RuntimeError(f"CG failed with exit code {exit_code}")
    elif method == "amgx":
        if "amgx" not in available_backends:
            raise RuntimeError(ERR_AMGX_MISSING)
        # AMG hierarchy, built from scratch unless one was already passed in,
        # in which case only its coefficients are refreshed.
        if M is None:
            # AmgX uses coordinates insteas of null space B
            coords = None
            block_size = {6: 3, 3: 2}.get(B.shape[1], 1) if B is not None else 1
            if B is not None and B.shape[1] == 6:
                x, y, z = (
                    np.ascontiguousarray(c.detach().cpu().numpy())
                    for c in (B[1::3, 5], B[2::3, 3], B[0::3, 4])
                )
                coords = x, y, z
            M = AmgXSolver(shape[0], stol, block_size, coords)
            M.setup(A_cp)
        else:
            assert isinstance(M, AmgXSolver)
            M.resetup(A_cp)
        x_xp = M.solve(b_cp, x0_cp)

    return x_xp, M


def _solve_cpu(
    A: Tensor,
    b: Tensor,
    B: Tensor | None,
    method: str,
    stol: float,
    M: LinearOperator | AmgXSolver | None,
    shape: torch.Size,
    x0: Tensor | None,
) -> tuple[Any, LinearOperator | AmgXSolver | None]:
    """Solve `A x = b` on the CPU via SciPy. See `sparse_solve` for arguments.

    Iterative methods build an AMG preconditioner from `A` and `B` unless `M` is
    supplied. `pardiso` reorders with reverse Cuthill-McKee before factorising.
    """
    A_np = scipy_coo_matrix(
        (A._values(), (A._indices()[0], A._indices()[1])), shape=shape
    ).tocsr()
    b_np = b.data.numpy()

    if x0 is not None:
        x0_np = x0.data.numpy()
    else:
        x0_np = None

    if B is None:
        B_np = None
    else:
        B_np = B.data.numpy()

    if method == "pardiso":
        if "pypardiso" not in available_backends:
            raise RuntimeError(ERR_PYPARDISO_MISSING)
        # Reorder the matrix using reverse Cuthill-McKee algorithm
        rcm_order = csgraph.reverse_cuthill_mckee(A_np)
        A_rcm = A_np[np.ix_(rcm_order, rcm_order)]
        b_rcm = b_np[rcm_order]
        # Solve with pypardiso
        x_rcm = pypardiso.spsolve(A_rcm, b_rcm)
        # Restore the original order
        inv_rcm_order = np.argsort(rcm_order)
        x_xp = x_rcm[inv_rcm_order]
        M = None
    elif method == "spsolve":
        x_xp = scipy_spsolve(A_np, b_np)
        M = None
    elif method == "minres":
        # AMG preconditioner with Jacobi smoother
        if M is None:
            ml = pyamg.smoothed_aggregation_solver(A_np, B_np, smooth="jacobi")
            M = ml.aspreconditioner()

        # Solve with minres
        x_xp, exit_code = scipy_minres(A_np, b_np, M=M, rtol=stol, x0=x0_np)  # type: ignore
        if exit_code != 0:
            raise RuntimeError(f"minres failed with exit code {exit_code}")
    elif method == "cg":
        # AMG preconditioner with Jacobi smoother
        if M is None:
            ml = pyamg.smoothed_aggregation_solver(A_np, B_np, smooth="jacobi")
            M = ml.aspreconditioner()

        # Solve with cg
        x_xp, exit_code = scipy_cg(A_np, b_np, M=M, rtol=stol, x0=x0_np)
        if exit_code != 0:
            raise RuntimeError(f"CG failed with exit code {exit_code}")

    return x_xp, M


class NewtonRaphsonAdjoint(Function):
    """Custom autograd function for nonlinear Newton-Raphson solves.

    The forward pass performs Newton iterations on the residual callback
    `eval_residual(du, iter, u_prev, grad_prev, flux_prev, state_prev) ->
    (residual, tangent)` and returns the converged increment `du`.

    In the backward pass, gradients are computed by an implicit adjoint
    relation at the converged state:

    1) Solve the adjoint linear system `K^T lambda = grad_du`.
    2) Recompute the residual with a differentiable local state.
    3) Differentiate the residual with respect to the previous increment's
       state (`u_prev`, `grad_prev`, `flux_prev`, `state_prev`) and
       the explicit parameter tensors passed to `newton_solve`. The state
       gradients let autograd chain sensitivities across load increments.

    This avoids differentiating through all Newton iterations.

    Inspired by:
    - https://doi.org/10.48550/arXiv.2601.13994
    """

    @staticmethod
    def forward(
        ctx,
        eval_residual: Callable,
        du: Tensor,
        B: Tensor,
        max_iter: int,
        rtol: float,
        atol: float,
        stol: float,
        report: SolveReport | None,
        method: str | None = None,
        device: str | None = None,
        cached_solve: CachedSolve | None = None,
        update_cache: bool = False,
        u_prev: Tensor | None = None,
        grad_prev: Tensor | None = None,
        flux_prev: Tensor | None = None,
        state_prev: Tensor | None = None,
        *parameters: Tensor,
    ) -> Tensor:
        """Run Newton iterations until the residual meets `rtol` or `atol`.

        Only the converged state is saved for backward, and the warm start
        applies to the first Newton step only. See `newton_solve` for arguments.

        Raises:
            RuntimeError: If the residual becomes NaN or infinite, or the
                iteration limit is reached.
        """
        M = None
        converged_iter = max_iter - 1

        # Newton-Raphson iterations
        for i in range(max_iter):
            # Evaluate residual, stiffness matrix, and internal forces
            residual, K = eval_residual(du, i, u_prev, grad_prev, flux_prev, state_prev)

            # Compute residual norm
            res_norm = torch.linalg.norm(residual)

            # Save initial residual
            if i == 0:
                res_norm0 = res_norm

            # Report iteration information
            if report is not None:
                report.iteration(i, res_norm)

            # Check convergence
            if res_norm < rtol * res_norm0 or res_norm < atol:
                converged_iter = i
                break

            if torch.isnan(res_norm) or torch.isinf(res_norm):
                raise RuntimeError("Newton-Raphson iteration did not converge")

            x0 = None
            if (
                i == 0
                and cached_solve is not None
                and cached_solve.previous_x is not None
            ):
                x0 = cached_solve.previous_x

            # Solve for displacement increment
            du_i, M = sparse_solve(K, residual, B, stol, device, method, M, x0=x0)

            if i == 0 and update_cache and cached_solve is not None:
                cached_solve.update_x(du_i)

            du = du - du_i

        # Final convergence check
        if res_norm > rtol * res_norm0 and res_norm > atol:
            raise RuntimeError("Newton-Raphson iteration did not converge.")

        ctx.save_for_backward(
            K, du, u_prev, grad_prev, flux_prev, state_prev, *parameters
        )
        ctx.B = B
        ctx.M = M
        ctx.stol = stol
        ctx.device = device
        ctx.method = method
        ctx.eval_residual = eval_residual
        ctx.cached_solve = cached_solve
        ctx.update_cache = update_cache
        ctx.n_parameters = len(parameters)
        ctx.converged_iter = converged_iter

        return du

    @staticmethod
    def backward(ctx, *grad_outputs) -> tuple:
        """Differentiate the converged solve via the adjoint `K^T λ = grad_du`.

        The residual is recomputed once with differentiable inputs and `-λ`
        pulled back through it, so gradients reach the previous increment's
        state and chain across load increments.

        Returns:
            `None` for solver arguments, then state and parameter gradients.
        """
        grad_du = grad_outputs[0]

        K, du, u_prev, grad_prev, flux_prev, state_prev, *parameters = ctx.saved_tensors

        B = ctx.B
        M = ctx.M
        stol = ctx.stol
        device = ctx.device
        method = ctx.method
        eval_residual = ctx.eval_residual
        cached_solve = ctx.cached_solve
        update_cache = ctx.update_cache
        converged_iter = ctx.converged_iter

        x0 = None
        if cached_solve is not None and cached_solve.previous_grad is not None:
            x0 = cached_solve.previous_grad

        # Solve adjoint system.
        lambda_, _ = sparse_solve(
            K.T,
            grad_du,
            B,
            stol,
            device,
            method,
            M,
            x0=x0,
        )

        if update_cache and cached_solve is not None:
            cached_solve.update_grad(lambda_)

        # Recompute the residual with a differentiable local state.
        du_local = du.detach().requires_grad_(True)
        prev_local = tuple(
            p.detach().requires_grad_(True)
            for p in (u_prev, grad_prev, flux_prev, state_prev)
        )
        with torch.enable_grad(), torch.device(du_local.device):
            residual, _ = eval_residual(du_local, converged_iter, *prev_local)

        grad_inputs = (du_local, *prev_local, *parameters)
        grads = torch.autograd.grad(
            residual,
            grad_inputs,
            grad_outputs=-lambda_,
            allow_unused=True,
            retain_graph=True,
        )
        grad_prev_state = grads[1:5]
        grad_parameters = grads[5:]

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *grad_prev_state,
            *grad_parameters,
        )


def newton_solve(
    eval_residual: Callable,
    du: Tensor,
    B: Tensor,
    max_iter: int,
    rtol: float,
    atol: float,
    stol: float,
    report: SolveReport | None,
    method: str | None = None,
    device: str | None = None,
    cached_solve: CachedSolve | None = None,
    update_cache: bool = False,
    u_prev: Tensor | None = None,
    grad_prev: Tensor | None = None,
    flux_prev: Tensor | None = None,
    state_prev: Tensor | None = None,
    *parameters: Tensor,
) -> Tensor:
    """Solve a nonlinear residual equation with adjoint-safe Newton iterations.

    Args:
        eval_residual: Callback returning `(residual, tangent)` for the
            current iterate, Newton iteration index, and previous state.
        du: Initial guess for the unknown increment.
        B: Null-space rigid-body basis for AMG preconditioning.
        max_iter: Maximum Newton iterations.
        rtol: Relative residual tolerance.
        atol: Absolute residual tolerance.
        stol: Linear-solver tolerance used inside Newton steps for iterative solvers.
        report: Optional progress report receiving the iteration residuals.
        method: Sparse backend method name.
        device: Optional sparse backend device hint.
        cached_solve: Optional storage for warm-start vectors.
        update_cache: If True, updates cached vectors.
        u_prev: Previous increment's field values. Receives residual gradients
            in backward so sensitivities chain across increments.
        grad_prev: Previous increment's gradient (e.g. deformation gradient).
        flux_prev: Previous increment's flux (e.g. stress).
        state_prev: Previous increment's internal state variables.
        *parameters: Explicit tensors that should receive gradients via the
            implicit adjoint backward.

    Returns:
        du (Tensor): Converged increment.
            *Shape:* `(n_dofs,)`.
    """
    du = NewtonRaphsonAdjoint.apply(
        eval_residual,
        du,
        B,
        max_iter,
        rtol,
        atol,
        stol,
        report,
        method,
        device,
        cached_solve,
        update_cache,
        u_prev,
        grad_prev,
        flux_prev,
        state_prev,
        *parameters,
    )  # type: ignore
    if du is None:
        raise RuntimeError("Solve.apply returned None, expected a Tensor.")
    return du


def _eigsolve_cpu(
    K: Tensor,
    M: Tensor,
    n_modes: int,
    free_indices: Tensor,
    shape: torch.Size,
    n_dofs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigenproblem on the CPU via SciPy.

    Shift-invert at `sigma=0.0` targets the lowest modes. Eigenvectors are
    scattered back from the free-DOF subspace, leaving constrained rows at zero.
    See `modal_eigsolve` for arguments.
    """
    K_csr = scipy_coo_matrix(
        (K._values(), (K._indices()[0], K._indices()[1])), shape=shape
    ).tocsr()
    M_csr = scipy_coo_matrix(
        (M._values(), (M._indices()[0], M._indices()[1])), shape=shape
    ).tocsr()

    fi = free_indices.cpu().numpy()
    eigenvalues, evecs_free = scipy_eigsh(
        K_csr[fi, :][:, fi], k=n_modes, M=M_csr[fi, :][:, fi], sigma=0.0
    )
    eigenvectors = np.zeros((n_dofs, n_modes))
    eigenvectors[fi] = evecs_free
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def _eigsolve_gpu(
    K: Tensor,
    M: Tensor,
    n_modes: int,
    free_indices: Tensor,
    shape: torch.Size,
    n_dofs: int,
) -> tuple[Any, Any]:
    """Solve the generalized eigenproblem on the GPU via CuPy.

    Mirrors `_eigsolve_cpu` and returns CuPy arrays.
    """
    if "cupy" not in available_backends:
        raise RuntimeError(ERR_CUPY_MISSING)
    K_csr = cupy_coo_matrix(
        (
            cupy.asarray(K._values()),
            (cupy.asarray(K._indices()[0]), cupy.asarray(K._indices()[1])),
        ),
        shape=shape,
    ).tocsr()
    M_csr = cupy_coo_matrix(
        (
            cupy.asarray(M._values()),
            (cupy.asarray(M._indices()[0]), cupy.asarray(M._indices()[1])),
        ),
        shape=shape,
    ).tocsr()

    fi = free_indices.cpu().numpy()
    eigenvalues, evecs_free = cupy_eigsh(
        K_csr[fi, :][:, fi], k=n_modes, M=M_csr[fi, :][:, fi], sigma=0.0
    )
    eigenvectors = cupy.zeros((n_dofs, n_modes), dtype=eigenvalues.dtype)
    eigenvectors[fi] = evecs_free
    order = cupy.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def modal_eigsolve(
    K: Tensor,
    M: Tensor,
    n_modes: int,
    free_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Solve the generalized eigenvalue problem `K φ = ω² M φ`.

    Args:
        K (sparse_coo_tensor): Stiffness matrix K.
        M (sparse_coo_tensor): Mass matrix M.
        n_modes (int): Number of eigenpairs to compute.
        free_indices (Tensor): Free DOF indices for subspace extraction.
            The eigenproblem is solved in the free-DOF subspace to avoid
            spurious eigenvalues from the Dirichlet penalty
            (`K_ii = M_ii = 1  =>  ω² = 1`).

    Returns:
        eigenvalues (Tensor): Squared angular frequencies, ascending.
            *Shape:* `(n_modes,)`.
        eigenvectors (Tensor): Mode shapes, constrained rows left at zero.
            *Shape:* `(n_dofs, n_modes)`.
    """
    shape = K.size()
    n_dofs = shape[0]

    if K.device.type == "cuda":
        vals, vecs = _eigsolve_gpu(K, M, n_modes, free_indices, shape, n_dofs)
    else:
        vals, vecs = _eigsolve_cpu(K, M, n_modes, free_indices, shape, n_dofs)

    return (
        torch.tensor(vals, dtype=K.dtype, device=K.device),
        torch.tensor(vecs, dtype=K.dtype, device=K.device),
    )


class Eigensolve(Function):
    """Differentiable eigenvalue solver for K v = ω² M v.

    Gradients are computed via the Rayleigh-quotient sensitivity formula.
    Only eigenvalue gradients are supported; eigenvector gradients are not.
    """

    @staticmethod
    def forward(
        K: Tensor,
        M: Tensor,
        n_modes: int,
        free_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the eigenpairs. See `modal_eigsolve` for the arguments."""
        return modal_eigsolve(K, M, n_modes, free_indices)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        """Save `K`, `M` and the computed eigenpairs for `backward`."""
        K, M, n_modes, free_indices = inputs
        lambdas, phis = output
        ctx.save_for_backward(K, M, lambdas, phis)

    @staticmethod
    def backward(ctx, *grad_outputs) -> tuple:
        """Differentiate the eigenvalues with the Rayleigh-quotient sensitivities.

        For `φ` normalised so that `φ^T M φ = 1`, the sensitivities are
        `dλ/dK_ij = φ_i φ_j` and `dλ/dM_ij = -λ φ_i φ_j`, evaluated only at the
        sparsity pattern of the respective matrix.

        Eigenvector gradients are **not** supported: only the eigenvalue
        gradient is consumed, which is why `differentiable_modal_eigsolve`
        detaches the mode shapes.

        Returns:
            Sparse gradients for `K` and `M`, then `None` for the other arguments.
        """
        grad_lambdas = grad_outputs[0]
        K, M, lambdas, phis = ctx.saved_tensors

        # Re-normalise eigenvectors to unit norm in the M-metric.
        # (eigsh returns M-normalised vectors, but we re-normalise for safety)
        M_phis = torch.sparse.mm(M.coalesce(), phis)  # [n_dofs, n_modes]
        denom = (phis * M_phis).sum(0).abs()  # [n_modes]
        phi_hat = phis / denom.sqrt().unsqueeze(0)  # [n_dofs, n_modes]

        grad_K = None
        grad_M = None

        if grad_lambdas is not None:
            # dL/dK_ij = sum_k dL/dlambda_k * phi_hat_i_k * phi_hat_j_k
            if K.requires_grad:
                idx = K._indices()
                row, col = idx[0], idx[1]
                weighted = phi_hat[row] * phi_hat[col]
                grad_K_vals = (weighted * grad_lambdas.unsqueeze(0)).sum(-1)
                with torch.sparse.check_sparse_tensor_invariants(False):
                    grad_K = torch.sparse_coo_tensor(
                        idx, grad_K_vals, K.shape, is_coalesced=True
                    )

            # dL/dM_ij = -sum_k dL/dlambda_k * lambda_k * phi_hat_i_k * phi_hat_j_k
            if M.requires_grad:
                idx = M._indices()
                row, col = idx[0], idx[1]
                weighted_lam = phi_hat[row] * phi_hat[col]
                grad_M_vals = -(
                    weighted_lam * (lambdas * grad_lambdas).unsqueeze(0)
                ).sum(-1)
                with torch.sparse.check_sparse_tensor_invariants(False):
                    grad_M = torch.sparse_coo_tensor(
                        idx, grad_M_vals, M.shape, is_coalesced=True
                    )

        # Return None for n_modes and free_indices (non-tensor inputs)
        return grad_K, grad_M, None, None


def differentiable_modal_eigsolve(
    K: Tensor,
    M: Tensor,
    n_modes: int,
    free_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Solve the modal eigenvalue problem `K φ = ω² M φ`.

    Args:
        K (sparse_coo_tensor): Stiffness matrix K.
        M (sparse_coo_tensor): Mass matrix M.
        n_modes (int): Number of eigenpairs to compute.
        free_indices (Tensor): Free (unconstrained) DOF indices.
            The eigenproblem is solved in the free-DOF subspace to avoid
            spurious eigenvalues from the Dirichlet penalty (K_ii = M_ii = 1).

    Returns:
        lambdas (Tensor): Squared angular frequencies, differentiable.
            *Shape:* `(n_modes,)`.
        phis (Tensor): Mode shapes, detached since only eigenvalue gradients are
            supported.
            *Shape:* `(n_dofs, n_modes)`.
    """
    lambdas, phis = Eigensolve.apply(K, M, n_modes, free_indices)  # type: ignore
    if lambdas is None:
        raise RuntimeError("Eigensolve.apply returned None.")
    return lambdas, phis.detach()
