from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pyamg
import torch
from scipy.sparse import csc_matrix as scipy_csc_matrix
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix as scipy_csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab as scipy_bicgstab
from scipy.sparse.linalg import cg as scipy_cg
from scipy.sparse.linalg import eigsh as scipy_eigsh
from scipy.sparse.linalg import spsolve as scipy_spsolve
from torch import Tensor
from torch.autograd import Function

if TYPE_CHECKING:
    from .amgx import AmgXSolver
    from .report import SolveReport

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
    import pypardiso

    available_backends.append("pypardiso")
except ImportError:
    pass

try:
    from .amgx import AmgXSolver

    available_backends.append("amgx")
except ImportError:
    pass


def resolve_method(n_dofs: int, method: str | None, symmetric: bool = True) -> str:
    """Return the method `sparse_solve` uses for a system of this size.

    A direct solve for a small system, and beyond it the Krylov method the
    tangent's symmetry allows: `cg` needs a symmetric matrix, `bicgstab` does
    not. Where the crossover really lies depends on the mesh, the device and the
    backend, so the threshold is one compromise for all of them.
    """
    if method is not None:
        return method
    if n_dofs < 10000:
        return "direct"
    return "cg" if symmetric else "bicgstab"


def resolve_preconditioner(method: str, device: str, preconditioner: str | None) -> str:
    """Return the preconditioner `sparse_solve` builds for this method.

    Algebraic multigrid wherever it is available, which is every CPU solve and
    those CUDA solves that AmgX provides for, and a Jacobi diagonal otherwise.

    Raises:
        ValueError: If a preconditioner is requested for a direct solve.
        RuntimeError: If AMG is requested on CUDA without AmgX installed.
    """
    if method == "direct":
        if preconditioner not in (None, "none"):
            raise ValueError("A direct solve takes no preconditioner.")
        return "none"
    if preconditioner is None:
        if device == "cpu" or "amgx" in available_backends:
            return "amg"
        return "jacobi"
    if preconditioner == "amg" and device == "cuda":
        if "amgx" not in available_backends:
            raise RuntimeError(ERR_AMGX_MISSING)
    return preconditioner


def resolve_library(method: str, device: str, preconditioner: str) -> str:
    """Return the library that implements this method and preconditioner.

    AmgX carries its own Krylov solver, so it implements the whole solve rather
    than only the AMG hierarchy it is named for. An iterative solve over a
    diagonal needs nothing but sparse matrix-vector products, which torch has.
    Everything else is SciPy's, on the CPU even where the matrix is not.
    """
    if method != "direct" and preconditioner != "amg":
        return "torch"
    if preconditioner == "amg" and device == "cuda":
        return "amgx"
    if method == "direct" and "pypardiso" in available_backends:
        return "pypardiso"
    return "scipy"


def describe_method(method: str, device: str, preconditioner: str | None) -> str:
    """Describe a resolved solve, for verbose output."""
    p = resolve_preconditioner(method, device, preconditioner)
    kind = "direct" if method == "direct" else f"iterative | {p}"
    return f"{method} | {kind} | {resolve_library(method, device, p)} | {device}"


def _rows(crow: Tensor) -> Tensor:
    """Row index of every stored entry, from the row offsets that omit it."""
    return torch.repeat_interleave(
        torch.arange(crow.numel() - 1, dtype=torch.int32, device=crow.device),
        crow.diff(),
    )


# Iterations between residual tests in `_krylov`, which each cost a host sync.
_CHECK = 10


def _as_csr(A: Tensor, method: str) -> Tensor:
    """Return `A` compressed by row, as every solver here reads it.

    An adjoint passes `K.t()`, compressed by column. `cg` assumes a symmetric
    matrix, so its columns are already the rows wanted and the arrays are reread
    in place; `bicgstab` assumes nothing and pays for a transposed copy.
    """
    if A.layout != torch.sparse_csc:
        return A
    if method != "cg":
        return A.to_sparse_csr()
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_csr_tensor(
            A.ccol_indices(), A.row_indices(), A.values(), size=A.shape
        )


def _diagonal(A: Tensor) -> Tensor:
    """Diagonal of a matrix compressed by row, as a dense vector."""
    col = A.col_indices()
    on_diagonal = _rows(A.crow_indices()).to(col.dtype) == col
    diag = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
    diag[col[on_diagonal].long()] = A.values()[on_diagonal]
    return diag


def _krylov(
    A: Tensor,
    b: Tensor,
    method: str,
    preconditioner: str,
    stol: float,
) -> Tensor:
    """Solve `A x = b` in torch, over a Jacobi diagonal or nothing.

    The diagonal is rebuilt per solve rather than carried between them, costing
    a fraction of one iteration. Testing the residual every `_CHECK` iterations
    rather than every one synchronises with the host that much less often, at
    the cost of overshooting the tolerance by up to `_CHECK - 1` iterations.

    Raises:
        RuntimeError: If the iteration limit is reached before `stol`.
    """
    A = _as_csr(A, method)
    M = 1.0 / _diagonal(A) if preconditioner == "jacobi" else None

    def precondition(v: Tensor) -> Tensor:
        return v if M is None else v * M

    x = torch.zeros_like(b)
    r = b.clone()
    threshold = stol * stol * torch.dot(b, b)

    if method == "cg":
        z = precondition(r)
        p = z.clone()
        rz = torch.dot(r, z)
        for i in range(10 * len(b)):
            Ap = A @ p
            alpha = rz / torch.dot(p, Ap)
            x.add_(p, alpha=alpha)  # type: ignore[arg-type]
            r.sub_(Ap, alpha=alpha)  # type: ignore[arg-type]
            z = precondition(r)
            rz_next = torch.dot(r, z)
            if i % _CHECK == _CHECK - 1 and torch.dot(r, r) <= threshold:
                return x
            p.mul_(rz_next / rz).add_(z)
            rz = rz_next
    else:
        # BiCGStab, with the shadow residual fixed at the initial one
        r0 = r.clone()
        p = torch.zeros_like(b)
        v = torch.zeros_like(b)
        rho = alpha = omega = torch.ones((), dtype=b.dtype, device=b.device)
        for i in range(10 * len(b)):
            rho_next = torch.dot(r0, r)
            p = r + (rho_next / rho) * (alpha / omega) * (p - omega * v)
            rho = rho_next
            y = precondition(p)
            v = A @ y
            alpha = rho / torch.dot(r0, v)
            s = r - alpha * v
            zs = precondition(s)
            t = A @ zs
            omega = torch.dot(t, s) / torch.dot(t, t)
            x = x + alpha * y + omega * zs
            r = s - omega * t
            if i % _CHECK == _CHECK - 1 and torch.dot(r, r) <= threshold:
                return x

    raise RuntimeError(f"{method} did not reach {stol:g} within the iteration limit.")


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
        nodes: Tensor | None = None,
        stol: float = 1e-10,
        device: str | None = None,
        method: str | None = None,
        preconditioner: str | None = None,
        M: LinearOperator | None = None,
    ) -> tuple[Tensor, LinearOperator | None]:
        """Solve `A x = b`.

        See `sparse_solve` for the arguments.

        Returns:
            x (Tensor): Solution vector.
                *Shape:* `(n_dofs,)`.
            M (LinearOperator | None): Preconditioner built or reused by the
                solve, passed on to `backward` for the adjoint system.
        """
        x, M, solver = sparse_solve(
            A, b, B, nodes, stol, device, method, preconditioner, M
        )

        # An AmgX solver built here is freed here, and `backward` builds its
        # own: nothing else frees one. See `torchfem.amgx`.
        if solver is not None:
            solver.close()

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

        # Adjoint solve: A^T lambda = grad_x, where `A.t()` is a CSC view of the
        # same arrays rather than a transposed copy.
        gradb, _, solver = sparse_solve(
            A.t(),
            grad_x,
            ctx.B,
            ctx.nodes,
            ctx.stol,
            ctx.device,
            ctx.method,
            ctx.pre,
            ctx.M,
        )

        # An AmgX solver built for this adjoint does not outlive it either
        if solver is not None:
            solver.close()

        # Backprop rule: gradA = -gradb @ x^T, sparse version. The gradient
        # reuses the index arrays of `A`.
        crow, col = A.crow_indices(), A.col_indices()
        row = _rows(crow)
        val = -gradb[row] * x[col]
        with torch.sparse.check_sparse_tensor_invariants(False):
            gradA = torch.sparse_csr_tensor(crow, col, val, size=A.shape)

        return gradA, gradb, None, None, None, None, None, None, None

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        """Save `A`, `x` and the solver settings for `backward`.

        Stores the preconditioner *returned* by the forward pass, not the one
        passed in, so the adjoint solve reuses the AMG hierarchy built there.
        """
        A, b, B, nodes, stol, device, method, preconditioner, M = inputs
        x, M_computed = output
        ctx.save_for_backward(A, x)

        # Save the parameters for backward pass (including the preconditioner)
        ctx.B = None if B is None else B.detach()
        ctx.nodes = None if nodes is None else nodes.detach()
        ctx.stol = stol
        ctx.device = device
        ctx.method = method
        ctx.pre = preconditioner
        ctx.M = M_computed


def differentiable_sparse_solve(
    A: Tensor,
    b: Tensor,
    B: Tensor | None = None,
    nodes: Tensor | None = None,
    stol: float = 1e-10,
    device: str | None = None,
    method: str | None = None,
    preconditioner: str | None = None,
    M: LinearOperator | None = None,
) -> Tensor:
    """Solve `A x = b` with custom sparse adjoint autograd support.

    The forward pass may use non-differentiable sparse backends (SciPy,
    Pardiso, AmgX). The backward pass solves the adjoint system and returns gradients
    with respect to both `A` and `b`.
    """
    # Compressed here, not inside `Solve`, so the adjoint fills one layout and a
    # COO caller still gets its gradient back through this conversion.
    if A.layout == torch.sparse_coo:
        A = A.to_sparse_csr()
    result, _ = Solve.apply(  # type: ignore
        A, b, B, nodes, stol, device, method, preconditioner, M
    )
    if result is None:
        raise RuntimeError("Solve.apply returned None, expected a Tensor.")
    return result


def sparse_solve(
    A: Tensor,
    b: Tensor,
    B: Tensor | None = None,
    nodes: Tensor | None = None,
    stol: float = 1e-10,
    device: str | None = None,
    method: str | None = None,
    preconditioner: str | None = None,
    M: LinearOperator | None = None,
    solver: AmgXSolver | None = None,
) -> tuple[Tensor, LinearOperator | None, AmgXSolver | None]:
    """
    Solve the linear system Ax = b.

    Args:
        A (sparse_csr_tensor): Sparse matrix A, compressed by row. A `t()` of
            one, which is compressed by column, is accepted as its transpose.
        b (Tensor): Right-hand side vector b.
        B (Tensor, optional): Null space rigid body modes for AMG preconditioner.
        nodes (Tensor, optional): Nodal coordinates, which AmgX aggregates over
            instead of a null space. Their count gives the degrees of freedom
            per node, so they must describe every row of `A` in nodal blocks.
            Defaults to None, where AmgX aggregates single rows.
        stol (float, optional): Relative solver tolerance for the iterative solver.
            Defaults to 1e-10.
        device (str, optional): Device to run the computation on ('cpu' or 'cuda').
            Defaults to None, which uses the current device.
        method (str, optional): Method to use for solving ('direct', 'cg' or
            'bicgstab'). Defaults to None for `resolve_method` to choose by size.
            'cg' needs a symmetric positive definite matrix, 'bicgstab' needs
            neither.
        preconditioner (str, optional): Preconditioner to build for an iterative
            method ('amg', 'jacobi' or 'none'). Defaults to None for
            `resolve_preconditioner` to choose by device and available backends.
        M (LinearOperator, optional): Preconditioner for iterative methods.
            Defaults to None, which builds one.
        solver (AmgXSolver, optional): Solver to reuse for method='amgx'.
            Defaults to None, which builds one. Kept apart from `M` because it
            is freed by `close()` rather than by the garbage collector, so it
            must not outlive the loop that owns it.

    Returns:
        x (Tensor): Solution vector.
            *Shape:* `(n_dofs,)`.
        M (LinearOperator | None): Preconditioner built or reused by the solve,
            `None` for direct methods and for 'amgx'.
        solver (AmgXSolver | None): Solver built or reused by the solve, `None`
            for every method but 'amgx'. Holds its hierarchy until `close()`.
    """
    # Check the input shape
    if A.ndim != 2 or (A.shape[0] != A.shape[1]):
        raise ValueError("A should be a square 2D matrix.")
    out_device = b.device

    # Check the input method
    if method is not None and method not in ["direct", "cg", "bicgstab"]:
        raise ValueError(
            f"Method {method} is not supported. "
            "Choose from 'direct', 'cg', or 'bicgstab'."
        )
    if preconditioner is not None and preconditioner not in ["amg", "jacobi", "none"]:
        raise ValueError(
            f"Preconditioner {preconditioner} is not supported. "
            "Choose from 'amg', 'jacobi', or 'none'."
        )

    # Move to requested device, if available
    if device is not None:
        A = A.to(device)
        b = b.to(device)
        if B is not None:
            B = B.to(device)

    # Make default solver choice based on shape and available backends
    method = resolve_method(A.shape[0], method)
    preconditioner = resolve_preconditioner(method, A.device.type, preconditioner)

    # Only a hierarchy or a factorisation is worth another library's arrays.
    if method != "direct" and preconditioner != "amg":
        x = _krylov(A, b, method, preconditioner, stol)
        return x.to(out_device), M, solver

    # An AMG hierarchy on CUDA is AmgX, which runs the whole solve.
    if preconditioner == "amg" and A.device.type == "cuda":
        x, solver = _solve_amgx(A, b, nodes, method, stol, solver)
        return x.to(out_device), M, solver

    # SciPy takes the rest, CUDA data included: cuSOLVER's sparse LU is slower
    # than SuperLU even counting the transfer either way.
    x_np, M = _solve_cpu(
        A.cpu(), b.cpu(), None if B is None else B.cpu(), method, stol, M
    )
    return torch.tensor(x_np, dtype=b.dtype, device=out_device), M, solver


def _solve_amgx(
    A: Tensor,
    b: Tensor,
    nodes: Tensor | None,
    method: str,
    stol: float,
    solver: AmgXSolver | None,
) -> tuple[Tensor, AmgXSolver]:
    """Solve `A x = b` on the GPU via AmgX. See `sparse_solve` for arguments.

    AmgX brings its own Krylov solver and so runs the whole solve, reusing
    `solver` to refresh coefficients instead of building a fresh hierarchy, and
    taking its nodal blocks from `nodes`. It reads torch's arrays where they lie.
    """
    if "amgx" not in available_backends:
        raise RuntimeError(ERR_AMGX_MISSING)

    # AmgX allocates its hierarchy from what torch is not holding.
    torch.cuda.empty_cache()
    A = _as_csr(A, method)
    if solver is None:
        block_size = 1
        coords = None
        if nodes is not None:
            dofs = A.shape[0] // len(nodes)
            # AmgX aggregates blocks of at most 5, so a shell's 6 DOFs split
            # into a translational and a rotational block of 3.
            block_size = 3 if dofs == 6 else dofs
            # Geometry needs one coordinate triple per block row.
            if nodes.shape[1] == 3 and dofs == 3:
                x, y, z = (
                    np.ascontiguousarray(c.detach().cpu().numpy()) for c in nodes.t()
                )
                coords = x, y, z
        krylov = "PCG" if method == "cg" else "PBICGSTAB"
        solver = AmgXSolver(A.shape[0], stol, block_size, coords, krylov)
        solver.setup(A)
    else:
        solver.resetup(A)
    return solver.solve(b.contiguous()), solver


def _solve_cpu(
    A: Tensor,
    b: Tensor,
    B: Tensor | None,
    method: str,
    stol: float,
    M: LinearOperator | None,
) -> tuple[Any, LinearOperator | None]:
    """Solve `A x = b` on the CPU via SciPy. See `sparse_solve` for arguments.

    Takes the solves that need a library: a direct one, which goes to Pardiso
    where it is installed, reordering with reverse Cuthill-McKee before
    factorising, and to SuperLU otherwise; and an AMG-preconditioned one, whose
    hierarchy is built by pyamg from `A` and `B` unless `M` is supplied.
    """
    # SciPy takes the arrays as they are. An adjoint passes `K.t()`, the same
    # arrays read as CSC.
    if A.layout == torch.sparse_csc:
        A_np = scipy_csc_matrix(
            (A.values(), A.row_indices(), A.ccol_indices()), shape=A.shape
        )
    else:
        A_np = scipy_csr_matrix(
            (A.values(), A.col_indices(), A.crow_indices()), shape=A.shape
        )
    b_np = b.data.numpy()

    if method == "direct":
        if "pypardiso" not in available_backends:
            return scipy_spsolve(A_np, b_np), None
        # Reorder the matrix using reverse Cuthill-McKee algorithm
        rcm_order = csgraph.reverse_cuthill_mckee(A_np)
        A_rcm = A_np[np.ix_(rcm_order, rcm_order)]
        x_rcm = pypardiso.spsolve(A_rcm, b_np[rcm_order])
        # Restore the original order
        return x_rcm[np.argsort(rcm_order)], None

    if M is None:
        # AMG preconditioner with Jacobi smoother
        B_np = None if B is None else B.data.numpy()
        ml = pyamg.smoothed_aggregation_solver(A_np, B_np, smooth="jacobi")
        M = ml.aspreconditioner()

    solve = scipy_cg if method == "cg" else scipy_bicgstab
    x_xp, exit_code = solve(A_np, b_np, M=M, rtol=stol)
    if exit_code != 0:
        raise RuntimeError(f"{method} failed with exit code {exit_code}")

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
        nodes: Tensor | None,
        max_iter: int,
        rtol: float,
        atol: float,
        stol: float,
        report: SolveReport | None,
        method: str | None = None,
        preconditioner: str | None = None,
        device: str | None = None,
        u_prev: Tensor | None = None,
        grad_prev: Tensor | None = None,
        flux_prev: Tensor | None = None,
        state_prev: Tensor | None = None,
        *parameters: Tensor,
    ) -> Tensor:
        """Run Newton iterations until the residual meets `rtol` or `atol`.

        Only the converged state is saved for backward. See `newton_solve` for
        arguments.

        Raises:
            RuntimeError: If the residual becomes NaN or infinite, or the
                iteration limit is reached.
        """
        M = None
        solver = None
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

            # A residual that is not finite never converges
            if not torch.isfinite(res_norm):
                break

            # Solve for displacement increment
            du_i, M, solver = sparse_solve(
                K, residual, B, nodes, stol, device, method, preconditioner, M, solver
            )

            du = du - du_i

        # An AmgX solver is freed only by `close()` and does not outlive the
        # call that built it: `backward` builds its own, and a solve that fails
        # closes itself. See `torchfem.amgx`.
        if solver is not None:
            solver.close()

        # Final convergence check, which a residual that is not finite fails
        if not (res_norm < rtol * res_norm0 or res_norm < atol):
            raise RuntimeError("Newton-Raphson iteration did not converge.")

        ctx.save_for_backward(
            K, du, u_prev, grad_prev, flux_prev, state_prev, *parameters
        )
        ctx.B = B.detach()
        ctx.nodes = None if nodes is None else nodes.detach()
        ctx.M = M
        ctx.stol = stol
        ctx.device = device
        ctx.method = method
        ctx.pre = preconditioner
        ctx.eval_residual = eval_residual
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
        nodes = ctx.nodes
        M = ctx.M
        stol = ctx.stol
        device = ctx.device
        method = ctx.method
        preconditioner = ctx.pre
        eval_residual = ctx.eval_residual
        converged_iter = ctx.converged_iter

        # Solve adjoint system, where `K.t()` is a CSC view of the same arrays.
        lambda_, _, solver = sparse_solve(
            K.t(), grad_du, B, nodes, stol, device, method, preconditioner, M
        )

        # An AmgX solver built for this adjoint does not outlive it either
        if solver is not None:
            solver.close()

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
    nodes: Tensor | None,
    max_iter: int,
    rtol: float,
    atol: float,
    stol: float,
    report: SolveReport | None,
    method: str | None = None,
    preconditioner: str | None = None,
    device: str | None = None,
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
        nodes: Nodal coordinates for AmgX, or None where the rows carry no
            nodal blocking.
        max_iter: Maximum Newton iterations.
        rtol: Relative residual tolerance.
        atol: Absolute residual tolerance.
        stol: Linear-solver tolerance used inside Newton steps for iterative solvers.
        report: Optional progress report receiving the iteration residuals.
        method: Sparse backend method name.
        preconditioner: Sparse backend preconditioner name.
        device: Optional sparse backend device hint.
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
        nodes,
        max_iter,
        rtol,
        atol,
        stol,
        report,
        method,
        preconditioner,
        device,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigenproblem via SciPy, wherever the matrices live.

    ARPACK is the only shift-invert this depends on, so a CUDA model pays the
    transfer and solves here. Shift-invert at `sigma=0.0` targets the lowest
    modes, and eigenvectors are scattered back from the free-DOF subspace,
    leaving constrained rows at zero. See `modal_eigsolve` for arguments.
    """
    K_csr = scipy_csr_matrix(
        (K.values(), K.col_indices(), K.crow_indices()), shape=K.shape
    )
    M_csr = scipy_csr_matrix(
        (M.values(), M.col_indices(), M.crow_indices()), shape=K.shape
    )

    fi = free_indices.cpu().numpy()
    eigenvalues, evecs_free = scipy_eigsh(
        K_csr[fi, :][:, fi], k=n_modes, M=M_csr[fi, :][:, fi], sigma=0.0
    )
    eigenvectors = np.zeros((K.shape[0], n_modes))
    eigenvectors[fi] = evecs_free
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def modal_eigsolve(
    K: Tensor,
    M: Tensor,
    n_modes: int,
    free_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Solve the generalized eigenvalue problem `K φ = ω² M φ`.

    Args:
        K (sparse_csr_tensor): Stiffness matrix K.
        M (sparse_csr_tensor): Mass matrix M.
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
    vals, vecs = _eigsolve_cpu(K.cpu(), M.cpu(), n_modes, free_indices)

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
        M_phis = torch.sparse.mm(M, phis)  # [n_dofs, n_modes]
        denom = (phis * M_phis).sum(0).abs()  # [n_modes]
        phi_hat = phis / denom.sqrt().unsqueeze(0)  # [n_dofs, n_modes]

        grad_K = None
        grad_M = None

        if grad_lambdas is not None:
            # dL/dK_ij = sum_k dL/dlambda_k * phi_hat_i_k * phi_hat_j_k
            if K.requires_grad:
                crow, col = K.crow_indices(), K.col_indices()
                row = _rows(crow)
                weighted = phi_hat[row] * phi_hat[col]
                grad_K_vals = (weighted * grad_lambdas.unsqueeze(0)).sum(-1)
                with torch.sparse.check_sparse_tensor_invariants(False):
                    grad_K = torch.sparse_csr_tensor(
                        crow, col, grad_K_vals, size=K.shape
                    )

            # dL/dM_ij = -sum_k dL/dlambda_k * lambda_k * phi_hat_i_k * phi_hat_j_k
            if M.requires_grad:
                crow, col = M.crow_indices(), M.col_indices()
                row = _rows(crow)
                weighted_lam = phi_hat[row] * phi_hat[col]
                grad_M_vals = -(
                    weighted_lam * (lambdas * grad_lambdas).unsqueeze(0)
                ).sum(-1)
                with torch.sparse.check_sparse_tensor_invariants(False):
                    grad_M = torch.sparse_csr_tensor(
                        crow, col, grad_M_vals, size=M.shape
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
        K (sparse_csr_tensor): Stiffness matrix K.
        M (sparse_csr_tensor): Mass matrix M.
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
