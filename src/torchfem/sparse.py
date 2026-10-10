from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pyamg
import torch
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix as scipy_csr_matrix
from scipy.sparse.linalg import eigsh as scipy_eigsh
from scipy.sparse.linalg import spsolve as scipy_spsolve
from torch import Tensor
from torch.autograd import Function

if TYPE_CHECKING:
    from .amgx import AmgXSolver
    from .report import SolveReport

ERR_AMGX_MISSING = (
    "AmgX is not available.\n\n"
    "AmgX ships no wheels, so build it from source and point AMGX_DLL at the "
    "shared library:\n"
    "> export AMGX_DLL=/path/to/libamgxsh.so # amgxsh.dll on Windows"
)


class ConvergenceError(RuntimeError):
    """A Newton or linear solve stalled short of its tolerance.

    Its own type, so an increment cutback retries this and nothing else.
    """


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
    tangent's symmetry allows. Where the crossover lies depends on the mesh, the
    device and the backend, so the threshold is one compromise for all of them.
    """
    if method is not None:
        return method
    if n_dofs < 10000:
        return "direct"
    return "cg" if symmetric else "bicgstab"


def resolve_preconditioner(method: str, device: str, preconditioner: str | None) -> str:
    """Return the preconditioner `sparse_solve` builds for this method.

    Algebraic multigrid wherever it is available, and a Jacobi diagonal
    otherwise.

    Raises:
        ValueError: If a preconditioner is requested for a direct solve.
        ImportError: If AMG is requested on CUDA without AmgX installed.
    """
    if method == "direct":
        if preconditioner not in (None, "none"):
            raise ValueError("A direct solve takes no preconditioner.")
        return "none"
    if preconditioner is None:
        return "amg" if device == "cpu" or "amgx" in available_backends else "jacobi"
    if preconditioner == "amg" and device == "cuda":
        if "amgx" not in available_backends:
            raise ImportError(ERR_AMGX_MISSING)
    return preconditioner


def resolve_library(method: str, device: str, preconditioner: str) -> str:
    """Return the library that implements this method and preconditioner.

    An iterative solve needs nothing but sparse matrix-vector products, which
    torch has. AmgX carries its own Krylov solver, so it takes the whole solve
    on CUDA rather than only the hierarchy it is named for. A direct solve is
    SciPy's or Pardiso's, on the CPU even where the matrix is not.
    """
    if preconditioner == "amg" and device == "cuda":
        return "amgx"
    if method != "direct":
        return "torch"
    if "pypardiso" in available_backends:
        return "pypardiso"
    return "scipy"


def describe_method(method: str, device: str, preconditioner: str | None) -> str:
    """Describe a resolved solve, for verbose output."""
    p = resolve_preconditioner(method, device, preconditioner)
    named = [method] if method == "direct" else [method, "iterative", p]
    return " | ".join([*named, resolve_library(method, device, p), device])


def _rows(crow: Tensor) -> Tensor:
    """Row index of every stored entry, from the row offsets that omit it."""
    return torch.repeat_interleave(
        torch.arange(crow.numel() - 1, dtype=torch.int32, device=crow.device),
        crow.diff(),
    )


def _to_scipy(A: Tensor) -> Any:
    """Return `A` as a SciPy matrix compressed by row."""
    return scipy_csr_matrix(
        (A.values().numpy(), A.col_indices().numpy(), A.crow_indices().numpy()),
        shape=A.shape,
    )


def _nodal(B: Tensor | None, n: int) -> tuple[int, tuple[Any, ...] | None]:
    """Nodal blocking and coordinates for AmgX, from the rigid body modes.

    A translation mode holds a one at each node's first degree of freedom, so an
    evenly strided pattern gives away how many a node carries, and the rotation
    modes hold the coordinates themselves. Rows in no such blocking, as an
    eliminated assembly's are, get neither.
    """
    if B is None or not len(B):
        return 1, None
    base = (B[:, 0] == 1.0).nonzero().ravel()
    dofs = n // len(base) if len(base) else 0
    if not dofs or len(base) * dofs != n:
        return 1, None
    if not torch.equal(base, torch.arange(len(base), device=base.device) * dofs):
        return 1, None
    # AmgX aggregates blocks of at most five, so a shell's six degrees of
    # freedom split into a translational and a rotational block of three.
    block_size = 3 if dofs == 6 else dofs
    # `GEO` needs one coordinate per block row, which a split node denies it.
    if dofs != block_size:
        return block_size, None
    # A rotation about z moves x by -y and y by x, which gives the coordinates.
    if B.shape[1] == 3:  # two translations and a rotation about z
        cols = B[base + 1, 2], -B[base, 2]
    elif B.shape[1] == 6:  # three of each
        cols = B[base + 1, 5], B[base + 2, 3], B[base, 4]
    else:
        return block_size, None
    return block_size, tuple(
        np.ascontiguousarray(c.detach().cpu().numpy()) for c in cols
    )


def _diagonal(A: Tensor) -> Tensor:
    """Diagonal of a matrix compressed by row, as a dense vector."""
    col = A.col_indices()
    on_diagonal = _rows(A.crow_indices()).to(col.dtype) == col
    diag = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
    diag[col[on_diagonal].long()] = A.values()[on_diagonal]
    return diag


def _multigrid(A: Tensor, B: Tensor | None) -> Callable[[Tensor], Tensor]:
    """Return a smoothed aggregation V-cycle, built by pyamg and applied in torch.

    The cycle is nearly all of a solve, so it runs on torch's threaded sparse
    products, with Chebyshev smoothing either side: symmetric, as CG needs, and
    parallel, as pyamg's Gauss-Seidel is not. Its interval comes from `ρ(D⁻¹A)`
    by power iteration, seeded for reproducible solves. The coarsest level of up
    to 2000 unknowns is solved exactly, where a further level would lose the
    modes that a barely supported body depends on.
    """
    B_np = None if B is None else B.detach().numpy()
    # pyamg counts `max_coarse` in aggregates, each of one unknown per mode in `B`
    max_coarse = 2000 // (1 if B is None else B.shape[1])
    ml = pyamg.smoothed_aggregation_solver(
        _to_scipy(A), B_np, strength=None, smooth="jacobi", max_coarse=max_coarse
    )
    levels = []
    seed = torch.Generator().manual_seed(0)
    for level in ml.levels[:-1]:
        # Sharing SciPy's arrays, so the finest level is `A` itself
        with torch.sparse.check_sparse_tensor_invariants(False):
            A_i, P, R = (
                torch.sparse_csr_tensor(
                    *map(torch.from_numpy, (S.indptr, S.indices, S.data)), size=S.shape
                )
                for S in (level.A.tocsr(), level.P.tocsr(), level.R.tocsr())
            )
        D_inv = torch.from_numpy(1.0 / level.A.diagonal())
        # Spectral radius of D⁻¹A by power iteration
        v = torch.rand(A_i.shape[0], dtype=A_i.dtype, generator=seed)
        for _ in range(15):
            v = D_inv * (A_i @ v)
            v /= v.norm()
        # Degree 2 over [hi / 10, hi] (Saad, Algorithm 12.1), with `hi` above
        # `ρ`, which power iteration estimates from below
        hi = 1.1 * (v @ (D_inv * (A_i @ v)))
        theta, delta = (hi + hi / 10) / 2, (hi - hi / 10) / 2
        r0 = delta / theta
        r1 = 1 / (2 / r0 - r0)
        levels.append((A_i, P, R, D_inv / theta, r1 * r0, 2 * r1 / delta * D_inv))
    # Cholesky where the coarsest operator is definite, as `R A P` of a definite `A`
    C = torch.from_numpy(ml.levels[-1].A.toarray())
    L, info = torch.linalg.cholesky_ex(C)
    coarse = torch.linalg.pinv(C, hermitian=True) if info else torch.cholesky_inverse(L)

    def smooth(i: int, r: Tensor, x: Tensor | None) -> Tensor:
        A, _, _, w0, c, w1 = levels[i]
        d = w0 * (r if x is None else r - A @ x)
        x = d if x is None else x + d
        return x + c * d + w1 * (r - A @ x)

    def cycle(r: Tensor, i: int = 0) -> Tensor:
        if i == len(levels):
            return coarse @ r
        A, P, R = levels[i][:3]
        x = smooth(i, r, None)
        x = x + P @ cycle(R @ (r - A @ x), i + 1)
        return smooth(i, r, x)

    return cycle


def _krylov(
    A: Tensor, b: Tensor, method: str, stol: float, M: Callable | None
) -> Tensor:
    """Solve `A x = b` in torch, preconditioned by `M` or nothing.

    Testing the residual every 10 iterations on CUDA rather than every one
    synchronises with the host that much less often, at the cost of overshooting
    the tolerance by up to 9 iterations.

    Raises:
        ConvergenceError: If the residual stops short of `stol` or is not finite.
    """
    check = 10 if A.is_cuda else 1
    x = torch.zeros_like(b)
    r = b.clone()
    threshold = stol * stol * torch.dot(b, b)
    # Zero solves to zero, and would divide by a residual that never shrinks.
    if not torch.any(b):
        return x

    if method == "cg":
        z = r if M is None else M(r)
        p = z.clone()
        rz = torch.dot(r, z)
        for i in range(10 * len(b)):
            Ap = A @ p
            alpha = rz / torch.dot(p, Ap)
            x.add_(p, alpha=alpha)  # type: ignore[arg-type]
            r.sub_(Ap, alpha=alpha)  # type: ignore[arg-type]
            # `not >` also stops at a residual that is no longer finite, to fail.
            if i % check == check - 1 and not torch.dot(r, r) > threshold:
                break
            z = r if M is None else M(r)
            rz_next = torch.dot(r, z)
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
            y = p if M is None else M(p)
            v = A @ y
            alpha = rho / torch.dot(r0, v)
            s = r - alpha * v
            zs = s if M is None else M(s)
            t = A @ zs
            omega = torch.dot(t, s) / torch.dot(t, t)
            x = x + alpha * y + omega * zs
            r = s - omega * t
            if i % check == check - 1 and not torch.dot(r, r) > threshold:
                break

    if torch.dot(r, r) <= threshold:
        return x
    raise ConvergenceError(f"{method} did not reach {stol:g}.")


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
        preconditioner: str | None = None,
        M: Callable | None = None,
    ) -> tuple[Tensor, Callable | None]:
        """Solve `A x = b`.

        See `sparse_solve` for the arguments.

        Returns:
            x (Tensor): Solution vector.
                *Shape:* `(n_dofs,)`.
            M (Callable | None): Preconditioner built or reused by the
                solve, passed on to `backward` for the adjoint system.
        """
        x, M, solver = sparse_solve(A, b, B, stol, device, method, preconditioner, M)

        # A solver is freed by the solve that built it. See `torchfem.amgx`.
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
        A, x = ctx.saved_tensors

        # `A.t()` is a CSC view of the same arrays, not a transposed copy.
        gradb, _, solver = sparse_solve(
            A.t(),
            grad_outputs[0],
            ctx.B,
            ctx.stol,
            ctx.device,
            ctx.method,
            ctx.pre,
            ctx.M,
        )
        if solver is not None:
            solver.close()

        # gradA = -gradb x^T, over the index arrays of `A`
        crow, col = A.crow_indices(), A.col_indices()
        row = _rows(crow)
        val = -gradb[row] * x[col]
        with torch.sparse.check_sparse_tensor_invariants(False):
            gradA = torch.sparse_csr_tensor(crow, col, val, size=A.shape)

        return gradA, gradb, None, None, None, None, None, None

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        """Save `A`, `x` and the solver settings for `backward`.

        Stores the preconditioner *returned* by the forward pass, not the one
        passed in, so the adjoint solve reuses the AMG hierarchy built there.
        """
        A, b, B, stol, device, method, preconditioner, M = inputs
        x, M_computed = output
        ctx.save_for_backward(A, x)

        # Save the parameters for backward pass (including the preconditioner)
        ctx.B = None if B is None else B.detach()
        ctx.stol = stol
        ctx.device = device
        ctx.method = method
        ctx.pre = preconditioner
        ctx.M = M_computed


def differentiable_sparse_solve(
    A: Tensor,
    b: Tensor,
    B: Tensor | None = None,
    stol: float = 1e-10,
    device: str | None = None,
    method: str | None = None,
    preconditioner: str | None = None,
    M: Callable | None = None,
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
        A, b, B, stol, device, method, preconditioner, M
    )
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
    preconditioner: str | None = None,
    M: Callable | None = None,
    solver: AmgXSolver | None = None,
) -> tuple[Tensor, Callable | None, AmgXSolver | None]:
    """
    Solve the linear system Ax = b.

    Args:
        A (sparse_csr_tensor): Sparse matrix A, compressed by row. A `t()` of
            one, which is compressed by column, is accepted as its transpose.
        b (Tensor): Right-hand side vector b.
        B (Tensor, optional): Null space rigid body modes for AMG preconditioner.
        stol (float, optional): Relative solver tolerance for the iterative solver.
            Defaults to 1e-10.
        device (str, optional): Device to run the computation on ('cpu' or 'cuda').
            Defaults to None, which uses the current device.
        method (str, optional): Method to use for solving ('direct', 'cg' or
            'bicgstab'). Defaults to None for `resolve_method` to choose by size,
            assuming symmetry: name 'bicgstab' for a matrix without it, since a
            model resolves the method from its own tangent before calling here.
            'cg' needs a symmetric positive definite matrix, 'bicgstab' needs
            neither.
        preconditioner (str, optional): Preconditioner to build for an iterative
            method ('amg', 'jacobi' or 'none'). Defaults to None for
            `resolve_preconditioner` to choose by device and available backends.
        M (Callable, optional): Preconditioner for iterative methods.
            Defaults to None, which builds one.
        solver (AmgXSolver, optional): Solver to reuse where AmgX runs the solve.
            Defaults to None, which builds one. Kept apart from `M` because it
            is freed by `close()` rather than by the garbage collector, so it
            must not outlive the loop that owns it.

    Returns:
        x (Tensor): Solution vector.
            *Shape:* `(n_dofs,)`.
        M (Callable | None): Preconditioner built or reused by the solve,
            `None` for a direct solve and wherever AmgX runs it.
        solver (AmgXSolver | None): Solver built or reused by the solve, `None`
            unless AmgX ran it. Holds its hierarchy until `close()`.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A should be a square 2D matrix.")
    for name, value, choices in (
        ("Method", method, ("direct", "cg", "bicgstab")),
        ("Preconditioner", preconditioner, ("amg", "jacobi", "none")),
    ):
        if value is not None and value not in choices:
            raise ValueError(f"{name} {value} is not supported. Choose from {choices}.")

    out_device = b.device
    if device is not None:
        A = A.to(device)
        b = b.to(device)
        if B is not None:
            B = B.to(device)

    method = resolve_method(A.shape[0], method)
    preconditioner = resolve_preconditioner(method, A.device.type, preconditioner)
    library = resolve_library(method, A.device.type, preconditioner)
    # `K.t()` from an adjoint is by column. `cg` undoes it, as `K` is symmetric.
    if A.layout == torch.sparse_csc:
        A = A.t() if method == "cg" else A.to_sparse_csr()

    if library == "torch":
        # A diagonal costs a fraction of an iteration, so it is rebuilt per solve.
        if preconditioner != "amg":
            M = _diagonal(A).reciprocal().mul if preconditioner == "jacobi" else None
        elif M is None:
            M = _multigrid(A, B)
        return _krylov(A, b, method, stol, M).to(out_device), M, solver

    if library == "amgx":
        # AmgX runs the whole solve on torch's arrays where they lie, allocating
        # from what torch is not holding, and refreshes `solver` when it is given.
        torch.cuda.empty_cache()
        if solver is None:
            block_size, coords = _nodal(B, A.shape[0])
            krylov = "PCG" if method == "cg" else "PBICGSTAB"
            solver = AmgXSolver(A.shape[0], stol, block_size, coords, krylov)
            solver.setup(A)
        else:
            solver.resetup(A)
        return solver.solve(b.contiguous()).to(out_device), M, solver

    # SciPy and Pardiso read host arrays, CUDA data included: cuSOLVER's sparse
    # LU is slower than SuperLU even counting the transfer either way.
    A_np, b_np = _to_scipy(A.cpu()), b.data.cpu().numpy()
    if "pypardiso" in available_backends:
        order = csgraph.reverse_cuthill_mckee(A_np)
        x = pypardiso.spsolve(A_np[np.ix_(order, order)], b_np[order])
        x = x[np.argsort(order)]
    else:
        x = scipy_spsolve(A_np, b_np)
    return torch.tensor(x, dtype=b.dtype, device=out_device), None, solver


class NewtonRaphsonAdjoint(Function):
    """Custom autograd function for nonlinear Newton-Raphson solves.

    The forward pass performs Newton iterations on the residual callback
    `eval_residual(du, iter, (u_prev, grad_prev, flux_prev, state_prev)) ->
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
            ConvergenceError: If the residual becomes NaN or infinite, or the
                iteration limit is reached.
        """
        M = None
        solver = None
        converged_iter = max_iter - 1

        # Newton-Raphson iterations
        for i in range(max_iter):
            # Evaluate residual, stiffness matrix, and internal forces
            prev = (u_prev, grad_prev, flux_prev, state_prev)
            residual, K = eval_residual(du, i, prev)

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
                K, residual, B, stol, device, method, preconditioner, M, solver
            )

            du = du - du_i

        # A solver is freed by the loop that built it. See `torchfem.amgx`.
        if solver is not None:
            solver.close()

        # Final convergence check, which a residual that is not finite fails
        if not (res_norm < rtol * res_norm0 or res_norm < atol):
            raise ConvergenceError("Newton-Raphson iteration did not converge.")

        ctx.save_for_backward(
            K, du, u_prev, grad_prev, flux_prev, state_prev, *parameters
        )
        ctx.B = B.detach()
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
        K, du, u_prev, grad_prev, flux_prev, state_prev, *parameters = ctx.saved_tensors
        prev = (u_prev, grad_prev, flux_prev, state_prev)

        # `K.t()` is a CSC view of the same arrays, not a transposed copy.
        lambda_, _, solver = sparse_solve(
            K.t(),
            grad_outputs[0],
            ctx.B,
            ctx.stol,
            ctx.device,
            ctx.method,
            ctx.pre,
            ctx.M,
        )
        if solver is not None:
            solver.close()

        du_local = du.detach().requires_grad_(True)
        prev_local = tuple(p.detach().requires_grad_(True) for p in prev)
        with torch.enable_grad(), torch.device(du_local.device):
            residual, _ = ctx.eval_residual(du_local, ctx.converged_iter, prev_local)

        grads = torch.autograd.grad(
            residual,
            (du_local, *prev_local, *parameters),
            grad_outputs=-lambda_,
            allow_unused=True,
            retain_graph=True,
        )
        # The gradient for `du` itself is not returned; the rest are the state
        # gradients that chain across increments, then the parameters.
        return (*(None,) * 11, *grads[1:])


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
            current iterate, Newton iteration index, and the previous
            state as one tuple.
        du: Initial guess for the unknown increment.
        B: Null-space rigid-body basis for AMG preconditioning.
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


def modal_eigsolve(
    K: Tensor, M: Tensor, n_modes: int, free_indices: Tensor
) -> tuple[Tensor, Tensor]:
    """Solve the generalized eigenvalue problem `K φ = ω² M φ`.

    ARPACK is the only shift-invert this depends on, so a CUDA model pays the
    transfer and solves on the CPU. Shift-invert at `sigma=0.0` targets the
    lowest modes.

    Args:
        K: Stiffness matrix.
        M: Mass matrix.
        n_modes: Number of eigenpairs to compute.
        free_indices: Free DOF indices. The eigenproblem is solved in their
            subspace to avoid spurious eigenvalues from the Dirichlet penalty
            (`K_ii = M_ii = 1  =>  ω² = 1`).

    Returns:
        Squared angular frequencies ascending, and the mode shapes, scattered
        back from the free-DOF subspace with constrained rows left at zero.
    """
    K_np, M_np = _to_scipy(K.cpu()), _to_scipy(M.cpu())
    fi = free_indices.cpu().numpy()
    vals, free = scipy_eigsh(
        K_np[fi, :][:, fi], k=n_modes, M=M_np[fi, :][:, fi], sigma=0.0
    )
    vecs = np.zeros((K.shape[0], n_modes))
    vecs[fi] = free
    order = np.argsort(vals)
    return (
        torch.tensor(vals[order], dtype=K.dtype, device=K.device),
        torch.tensor(vecs[:, order], dtype=K.dtype, device=K.device),
    )


class Eigensolve(Function):
    """Differentiable eigenvalue solver for K v = ω² M v.

    Gradients are computed via the Rayleigh-quotient sensitivity formula.
    Only eigenvalue gradients are supported; eigenvector gradients are not.
    """

    @staticmethod
    def forward(
        K: Tensor, M: Tensor, n_modes: int, free_indices: Tensor
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
        if grad_lambdas is None:
            return None, None, None, None

        # eigsh M-normalises the mode shapes; this holds whether or not it did
        phi_hat = phis / (phis * torch.sparse.mm(M, phis)).sum(0).abs().sqrt()

        def gradient(mat: Tensor, weights: Tensor) -> Tensor:
            """`phi_i phi_j` weighted per mode and summed, at `mat`'s pattern."""
            crow, col = mat.crow_indices(), mat.col_indices()
            values = (phi_hat[_rows(crow)] * phi_hat[col] * weights).sum(-1)
            with torch.sparse.check_sparse_tensor_invariants(False):
                return torch.sparse_csr_tensor(crow, col, values, size=mat.shape)

        return (
            gradient(K, grad_lambdas) if K.requires_grad else None,
            gradient(M, -lambdas * grad_lambdas) if M.requires_grad else None,
            None,
            None,
        )


def differentiable_modal_eigsolve(
    K: Tensor, M: Tensor, n_modes: int, free_indices: Tensor
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
