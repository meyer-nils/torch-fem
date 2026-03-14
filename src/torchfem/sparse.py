from typing import Callable, Tuple

import numpy as np
import pyamg
import torch
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg as scipy_cg
from scipy.sparse.linalg import minres as scipy_minres
from scipy.sparse.linalg import spsolve as scipy_spsolve
from torch import Tensor
from torch.autograd import Function

available_backends = ["scipy"]

try:
    import cupy
    from cupyx.scipy.sparse import coo_matrix as cupy_coo_matrix
    from cupyx.scipy.sparse import diags as cupy_diags
    from cupyx.scipy.sparse.linalg import cg as cupy_cg
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


class NewtonRaphsonAdjoint(Function):

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
        verbose: bool,
        method: str | None = None,
        device: str | None = None,
        *parameters: Tensor,
    ) -> Tensor:
        M = None

        # Newton-Raphson iterations
        for i in range(max_iter):

            # Evaluate residual, stiffness matrix, and internal forces
            residual, K = eval_residual(du, i)

            # Compute residual norm
            res_norm = torch.linalg.norm(residual)

            # Save initial residual
            if i == 0:
                res_norm0 = res_norm

            # Print iteration information
            if verbose:
                print(f"Solver iteration {i + 1} | Residual: {res_norm:.5e}")

            # Check convergence
            if res_norm < rtol * res_norm0 or res_norm < atol:
                break

            if torch.isnan(res_norm) or torch.isinf(res_norm):
                raise Exception("Newton-Raphson iteration did not converge")

            # Solve for displacement increment
            du_i, M = sparse_solve(K, residual, B, stol, device, method, M)
            du = du - du_i

        # Final convergence check
        if res_norm > rtol * res_norm0 and res_norm > atol:
            raise Exception("Newton-Raphson iteration did not converge.")

        ctx.save_for_backward(K, du, *parameters)
        ctx.B = B
        ctx.M = M
        ctx.stol = stol
        ctx.device = device
        ctx.method = method
        ctx.eval_residual = eval_residual
        ctx.n_parameters = len(parameters)

        return du

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_du = grad_outputs[0]

        K, du, *parameters = ctx.saved_tensors

        B = ctx.B
        M = ctx.M
        stol = ctx.stol
        device = ctx.device
        method = ctx.method
        eval_residual = ctx.eval_residual

        # Solve adjoint system.
        lambda_, _ = sparse_solve(
            K.T,
            grad_du,
            B,
            stol,
            device,
            method,
            M,
        )

        # Recompute the residual with a differentiable local state.
        du_local = du.detach().requires_grad_(True)
        with torch.enable_grad():
            residual, _ = eval_residual(du_local, 0)

        grad_inputs = (du_local, *parameters)
        grads = torch.autograd.grad(
            residual,
            grad_inputs,
            grad_outputs=-lambda_,
            allow_unused=True,
            retain_graph=True,
        )
        grad_du_input = grads[0]
        grad_parameters = grads[1:]

        return (
            None,
            grad_du_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
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
    verbose: bool,
    method: str | None = None,
    device: str | None = None,
    *parameters: Tensor,
) -> Tensor:
    du = NewtonRaphsonAdjoint.apply(
        eval_residual,
        du,
        B,
        max_iter,
        rtol,
        atol,
        stol,
        verbose,
        method,
        device,
        *parameters,
    )  # type: ignore
    if du is None:
        raise RuntimeError("Solve.apply returned None, expected a Tensor.")
    return du


def sparse_solve(
    A: Tensor,
    b: Tensor,
    B: Tensor | None = None,
    rtol: float = 1e-10,
    device: str | None = None,
    method: str | None = None,
    M: LinearOperator | None = None,
    x0: Tensor | None = None,
) -> Tuple[Tensor, LinearOperator | None]:
    """
    Solve the linear system Ax = b.

    Args:
        A (sparse_coo_tensor): Sparse matrix A.
        b (Tensor): Right-hand side vector b.
        B (Tensor, optional): Null space rigid body modes for AMG preconditioner.
        rtol (float, optional): Relative tolerance for the iterative solver.
            Defaults to 1e-10.
        device (str, optional): Device to run the computation on ('cpu' or 'cuda').
            Defaults to None, which uses the current device.
        method (str, optional): Method to use for solving ('spsolve', 'minres',
            'cg', 'pardiso'). Defaults to None for automatic selection based on the
            input size and available backends.
        M (Tensor, optional): Preconditioner matrix for iterative methods.
            Defaults to None.
        x0 (Tensor, optional): Initial guess for iterative solvers. Defaults to None.
    Returns:
        Tuple[Tensor, LinearOperator | None]: Solution vector x and preconditioner  M.
    """
    # Check the input shape
    if A.ndim != 2 or (A.shape[0] != A.shape[1]):
        raise ValueError("A should be a square 2D matrix.")
    shape = A.size()
    out_device = b.device

    # Check the input method
    if method is not None and method not in ["spsolve", "minres", "cg", "pardiso"]:
        raise ValueError(
            f"Method {method} is not supported. "
            "Choose from 'spsolve', 'minres', 'cg', or 'pardiso'."
        )

    # Move to requested device, if available
    if device is not None:
        A = A.to(device)
        b = b.to(device)

    # Make default solver choice based on shape and available backends
    if method is None:
        if shape[0] < 10000:
            if A.device.type == "cpu" and "pypardiso" in available_backends:
                method = "pardiso"
            else:
                method = "spsolve"
        else:
            method = "minres"

    # Solve either on CPU or GPU
    if A.device.type == "cuda":
        x_xp, M_xp = _solve_gpu(A, b, B, method, rtol, M, shape, x0)
    else:
        x_xp, M_xp = _solve_cpu(A, b, B, method, rtol, M, shape, x0)

    # Convert back to torch
    x = torch.tensor(x_xp, dtype=b.dtype, device=out_device)

    return x, M_xp


def _solve_gpu(A, b, B, method, stol, M, shape, x0):
    if "cupy" not in available_backends:
        raise RuntimeError(
            "CuPy is not available.\n\n"
            "Please install CuPy to use GPU acceleration:\n"
            "> pip install cupy-cuda11x # v11.2 - 11.8\n"
            "> pip install cupy-cuda12x # v12.x"
        )
    A_cp = cupy_coo_matrix(
        (
            cupy.asarray(A._values()),
            (cupy.asarray(A._indices()[0]), cupy.asarray(A._indices()[1])),
        ),
        shape=shape,
    ).tocsr()
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
        # Jacobi preconditioner
        M = cupy_diags(1.0 / A_cp.diagonal())
        # Solve with minres
        x_xp, exit_code = cupy_minres(A_cp, b_cp, M=M, tol=stol, x0=x0_cp)
        if exit_code != 0:
            raise RuntimeError(f"minres failed with exit code {exit_code}")
    elif method == "cg":
        # Jacobi preconditioner
        M = cupy_diags(1.0 / A_cp.diagonal())
        # Solve with conjugated gradients
        x_xp, exit_code = cupy_cg(A_cp, b_cp, M=M, tol=stol, x0=x0_cp)
        if exit_code != 0:
            raise RuntimeError(f"CG failed with exit code {exit_code}")

    return x_xp, M


def _solve_cpu(A, b, B, method, stol, M, shape, x0):
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
            raise RuntimeError(
                "PyPardiso is not available.\n\n"
                "Please install Pypardiso seperately:\n"
                "> pip install pypardiso"
            )
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
        x_xp, exit_code = scipy_minres(A_np, b_np, M=M, rtol=stol, x0=x0_np)
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
