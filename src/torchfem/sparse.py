import numpy as np
import pyamg
import torch
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import cg as scipy_cg
from scipy.sparse.linalg import minres as scipy_minres
from scipy.sparse.linalg import spsolve as scipy_spsolve
from torch import Tensor, sparse_coo_tensor
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


class Solve(Function):
    """
    Inspired by
    - https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
    - https://github.com/pytorch/pytorch/issues/69538
    - https://github.com/cai4cai/torchsparsegradutils
    """

    @staticmethod
    def forward(
        A: sparse_coo_tensor,
        b: Tensor,
        B: Tensor = None,
        rtol: float = 1e-10,
        device: str = None,
        method: str = None,
        M: Tensor = None,
    ):
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
        Returns:
            Tensor: Solution vector x.
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
            x_xp = Solve._solve_gpu(A, b, B, method, rtol, M, shape)
        else:
            x_xp = Solve._solve_cpu(A, b, B, method, rtol, M, shape)

        # Convert back to torch
        x = torch.tensor(x_xp, requires_grad=True, dtype=b.dtype, device=out_device)

        return x

    @staticmethod
    def backward(ctx, grad):
        # Access the saved variables
        A, x = ctx.saved_tensors

        # Backprop rule: gradb = A^T @ grad
        gradb = Solve.apply(A.T, grad, ctx.B, ctx.rtol, ctx.device, ctx.method, ctx.M)

        # Backprop rule: gradA = -gradb @ x^T, sparse version
        row = A._indices()[0, :]
        col = A._indices()[1, :]
        val = -gradb[row] * x[col]
        gradA = torch.sparse_coo_tensor(torch.stack([row, col]), val, A.shape)

        return gradA, gradb, None, None, None, None, None

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, b, B, rtol, device, method, M = inputs
        x = output
        ctx.save_for_backward(A, x)

        # Save the parameters for backward pass (including the preconditioner)
        ctx.rtol = rtol
        ctx.device = device
        ctx.method = method
        ctx.B = B
        ctx.M = M

    @staticmethod
    def _solve_gpu(A, b, B, method, rtol, M, shape):
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
        if method == "pardiso":
            raise RuntimeError("Pardiso backend is not available on GPU.")
        elif method == "spsolve":
            x_xp = cupy_spsolve(A_cp, b_cp)
        elif method == "minres":
            # Jacobi preconditioner
            M = cupy_diags(1.0 / A_cp.diagonal())
            # Solve with minres
            x_xp, exit_code = cupy_minres(A_cp, b_cp, M=M, tol=rtol)
            if exit_code != 0:
                raise RuntimeError(f"minres failed with exit code {exit_code}")
        elif method == "cg":
            # Jacobi preconditioner
            M = cupy_diags(1.0 / A_cp.diagonal())
            # Solve with conjugated gradients
            x_xp, exit_code = cupy_cg(A_cp, b_cp, M=M, tol=rtol)
            if exit_code != 0:
                raise RuntimeError(f"CG failed with exit code {exit_code}")

        return x_xp

    @staticmethod
    def _solve_cpu(A, b, B, method, rtol, M, shape):
        A_np = scipy_coo_matrix(
            (A._values(), (A._indices()[0], A._indices()[1])), shape=shape
        ).tocsr()
        b_np = b.data.numpy()
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
            A_rcm = A_np[rcm_order, :][:, rcm_order]
            b_rcm = b_np[rcm_order]
            # Solve with pypardiso
            x_rcm = pypardiso.spsolve(A_rcm, b_rcm)
            # Restore the original order
            inv_rcm_order = np.argsort(rcm_order)
            x_xp = x_rcm[inv_rcm_order]
        elif method == "spsolve":
            x_xp = scipy_spsolve(A_np, b_np)
        elif method == "minres":
            # AMG preconditioner with Jacobi smoother
            if M is None:
                ml = pyamg.smoothed_aggregation_solver(A_np, B_np, smooth="jacobi")
                M = ml.aspreconditioner()

            # Solve with minres
            x_xp, exit_code = scipy_minres(A_np, b_np, M=M, rtol=rtol)
            if exit_code != 0:
                raise RuntimeError(f"minres failed with exit code {exit_code}")
        elif method == "cg":
            # AMG preconditioner with Jacobi smoother
            if M is None:
                ml = pyamg.smoothed_aggregation_solver(A_np, B_np, smooth="jacobi")
                M = ml.aspreconditioner()

            # Solve with minres
            x_xp, exit_code = scipy_cg(A_np, b_np, M=M, rtol=rtol)
            if exit_code != 0:
                raise RuntimeError(f"CG failed with exit code {exit_code}")

        return x_xp


sparse_solve = Solve.apply


def sparse_index_select(t: Tensor, slices: list[Tensor | None]) -> Tensor:
    coalesced = t.is_coalesced()
    indices = t.indices()
    values = t.values()
    in_shape = t.shape
    out_shape = []
    for dim, slice in enumerate(slices):
        if slice is None:
            out_shape.append(in_shape[dim])
        else:
            out_shape.append(len(slice))
            mask = torch.isin(indices[dim], slice)
            cumsum = torch.cumsum(torch.isin(torch.arange(0, in_shape[dim]), slice), 0)
            indices = indices[:, mask]
            values = values[mask]
            indices[dim] = cumsum[indices[dim]] - 1

    return torch.sparse_coo_tensor(indices, values, out_shape, is_coalesced=coalesced)
