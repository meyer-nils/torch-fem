import pyamg
import torch
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse.linalg import minres as scipy_minres
from scipy.sparse.linalg import spsolve as scipy_spsolve
from torch import Tensor
from torch.autograd import Function
from .preconditioner import ssor_preconditioner

try:
    import cupy
    from cupyx.scipy.sparse import coo_matrix as cupy_coo_matrix
    from cupyx.scipy.sparse import diags as cupy_diags
    from cupyx.scipy.sparse.linalg import minres as cupy_minres
    from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve

    cupy_available = True
except ImportError:
    cupy_available = False


class Solve(Function):
    """
    Inspired by
    - https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
    - https://github.com/pytorch/pytorch/issues/69538
    - https://github.com/cai4cai/torchsparsegradutils
    """

    @staticmethod
    def forward(A, b, B=None, rtol=1e-10, device=None, direct=None, preconditioner=None):
        # Check the input shape
        if A.ndim != 2 or (A.shape[0] != A.shape[1]):
            raise ValueError("A should be a square 2D matrix.")
        shape = A.size()

        # Move to requested device, if available
        if device is not None:
            A = A.to(device)
            b = b.to(device)

        # Default to direct solver for small matrices
        if direct is not None:
            direct = shape[0] < 10000

        if A.device.type == "cuda" and cupy_available:
            A_cp = cupy_coo_matrix(
                (
                    cupy.asarray(A._values()),
                    (cupy.asarray(A._indices()[0]), cupy.asarray(A._indices()[1])),
                ),
                shape=shape,
            ).tocsr()
            b_cp = cupy.asarray(b.data)
            if direct:
                x_xp = cupy_spsolve(A_cp, b_cp)
            else:
                # Jacobi preconditioner
                if(preconditioner['name'] == 'jacobi'):
                    M = cupy_diags(1.0 / A_cp.diagonal())
                elif(preconditioner['name'] == 'ssor'):
                    M = ssor_preconditioner(A, omega=preconditioner['omega'], filter=preconditioner['filter'])

                # Solve with minres
                x_xp, exit_code = cupy_minres(A_cp, b_cp, M=M, tol=rtol)
                if exit_code != 0:
                    raise RuntimeError(f"minres failed with exit code {exit_code}")
        else:
            A_np = scipy_coo_matrix(
                (A._values(), (A._indices()[0], A._indices()[1])), shape=shape
            ).tocsr()
            b_np = b.data.numpy()
            if B is None:
                B_np = None
            else:
                B_np = B.data.numpy()
            if direct:
                x_xp = scipy_spsolve(A_np, b_np)
            else:
                # AMG preconditioner with Jacobi smoother
                if(preconditioner['name'] == 'ssor'):
                    raise NotImplementedError("SSOR Preconditioner is only implemented on GPU.")
                else: # Default on CPU is AMG preconditioner
                    ml = pyamg.smoothed_aggregation_solver(A_np, B_np, smooth="jacobi")
                    M = ml.aspreconditioner()

                # Solve with minres
                x_xp, exit_code = scipy_minres(A_np, b_np, M=M, rtol=rtol)
                if exit_code != 0:
                    raise RuntimeError(f"minres failed with exit code {exit_code}")

        # Convert back to torch
        x = torch.tensor(x_xp, requires_grad=True, dtype=b.dtype, device=b.device)

        return x

    @staticmethod
    def backward(ctx, grad):
        # Access the saved variables
        A, x = ctx.saved_tensors

        # Backprop rule: gradb = A^T @ grad
        gradb = Solve.apply(A.T, grad, ctx.B, ctx.rtol, ctx.device, ctx.direct, ctx.preconditioner)

        # Backprop rule: gradA = -gradb @ x^T, sparse version
        row = A._indices()[0, :]
        col = A._indices()[1, :]
        val = -gradb[row] * x[col]
        gradA = torch.sparse_coo_tensor(torch.stack([row, col]), val, A.shape)

        return gradA, gradb, None, None, None, None, None

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, b, B, rtol, device, direct, preconditioner = inputs
        x = output
        ctx.save_for_backward(A, x)

        # Save the parameters for backward pass (including the preconditioner)
        ctx.rtol = rtol
        ctx.device = device
        ctx.direct = direct
        ctx.B = B
        ctx.preconditioner = preconditioner


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
