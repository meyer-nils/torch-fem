import torch
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse.linalg import minres as scipy_minres
from scipy.sparse.linalg import spsolve as scipy_spsolve
from torch import Tensor
from torch.autograd import Function

available_backends = ["scipy"]

try:
    import cupy
    from cupyx.scipy.sparse import coo_matrix as cupy_coo_matrix
    from cupyx.scipy.sparse.linalg import minres as cupy_minres
    from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve

    available_backends.append("cupy")
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
    def forward(ctx, A, b):
        # Check the input and make sure it is coalesced
        if A.ndim != 2 or (A.shape[0] != A.shape[1]):
            raise ValueError("A should be a square 2D matrix.")
        shape = A.size()

        if A.device.type == "cuda" and "cupy" in available_backends:
            A_cp = cupy_coo_matrix(
                (
                    cupy.asarray(A._values()),
                    (cupy.asarray(A._indices()[0]), cupy.asarray(A._indices()[1])),
                ),
                shape=shape,
            ).tocsr()
            b_cp = cupy.asarray(b.data)
            if shape[0] < 10000:
                x_xp = cupy_spsolve(A_cp, b_cp)
            else:
                x_xp, exit_code = cupy_minres(A_cp, b_cp, tol=1e-10)
                if exit_code != 0:
                    raise RuntimeError(f"minres failed with exit code {exit_code}")
        else:
            A_np = scipy_coo_matrix(
                (A._values(), (A._indices()[0], A._indices()[1])), shape=shape
            ).tocsr()
            b_np = b.data.numpy()
            if shape[0] < 10000:
                x_xp = scipy_spsolve(A_np, b_np)
            else:
                x_xp, exit_code = scipy_minres(A_np, b_np, rtol=1e-10)
                if exit_code != 0:
                    raise RuntimeError(f"minres failed with exit code {exit_code}")

        # Convert back to torch
        x = torch.tensor(x_xp, requires_grad=True, dtype=b.dtype, device=b.device)

        # Save the variables
        ctx.save_for_backward(A, x)

        return x

    @staticmethod
    def backward(ctx, grad):
        # Access the saved variables
        A, x = ctx.saved_tensors

        # Backprop rule: gradb = A^T @ grad
        gradb = Solve.apply(A.T, grad)

        # Backprop rule: gradA = -gradb @ x^T, sparse version
        row = A.indices()[0, :]
        col = A.indices()[1, :]
        val = -gradb[row] * x[col]
        gradA = torch.sparse_coo_tensor(torch.stack([row, col]), val, A.shape)

        return gradA, gradb


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
