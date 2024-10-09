import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import minres, spsolve
from torch.autograd import Function


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

        A_np = coo_matrix(
            (A._values(), (A._indices()[0], A._indices()[1])), shape=shape
        ).tocsr()
        b_np = b.data.numpy()
        if shape[0] < 50000:
            # Solve small systems with direct solver
            x_np = spsolve(A_np, b_np)
        else:
            # Solve large systems with iterative solver
            x_np, exit_code = minres(A_np, b_np, rtol=1e-10)
            if exit_code != 0:
                raise RuntimeError(f"minres failed with exit code {exit_code}")

        # Convert back to torch
        x = torch.tensor(x_np, requires_grad=True, dtype=b.dtype, device=b.device)

        # Save the variables
        ctx.save_for_backward(A, b, x)

        return x

    @staticmethod
    def backward(ctx, grad):
        # Access the saved variables
        A, b, x = ctx.saved_tensors

        # Backprop rule: gradb = A^T @ grad
        gradb = Solve.apply(A.T, grad)

        # Backprop rule: gradA = -gradb @ x^T, sparse version
        row = A.indices()[0, :]
        col = A.indices()[1, :]
        val = -gradb[row] * x[col]
        gradA = torch.sparse_coo_tensor(torch.stack([row, col]), val, A.shape)

        return gradA, gradb


sparse_solve = Solve.apply


def sparse_index_select(t, slices):
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
