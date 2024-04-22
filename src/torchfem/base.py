import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from torch.autograd import Function


class Solve(Function):
    @staticmethod
    def forward(ctx, A, b):
        if A.ndim != 2 or (A.shape[0] != A.shape[1]):
            raise ValueError("A should be a square 2D matrix.")
        shape = A.size()
        A_np = coo_matrix(
            (A._values(), (A._indices()[0], A._indices()[1])), shape=shape
        ).tocsr()
        b_np = b.data.numpy()
        x_np = spsolve(A_np, b_np)
        x = torch.tensor(x_np, requires_grad=True)
        ctx.save_for_backward(A, b, x)
        return x

    @staticmethod
    def backward(ctx, grad):
        A, b, x = ctx.saved_tensors
        gradb = Solve.apply(A.T, grad)
        gradA = -gradb[:, None] * x[None, :]
        return gradA, gradb


sparse_solve = Solve.apply


def sparse_index_select(t, slices):
    indices = t.indices()
    values = t.values()
    shape = t.shape
    for dim, slice in enumerate(slices):
        mask = torch.isin(indices[dim], slice)
        cumsum = torch.cumsum(torch.isin(torch.arange(0, shape[dim]), slice), 0)
        indices = indices[:, mask]
        values = values[mask]
        indices[dim] = cumsum[indices[dim]] - 1

    return torch.sparse_coo_tensor(indices, values, [len(s) for s in slices])
