import torch


def cube_hexa(
    Nx: int, Ny: int, Nz: int, Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    # Create nodes
    X = torch.linspace(0, Lx, Nx)
    Y = torch.linspace(0, Ly, Ny)
    Z = torch.linspace(0, Lz, Nz)
    x, y, z = torch.meshgrid(X, Y, Z, indexing="ij")
    nodes = torch.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Create elements
    indices = torch.arange(Nx * Ny * Nz).reshape((Nx, Ny, Nz))
    n0 = indices[:-1, :-1, :-1].ravel()
    n1 = indices[1:, :-1, :-1].ravel()
    n2 = indices[:-1, 1:, :-1].ravel()
    n3 = indices[1:, 1:, :-1].ravel()
    n4 = indices[:-1, :-1, 1:].ravel()
    n5 = indices[1:, :-1, 1:].ravel()
    n6 = indices[:-1, 1:, 1:].ravel()
    n7 = indices[1:, 1:, 1:].ravel()
    elements = torch.vstack([n0, n1, n3, n2, n4, n5, n7, n6]).T

    return nodes, elements
