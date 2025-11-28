from typing import Literal

import torch


def cube_hexa(
    Nx: int, Ny: int, Nz: int, Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a structured hexahedral mesh on a cube.

    Args:
        Nx: Number of grid points along the x-axis.
        Ny: Number of grid points along the y-axis.
        Nz: Number of grid points along the z-axis.
        Lx: Total length of the cube along the x-axis (default 1.0).
        Ly: Total length of the cube along the y-axis (default 1.0).
        Lz: Total length of the cube along the z-axis (default 1.0).

    Returns:
        A tuple of (nodes, cells):
        - nodes: Node coordinates as a tensor of shape (num_nodes, 3).
        - cells: Hexahedral element connectivity as a tensor of shape (num_hexes, 8).
    """
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


def cube_tetra(
    Nx: int, Ny: int, Nz: int, Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a structured tetrahedral mesh on a cube via splitting hexahedra into five
    tetrahedra each. The splitting alternates between two different patterns in a
    checkerboard fashion to improve mesh quality.

    Args:
        Nx: Number of grid points along the x-axis.
        Ny: Number of grid points along the y-axis.
        Nz: Number of grid points along the z-axis.
        Lx: Total length of the cube along the x-axis (default 1.0).
        Ly: Total length of the cube along the y-axis (default 1.0).
        Lz: Total length of the cube along the z-axis (default 1.0).

    Returns:
        A tuple of (nodes, cells):
        - nodes: Node coordinates as a tensor of shape (num_nodes, 3).
        - cells: Tetrahedral element connectivity as a tensor of shape (num_tetras, 4).
    """

    # Compute base hexas
    nodes, elements = cube_hexa(Nx, Ny, Nz, Lx, Ly, Lz)

    # Create a checkerboard pattern to alternate the two 5-tet splits
    I, J, K = torch.meshgrid(
        torch.arange(Nx - 1),
        torch.arange(Ny - 1),
        torch.arange(Nz - 1),
        indexing="ij",
    )
    idx = ((I + J + K) % 2 == 0).ravel()
    c_even = elements[idx]
    c_odd = elements[~idx]

    # Even splits
    template_even = torch.tensor(
        [[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [1, 4, 5, 6], [3, 4, 6, 7]]
    )
    tets_even = c_even[:, template_even].reshape(-1, 4)

    # Odd splits
    template_odd = torch.tensor(
        [[4, 5, 0, 7], [5, 6, 2, 7], [5, 7, 2, 0], [5, 0, 2, 1], [7, 0, 3, 2]]
    )
    tets_odd = c_odd[:, template_odd].reshape(-1, 4)

    # Combine
    elements = torch.cat([tets_even, tets_odd], dim=0)

    return nodes, elements


def rect_quad(
    Nx: int, Ny: int, Lx: float = 1.0, Ly: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a structured quadrilateral mesh on a rectangle.

    Args:
        Nx: Number of grid points along the x-axis.
        Ny: Number of grid points along the y-axis.
        Lx: Total length of the rectangle along the x-axis (default 1.0).
        Ly: Total length of the rectangle along the y-axis (default 1.0).

    Returns:
        A tuple of (nodes, cells):
        - nodes: Node coordinates as a tensor of shape (num_nodes, 2).
        - cells: Quadrilateral element connectivity as a tensor of shape (num_quads, 4).
    """

    # Create nodes
    X = torch.linspace(0, Lx, Nx)
    Y = torch.linspace(0, Ly, Ny)
    x, y = torch.meshgrid(X, Y, indexing="ij")
    nodes = torch.vstack([x.ravel(), y.ravel()]).T

    # Create elements
    indices = torch.arange(Nx * Ny).reshape((Nx, Ny))
    n0 = indices[:-1, :-1].ravel()
    n1 = indices[1:, :-1].ravel()
    n2 = indices[:-1, 1:].ravel()
    n3 = indices[1:, 1:].ravel()
    elements = torch.vstack([n0, n1, n3, n2]).T

    return nodes, elements


def rect_tri(
    Nx: int,
    Ny: int,
    Lx: float = 1.0,
    Ly: float = 1.0,
    variant: Literal["up", "down", "zigzag", "center"] = "zigzag",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a structured triangular mesh on a rectangle.

    Args:
        Nx: Number of grid points along the x-axis.
        Ny: Number of grid points along the y-axis.
        Lx: Total length of the rectangle along the x-axis (default 1.0).
        Ly: Total length of the rectangle along the y-axis (default 1.0).
        variant: Triangulation. Options are 'up', 'down', 'zigzag', and 'center'.

    Returns:
        A tuple of (nodes, cells):
        - nodes: Node coordinates as a tensor of shape (num_nodes, 2).
        - cells: Triangle element connectivity as a tensor of shape (num_triangles, 3).
    """

    # Start with quads
    nodes, quads = rect_quad(Nx, Ny, Lx, Ly)

    # Extract quad corners:
    n0, n1, n3, n2 = quads.unbind(dim=1)

    if variant == "up":
        # Up split
        T1 = torch.stack([n0, n1, n3], dim=1)
        T2 = torch.stack([n0, n3, n2], dim=1)
        return nodes, torch.vstack([T1, T2])

    elif variant == "down":
        # Down split
        T1 = torch.stack([n1, n3, n2], dim=1)
        T2 = torch.stack([n1, n2, n0], dim=1)
        return nodes, torch.vstack([T1, T2])

    elif variant == "zigzag":
        # Determine the type of split for each quad based on its position
        i = torch.div(n0, Ny, rounding_mode="floor")
        j = n0 % Ny
        is_type_A = ((i + j) % 2 == 0).unsqueeze(1)

        # Up split
        T1_A = torch.stack([n0, n1, n3], dim=1)
        T2_A = torch.stack([n0, n3, n2], dim=1)

        # Down split
        T1_B = torch.stack([n1, n3, n2], dim=1)
        T2_B = torch.stack([n1, n2, n0], dim=1)

        # Select T1 and T2 based on the mask
        T1 = torch.where(is_type_A, T1_A, T1_B)
        T2 = torch.where(is_type_A, T2_A, T2_B)
        return nodes, torch.vstack([T1, T2])

    elif variant == "center":
        ecenters = nodes[quads].mean(dim=1)
        nodes = torch.vstack([nodes, ecenters])

        # Get the indices of the new center nodes
        n_c = torch.arange(nodes.size(0) - ecenters.size(0), nodes.size(0))

        # Create the four triangles for each quad
        T1 = torch.stack([n0, n1, n_c], dim=1)
        T2 = torch.stack([n1, n3, n_c], dim=1)
        T3 = torch.stack([n3, n2, n_c], dim=1)
        T4 = torch.stack([n2, n0, n_c], dim=1)

        # Stack all four sets of triangles
        cells = torch.vstack([T1, T2, T3, T4]).long()
        return nodes, cells
    else:
        raise ValueError(f"Unknown variant: {variant}")
