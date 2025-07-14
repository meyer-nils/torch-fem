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

def cube_hexa_second_order(
    Nx: int, Ny: int, Nz: int, Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a second-order hexahedral mesh for a unit cube.
    
    Second-order hexahedral elements have 20 nodes:
    - 8 corner nodes (same as first-order)
    - 12 edge mid-point nodes
    
    Args:
        Nx, Ny, Nz: Number of nodes in each direction
        Lx, Ly, Lz: Dimensions of the cube
    
    Returns:
        nodes: (N, 3) tensor of node coordinates
        elements: (Ne, 20) tensor of element connectivity
    """
    # For second-order elements, we need twice as many intervals
    # but the same number of elements as first-order
    Nx_nodes = 2 * Nx - 1
    Ny_nodes = 2 * Ny - 1
    Nz_nodes = 2 * Nz - 1
    
    # Create nodes
    X = torch.linspace(0, Lx, Nx_nodes)
    Y = torch.linspace(0, Ly, Ny_nodes)
    Z = torch.linspace(0, Lz, Nz_nodes)
    x, y, z = torch.meshgrid(X, Y, Z, indexing="ij")
    nodes = torch.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    
    # Create elements
    indices = torch.arange(Nx_nodes * Ny_nodes * Nz_nodes).reshape((Nx_nodes, Ny_nodes, Nz_nodes))
    
    # Number of elements
    num_elements = (Nx - 1) * (Ny - 1) * (Nz - 1)
    elements = torch.zeros((num_elements, 20), dtype=torch.long)
    
    element_idx = 0
    for i in range(0, Nx_nodes - 1, 2):  # Step by 2 for second-order
        for j in range(0, Ny_nodes - 1, 2):
            for k in range(0, Nz_nodes - 1, 2):
                # Corner nodes (same ordering as first-order)
                n0 = indices[i, j, k]
                n1 = indices[i + 2, j, k]
                n2 = indices[i, j + 2, k]
                n3 = indices[i + 2, j + 2, k]
                n4 = indices[i, j, k + 2]
                n5 = indices[i + 2, j, k + 2]
                n6 = indices[i, j + 2, k + 2]
                n7 = indices[i + 2, j + 2, k + 2]
                
                # Edge mid-point nodes
                # Bottom face edges
                n8 = indices[i + 1, j, k]      # edge 0-1
                n9 = indices[i + 2, j + 1, k]  # edge 1-3
                n10 = indices[i + 1, j + 2, k] # edge 3-2
                n11 = indices[i, j + 1, k]     # edge 2-0
                
                # Top face edges
                n12 = indices[i + 1, j, k + 2]     # edge 4-5
                n13 = indices[i + 2, j + 1, k + 2] # edge 5-7
                n14 = indices[i + 1, j + 2, k + 2] # edge 7-6
                n15 = indices[i, j + 1, k + 2]     # edge 6-4
                
                # Vertical edges
                n16 = indices[i, j, k + 1]         # edge 0-4
                n17 = indices[i + 2, j, k + 1]     # edge 1-5
                n18 = indices[i + 2, j + 2, k + 1] # edge 3-7
                n19 = indices[i, j + 2, k + 1]     # edge 2-6
                
                elements[element_idx] = torch.tensor([
                    n0, n1, n3, n2, n4, n5, n7, n6,  # corner nodes
                    n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19  # edge nodes
                ])
                element_idx += 1
    
    return nodes, elements

def rect_quad(
    Nx: int, Ny: int, Lx: float = 1.0, Ly: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
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

def rect_quad_second_order(
    Nx: int, Ny: int, Lx: float = 1.0, Ly: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a second-order quadrilateral mesh for a unit rectangle.
    
    Second-order quadrilateral elements have 8 nodes:
    - 4 corner nodes (same as first-order)
    - 4 edge mid-point nodes
    
    Args:
        Nx, Ny: Number of nodes in each direction
        Lx, Ly: Dimensions of the rectangle
    
    Returns:
        nodes: (N, 2) tensor of node coordinates
        elements: (Ne, 8) tensor of element connectivity
    """
    # For second-order elements, we need twice as many intervals
    # but the same number of elements as first-order
    Nx_nodes = 2 * Nx - 1
    Ny_nodes = 2 * Ny - 1
    
    # Create nodes
    X = torch.linspace(0, Lx, Nx_nodes)
    Y = torch.linspace(0, Ly, Ny_nodes)
    x, y = torch.meshgrid(X, Y, indexing="ij")
    nodes = torch.vstack([x.ravel(), y.ravel()]).T
    
    # Create elements
    indices = torch.arange(Nx_nodes * Ny_nodes).reshape((Nx_nodes, Ny_nodes))
    
    # Number of elements
    num_elements = (Nx - 1) * (Ny - 1)
    elements = torch.zeros((num_elements, 8), dtype=torch.long)
    
    element_idx = 0
    for i in range(0, Nx_nodes - 1, 2):  # Step by 2 for second-order
        for j in range(0, Ny_nodes - 1, 2):
            # Corner nodes (same ordering as first-order)
            n0 = indices[i, j]
            n1 = indices[i + 2, j]
            n2 = indices[i, j + 2]
            n3 = indices[i + 2, j + 2]
            
            # Edge mid-point nodes
            n4 = indices[i + 1, j]      # edge 0-1
            n5 = indices[i + 2, j + 1]  # edge 1-3
            n6 = indices[i + 1, j + 2]  # edge 3-2
            n7 = indices[i, j + 1]      # edge 2-0
            
            elements[element_idx] = torch.tensor([n0, n1, n3, n2, n4, n5, n6, n7])
            element_idx += 1
    
    return nodes, elements
