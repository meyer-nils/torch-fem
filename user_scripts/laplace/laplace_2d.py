"""
2D Laplace equation example using torch-fem.

Solves: ∇²φ = 0 in Ω = [0,1] x [0,1]
Boundary conditions:
  - φ = 0 on left edge (x=0)
  - φ = 1 on right edge (x=1)
  - ∂φ/∂n = 0 on top/bottom (natural BC, automatically satisfied)

Analytical solution: φ(x,y) = x
"""

import sys
import os
# Add local src to path to use modified code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchfem.planar import Planar
from torchfem.operators import DiffusionOperator


def create_mesh(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0):
    """
    Create structured rectangular mesh.

    Args:
        nx: Number of elements in x direction
        ny: Number of elements in y direction
        lx: Domain length in x
        ly: Domain length in y

    Returns:
        (nodes, elements): Node coordinates and element connectivity
    """
    # Generate node grid
    x = torch.linspace(0, lx, nx + 1)
    y = torch.linspace(0, ly, ny + 1)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    nodes = torch.stack([xx.ravel(), yy.ravel()], dim=1)

    # Generate element connectivity (quad4)
    elements = []
    for i in range(nx):
        for j in range(ny):
            # Node numbering: column-major
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + (j + 1)
            n3 = i * (ny + 1) + (j + 1)
            elements.append([n0, n1, n2, n3])

    elements = torch.tensor(elements, dtype=torch.long)
    return nodes, elements


def identify_boundary_nodes(nodes, tol=1e-6):
    """
    Identify nodes on domain boundaries.

    Returns:
        dict with keys 'left', 'right', 'top', 'bottom'
    """
    x = nodes[:, 0]
    y = nodes[:, 1]

    boundaries = {
        'left': torch.where(torch.abs(x) < tol)[0],
        'right': torch.where(torch.abs(x - x.max()) < tol)[0],
        'bottom': torch.where(torch.abs(y) < tol)[0],
        'top': torch.where(torch.abs(y - y.max()) < tol)[0],
    }
    return boundaries


def analytical_solution(x, y):
    """
    Manufactured solution: φ(x,y) = sin(π*x) * sin(π*y)

    Satisfies: -∇²φ = 2π² sin(πx)sin(πy) = f
    Natural BCs: φ = 0 on all boundaries of [0,1]×[0,1]
    """
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_term(x, y):
    """Source term f = 2π² sin(πx)sin(πy) for -∇²φ = f"""
    return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def analytical_gradient(x, y):
    """
    Analytical gradient ∇φ = [∂φ/∂x, ∂φ/∂y].

    For φ = sin(πx)sin(πy):
    ∂φ/∂x = π cos(πx)sin(πy)
    ∂φ/∂y = π sin(πx)cos(πy)
    """
    dphidx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    dphidy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return dphidx, dphidy


def compute_h1_error(fem, u, analytical_sol, analytical_grad):
    """
    Compute H1 seminorm error: ||∇(φ_h - φ)||_L2.

    H1 seminorm: |u|_H1 = sqrt(∫_Ω |∇u|² dΩ)

    Args:
        fem: FEM object with mesh and elements
        u: Numerical solution (n_nodes, 1)
        analytical_sol: Function (x, y) -> φ
        analytical_grad: Function (x, y) -> (∂φ/∂x, ∂φ/∂y)

    Returns:
        h1_seminorm_error: H1 seminorm of the error
    """
    from torchfem.elements import Quad1

    etype = Quad1()
    n_elem = fem.n_elem

    # Extract coordinates
    x_coords = fem.nodes[:, 0].numpy()
    y_coords = fem.nodes[:, 1].numpy()

    # Numerical solution at nodes
    phi_h = u[:, 0]

    # Compute gradient error at quadrature points
    error_squared = 0.0

    for elem_idx in range(n_elem):
        # Get element nodes
        elem_nodes = fem.elements[elem_idx]
        x_elem = fem.nodes[elem_nodes, 0]
        y_elem = fem.nodes[elem_nodes, 1]
        phi_elem = phi_h[elem_nodes]

        # Integrate over element
        for w, xi in zip(etype.iweights(), etype.ipoints()):
            # Shape functions and derivatives
            N = etype.N(xi)
            dN_dxi = etype.B(xi)

            # Jacobian: J_ij = dX_i/dxi_j = sum_k (dN_k/dxi_j * X_ki)
            # dN_dxi has shape (n_dim, n_nodes) = (2, 4)
            # X has shape (n_nodes, n_dim) = (4, 2)
            X = torch.stack([x_elem, y_elem], dim=1)
            J = torch.einsum('ik,kj->ij', dN_dxi, X)
            detJ = torch.det(J)
            J_inv = torch.inverse(J)

            # Shape function derivatives in physical space
            # dN/dx = J^(-1) @ dN/dxi
            dN_dx = torch.einsum('ij,jk->ik', J_inv, dN_dxi)

            # Numerical gradient at quadrature point
            grad_phi_h = torch.einsum('ik,k->i', dN_dx, phi_elem)

            # Physical coordinates at quadrature point
            x_qp = torch.dot(N, x_elem).item()
            y_qp = torch.dot(N, y_elem).item()

            # Analytical gradient at quadrature point
            dphidx_exact, dphidy_exact = analytical_grad(x_qp, y_qp)
            grad_phi_exact = np.array([dphidx_exact, dphidy_exact])

            # Gradient error
            grad_error = grad_phi_h.numpy() - grad_phi_exact

            # Add to integral
            error_squared += w * detJ.item() * np.dot(grad_error,
                                                      grad_error)

    h1_seminorm = np.sqrt(error_squared)
    return h1_seminorm


def main():
    print("="*60)
    print("2D Poisson Equation with torch-fem")
    print("Manufactured: φ = sin(πx)sin(πy), -∇²φ = 2π²sin(πx)sin(πy)")
    print("="*60)

    # Mesh parameters
    nx, ny = 10, 10
    print(f"\nCreating {nx}x{ny} mesh...")
    nodes, elements = create_mesh(nx, ny)
    print(f"  Nodes: {nodes.shape[0]}")
    print(f"  Elements: {elements.shape[0]}")

    # Create diffusion operator (k=1)
    conductivity = 1.0
    operator = DiffusionOperator(conductivity, n_dim=2)
    print(f"\nDiffusion operator created (k={conductivity})")

    # Initialize FEM problem
    fem = Planar(nodes, elements, operator)
    print(f"FEM problem initialized (Planar with Quad elements)")

    # Apply Dirichlet BCs: φ=0 on all boundaries
    boundaries = identify_boundary_nodes(nodes)
    all_boundary = np.concatenate([
        boundaries['left'], boundaries['right'],
        boundaries['top'], boundaries['bottom']
    ])
    all_boundary = np.unique(all_boundary)

    print(f"\nApplying boundary conditions:")
    print(f"  All boundaries: φ = 0 ({len(all_boundary)} nodes)")

    # Set constraints (scalar field: 1 DOF per node)
    constraints = torch.zeros((nodes.shape[0], 1), dtype=torch.bool)
    constraints[all_boundary, 0] = True
    fem.constraints = constraints

    # Set prescribed values (all boundaries = 0)
    displacements = torch.zeros((nodes.shape[0], 1))
    fem.displacements = displacements

    # Set source term f = 2π² sin(πx)sin(πy)
    x_coords = nodes[:, 0].numpy()
    y_coords = nodes[:, 1].numpy()
    f_source = source_term(x_coords, y_coords)

    # Integrate source term over domain: F = ∫ N^T f dΩ
    # Use nodal quadrature approximation: each node owns area h²
    h = 1.0 / nx
    nodal_area = h * h

    forces = torch.tensor(
        f_source * nodal_area, dtype=torch.float32
    ).reshape(-1, 1)
    fem.forces = forces
    print(f"  Source: f = 2π²sin(πx)sin(πy)")

    # Solve using direct forward solver for linear problem
    print(f"\nSolving Poisson equation...")
    u, f_out = fem.solve_forward(
        verbose=True,
        method='spsolve'
    )

    # Extract solution
    phi = u[:, 0].numpy()

    # Analytical solution
    phi_analytical = analytical_solution(x_coords, y_coords)

    # Error analysis
    error = np.abs(phi - phi_analytical)
    max_error = error.max()
    mean_error = error.mean()
    l2_error = np.sqrt(np.mean(error**2))
    h1_error = compute_h1_error(
        fem, u, analytical_solution, analytical_gradient
    )

    print(f"\nError analysis:")
    print(f"  Max error: {max_error:.6e}")
    print(f"  Mean error: {mean_error:.6e}")
    print(f"  L2 error: {l2_error:.6e}")
    print(f"  H1 seminorm error: {h1_error:.6e}")

    # Visualization
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Numerical solution
    sc1 = axes[0].tricontourf(x_coords, y_coords, phi, levels=20,
                               cmap='viridis')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Numerical Solution φ(x,y)')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0])

    # Analytical solution
    sc2 = axes[1].tricontourf(x_coords, y_coords, phi_analytical,
                               levels=20, cmap='viridis')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Analytical: φ = sin(πx)sin(πy)')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1])

    # Error
    sc3 = axes[2].tricontourf(x_coords, y_coords, error, levels=20,
                               cmap='hot')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title('Absolute Error')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('poisson_2d_solution.png', dpi=150)
    print(f"Plot saved to: poisson_2d_solution.png")

    # Convergence study
    print(f"\n" + "="*60)
    print("Convergence Study")
    print("="*60)
    mesh_sizes = [5, 10, 20, 40, 80, 160]
    errors_l2 = []
    errors_h1 = []
    h_values = []

    for n in mesh_sizes:
        nodes_i, elements_i = create_mesh(n, n)
        op_i = DiffusionOperator(1.0, n_dim=2)
        fem_i = Planar(nodes_i, elements_i, op_i)

        # Apply BCs: φ=0 on all boundaries
        boundaries_i = identify_boundary_nodes(nodes_i)
        all_boundary_i = np.concatenate([
            boundaries_i['left'], boundaries_i['right'],
            boundaries_i['top'], boundaries_i['bottom']
        ])
        all_boundary_i = np.unique(all_boundary_i)

        constraints_i = torch.zeros((nodes_i.shape[0], 1), dtype=torch.bool)
        constraints_i[all_boundary_i, 0] = True
        fem_i.constraints = constraints_i

        displacements_i = torch.zeros((nodes_i.shape[0], 1))
        fem_i.displacements = displacements_i

        # Set source term with nodal quadrature
        x_i = nodes_i[:, 0].numpy()
        y_i = nodes_i[:, 1].numpy()
        f_i = source_term(x_i, y_i)
        h_i = 1.0 / n
        nodal_area_i = h_i * h_i
        forces_i = torch.tensor(
            f_i * nodal_area_i, dtype=torch.float32
        ).reshape(-1, 1)
        fem_i.forces = forces_i

        u_i, _ = fem_i.solve_forward(
            verbose=False,
            method='spsolve'
        )

        phi_i = u_i[:, 0].numpy()
        phi_exact_i = analytical_solution(x_i, y_i)
        error_l2 = np.sqrt(np.mean((phi_i - phi_exact_i)**2))
        error_h1 = compute_h1_error(
            fem_i, u_i, analytical_solution, analytical_gradient
        )
        h = 1.0 / n

        errors_l2.append(error_l2)
        errors_h1.append(error_h1)
        h_values.append(h)
        print(f"  n={n:2d}, h={h:.4f}, L2={error_l2:.6e}, "
              f"H1={error_h1:.6e}")

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # L2 error
    axes[0].loglog(h_values, errors_l2, 'o-', label='L2 error',
                   linewidth=2, markersize=8)
    axes[0].loglog(h_values, np.array(h_values)**2, '--',
                   label='O(h²)', linewidth=1.5)
    axes[0].set_xlabel('Mesh size h', fontsize=12)
    axes[0].set_ylabel('L2 Error', fontsize=12)
    axes[0].set_title('L2 Norm Convergence', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, which='both', alpha=0.3)

    # H1 error
    axes[1].loglog(h_values, errors_h1, 's-', label='H1 seminorm',
                   color='C1', linewidth=2, markersize=8)
    axes[1].loglog(h_values, np.array(h_values), '--',
                   label='O(h)', linewidth=1.5, color='C2')
    axes[1].set_xlabel('Mesh size h', fontsize=12)
    axes[1].set_ylabel('H1 Seminorm Error', fontsize=12)
    axes[1].set_title('H1 Seminorm Convergence', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('laplace_2d_convergence.png', dpi=150)
    print(f"\nConvergence plot saved to: laplace_2d_convergence.png")

    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
