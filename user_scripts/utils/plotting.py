import numpy as np
import matplotlib.pyplot as plt
import torch


# =============================================================================
def plot_displacement_field(
        coords_x, coords_y, u_x, u_y, node_coords, u_nodal, filename):
    """Plot displacement field within RVE at grid points.

    Args:
        coords_x (ndarray): X-coordinates grid of shape (res_x, res_y).
        coords_y (ndarray): Y-coordinates grid of shape (res_x, res_y).
        u_x (ndarray): X-displacement field of shape (res_x, res_y).
        u_y (ndarray): Y-displacement field of shape (res_x, res_y).
        node_coords (ndarray): Nodal coordinates of shape (n_nodes, 2).
        u_nodal (ndarray): Nodal displacements of shape (n_nodes, 2).
        filename (str): Output filename for the plot.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(12, 10))

    # Plot 1: Displacement magnitude contour
    u_magnitude = np.sqrt(u_x**2 + u_y**2)
    im1 = ax1.contourf(
        coords_x, coords_y, u_magnitude, levels=20, cmap='viridis')
    ax1.set_title('Displacement Magnitude')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)

    # Plot element boundary and nodes
    element_x = [
        node_coords[0, 0], node_coords[1, 0], node_coords[2, 0],
        node_coords[3, 0], node_coords[0, 0]]
    element_y = [
        node_coords[0, 1], node_coords[1, 1], node_coords[2, 1],
        node_coords[3, 1], node_coords[0, 1]]
    ax1.plot(element_x, element_y, 'k-', linewidth=2,
             label='Element boundary')
    ax1.scatter(
        node_coords[:, 0], node_coords[:, 1], c='red', s=50,
        zorder=5, label='Nodes')

    # Plot 2: X-displacement contour
    im2 = ax2.contourf(coords_x, coords_y, u_x, levels=20,
                       cmap='RdBu_r')
    ax2.set_title('X-Displacement')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    ax2.plot(element_x, element_y, 'k-', linewidth=2)
    ax2.scatter(
        node_coords[:, 0], node_coords[:, 1], c='red', s=50, zorder=5)

    # Plot 3: Y-displacement contour
    im3 = ax3.contourf(coords_x, coords_y, u_y, levels=20,
                       cmap='RdBu_r')
    ax3.set_title('Y-Displacement')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)
    ax3.plot(element_x, element_y, 'k-', linewidth=2)
    ax3.scatter(
        node_coords[:, 0], node_coords[:, 1], c='red', s=50, zorder=5)

    # Plot 4: Deformed shape with displacement vectors
    # Show every 5th point for clarity
    stride = 5
    coords_x_sub = coords_x[::stride, ::stride]
    coords_y_sub = coords_y[::stride, ::stride]
    u_x_sub = u_x[::stride, ::stride]
    u_y_sub = u_y[::stride, ::stride]

    # Scale factor for displacement visualization
    max_disp = np.max(np.sqrt(u_x**2 + u_y**2))
    scale_factor = 0.1 / max_disp if max_disp > 0 else 1

    ax4.quiver(
        coords_x_sub, coords_y_sub, u_x_sub, u_y_sub,
        scale_units='xy', scale=1/scale_factor, alpha=0.7, width=0.003)
    ax4.set_title('Displacement Vectors')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    ax4.plot(element_x, element_y, 'k-', linewidth=2)
    ax4.scatter(
        node_coords[:, 0], node_coords[:, 1], c='red', s=50, zorder=5)

    # Add deformed nodes
    deformed_coords = node_coords + u_nodal
    ax4.scatter(
        deformed_coords[:, 0], deformed_coords[:, 1], c='blue',
        s=50, zorder=5, label='Deformed nodes')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
# -----------------------------------------------------------------------------
def plot_domain_displacements(domain, u_disp, output_filename):
    """Plot displacement magnitude and components using domain.plot().

    Args:
        domain: FEM domain object with plot() method.
        u_disp (Tensor): Displacement field tensor of shape
            (num_increments, n_nodes, dim).
        output_filename (str): Base filename for output plots. Will be
            modified to create separate files for magnitude and component
            plots.
    """
    # Compute displacement magnitude at nodes
    u_magnitude = torch.sqrt(u_disp[-1, :, 0]**2 + u_disp[-1, :, 1]**2)

    # Plot displacement magnitude
    domain.plot(
        u=u_disp[-1],
        node_property=u_magnitude,
        title='Displacement Magnitude',
        colorbar=True,
        figsize=(6, 6),
        cmap='viridis'
    )
    plt.savefig(
        output_filename.replace('.pkl', '_torchfem_magnitude.png'),
        dpi=300, bbox_inches='tight')
    plt.close()

    # Plot X-displacement
    domain.plot(
        u=u_disp[-1],
        node_property=u_disp[-1, :, 0],
        title='X-Displacement',
        colorbar=True,
        figsize=(6, 6),
        cmap='RdBu_r'
    )
    plt.savefig(
        output_filename.replace('.pkl', '_torchfem_x_displacement.png'),
        dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Y-displacement
    domain.plot(
        u=u_disp[-1],
        node_property=u_disp[-1, :, 1],
        title='Y-Displacement',
        colorbar=True,
        figsize=(6, 6),
        cmap='RdBu_r'
    )
    plt.savefig(
        output_filename.replace('.pkl', '_torchfem_y_displacement.png'),
        dpi=300, bbox_inches='tight')
    plt.close()
# -----------------------------------------------------------------------------
def plot_shape_functions(domain, output_filename):
    """Plot all shape functions over the element.

    Args:
        domain: FEM domain object with nodes, elements, and element
            type attributes.
        output_filename (str): Base filename for output plot. Will be
            modified to append '_shape_functions.png'.
    """
    # Get node coordinates for the single element
    element_nodes = domain.elements[0]
    node_coords = domain.nodes[element_nodes]

    # Create a higher resolution grid
    xi_fine = torch.linspace(-1, 1, 100)
    eta_fine = torch.linspace(-1, 1, 100)
    xi_mesh_fine, eta_mesh_fine = torch.meshgrid(
        xi_fine, eta_fine, indexing='ij')

    # Flatten for shape function evaluation
    xi_flat_fine = xi_mesh_fine.flatten()
    eta_flat_fine = eta_mesh_fine.flatten()
    xi_points_fine = torch.stack([xi_flat_fine, eta_flat_fine], dim=-1)

    # Evaluate shape functions at grid points
    N_fine = domain.etype.N(xi_points_fine)

    # Interpolate physical coordinates
    coords_fine = torch.einsum('ij,jk->ik', N_fine, node_coords)
    coords_fine_x = coords_fine[:, 0].reshape(100, 100)
    coords_fine_y = coords_fine[:, 1].reshape(100, 100)

    # Plot all 4 shape functions
    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    shape_func_names = ['N1', 'N2', 'N3', 'N4']

    for i in range(4):
        # Reshape shape function values to grid
        N_i = N_fine[:, i].reshape(100, 100)

        im = axes[i].contourf(
            coords_fine_x.detach().cpu().numpy(),
            coords_fine_y.detach().cpu().numpy(),
            N_i.detach().cpu().numpy(),
            levels=20, cmap='viridis')

        axes[i].set_title(f'Shape Function {shape_func_names[i]}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_aspect('equal')

        plt.colorbar(im, ax=axes[i])

        # Plot element boundary
        element_x = [
            node_coords[0, 0], node_coords[1, 0],
            node_coords[2, 0], node_coords[3, 0],
            node_coords[0, 0]]
        element_y = [
            node_coords[0, 1], node_coords[1, 1],
            node_coords[2, 1], node_coords[3, 1],
            node_coords[0, 1]]
        axes[i].plot(element_x, element_y, 'k-', linewidth=2)

        # Highlight the node where shape function = 1
        axes[i].scatter(
            node_coords[i, 0], node_coords[i, 1],
            c='red', s=100, zorder=5,
            edgecolors='black', linewidth=2)

        # Add other nodes
        other_nodes = [j for j in range(4) if j != i]
        axes[i].scatter(
            node_coords[other_nodes, 0], node_coords[other_nodes, 1],
            c='white', s=50, zorder=5, edgecolors='black')

    plt.tight_layout()
    plt.savefig(
        output_filename.replace('.pkl', '_shape_functions.png'),
        dpi=300, bbox_inches='tight')
    plt.close()
# =============================================================================
