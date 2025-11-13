import numpy as np
import matplotlib.pyplot as plt
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch

# =============================================================================
def interpolate_displacement_field(
        domain, u_disp, material_behavior, mesh_nx, mesh_ny, mesh_nz,
        element_type, num_increments, res_x=51, res_y=51, res_z=51):
    """Interpolate displacement field at grid points for visualization.

    For single element meshes, interpolates displacement field at
    specified grid resolution.

    Args:
        domain: FEM domain object with nodes, elements, and element
            type attributes.
        u_disp (Tensor): Displacement field tensor of shape
            (num_increments, n_nodes, dim).
        material_behavior (str): Type of material ('elastic',
            'elastoplastic', etc.).
        mesh_nx (int): Number of elements in x direction.
        mesh_ny (int): Number of elements in y direction.
        mesh_nz (int): Number of elements in z direction.
        element_type (str): Type of element ('quad4', 'hex8', etc.).
        num_increments (int): Number of load increments.
        res_x (int): Grid resolution in x direction. Defaults to 51.
        res_y (int): Grid resolution in y direction. Defaults to 51.
        res_z (int): Grid resolution in z direction. Defaults to 51.

    Returns:
        tuple: For 2D: (coords_interp_x, coords_interp_y, u_interp_x,
            u_interp_y, node_coords, u_final) where each coordinate and
            displacement array has shape (res_x, res_y), node_coords
            has shape (n_nodes, 2), and u_final has shape (n_nodes, 2).
            For 3D: (coords_interp_x, coords_interp_y, coords_interp_z,
            u_interp_x, u_interp_y, u_interp_z, node_coords, u_final)
            where each array has shape (res_x, res_y, res_z) except
            node_coords (n_nodes, 3) and u_final (n_nodes, 3).
            Returns None if conditions for interpolation are not met.
    """
    # Determine problem dimension
    dim = domain.nodes.shape[1]

    if num_increments != 1:
        return None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2D case
    if dim == 2:
        if not (material_behavior == 'elastic' and mesh_nx == 1 and
                mesh_ny == 1 and element_type == 'quad4'):
            return None

        # Create grid in natural coordinates [-1, 1] x [-1, 1]
        xi_grid = torch.linspace(-1, 1, res_x)
        eta_grid = torch.linspace(-1, 1, res_y)
        xi_mesh, eta_mesh = torch.meshgrid(
            xi_grid, eta_grid, indexing='ij')

        # Flatten for easier processing
        xi_flat = xi_mesh.flatten()
        eta_flat = eta_mesh.flatten()
        xi_points = torch.stack([xi_flat, eta_flat], dim=-1)

        # Evaluate shape functions at grid points
        N_grid = domain.etype.N(xi_points)

        # Get node coordinates and displacements for single element
        element_nodes = domain.elements[0]
        node_coords = domain.nodes[element_nodes]

        # Interpolate coordinates at grid points
        coords_interp = torch.einsum('ij,jk->ik', N_grid, node_coords)

        # Only final displacements
        u_final = u_disp[-1, element_nodes, :]
        u_interp = torch.einsum('ij,jk->ik', N_grid, u_final)

        # Reshape back to grid
        u_interp_x = u_interp[:, 0].reshape(res_x, res_y)
        u_interp_y = u_interp[:, 1].reshape(res_x, res_y)
        coords_interp_x = coords_interp[:, 0].reshape(res_x, res_y)
        coords_interp_y = coords_interp[:, 1].reshape(res_x, res_y)

        return (coords_interp_x, coords_interp_y, u_interp_x, u_interp_y,
                node_coords, u_final)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3D case
    elif dim == 3:
        if not (material_behavior == 'elastic' and mesh_nx == 1 and
                mesh_ny == 1 and mesh_nz == 1 and
                element_type == 'hex8'):
            return None

        # Create grid in natural coordinates [-1, 1]^3
        xi_grid = torch.linspace(-1, 1, res_x)
        eta_grid = torch.linspace(-1, 1, res_y)
        zeta_grid = torch.linspace(-1, 1, res_z)
        xi_mesh, eta_mesh, zeta_mesh = torch.meshgrid(
            xi_grid, eta_grid, zeta_grid, indexing='ij')

        # Flatten for easier processing
        xi_flat = xi_mesh.flatten()
        eta_flat = eta_mesh.flatten()
        zeta_flat = zeta_mesh.flatten()
        xi_points = torch.stack([xi_flat, eta_flat, zeta_flat], dim=-1)

        # Evaluate shape functions at grid points
        N_grid = domain.etype.N(xi_points)

        # Get node coordinates and displacements for single element
        element_nodes = domain.elements[0]
        node_coords = domain.nodes[element_nodes]

        # Interpolate coordinates at grid points
        coords_interp = torch.einsum('ij,jk->ik', N_grid, node_coords)

        # Only final displacements
        u_final = u_disp[-1, element_nodes, :]
        u_interp = torch.einsum('ij,jk->ik', N_grid, u_final)

        # Reshape back to grid
        u_interp_x = u_interp[:, 0].reshape(res_x, res_y, res_z)
        u_interp_y = u_interp[:, 1].reshape(res_x, res_y, res_z)
        u_interp_z = u_interp[:, 2].reshape(res_x, res_y, res_z)
        coords_interp_x = coords_interp[:, 0].reshape(res_x, res_y, res_z)
        coords_interp_y = coords_interp[:, 1].reshape(res_x, res_y, res_z)
        coords_interp_z = coords_interp[:, 2].reshape(res_x, res_y, res_z)

        return (coords_interp_x, coords_interp_y, coords_interp_z,
                u_interp_x, u_interp_y, u_interp_z,
                node_coords, u_final)

    else:
        return None
# -----------------------------------------------------------------------------
def compute_strain_energy_density(
        sigma, def_grad, material_behavior, dim):
    """Compute strain energy density from stress and deformation gradient.

    For elastic materials: W = 0.5 * sigma : epsilon
    For hyperelastic materials: W is computed from strain energy function
    For elastoplastic materials: W = 0.5 * sigma : epsilon_elastic

    Args:
        sigma (Tensor): Stress tensor of shape (num_increments, num_elem,
            dim, dim) or (num_increments, dim, dim).
        def_grad (Tensor): Deformation gradient of shape
            (num_increments, num_elem, dim, dim) or
            (num_increments, dim, dim).
        material_behavior (str): Material type, one of 'elastic',
            'hyperelastic', or 'elastoplastic'.
        dim (int): Spatial dimension (2 or 3).

    Returns:
        Tensor: Strain energy density of shape (num_increments, num_elem).

    Raises:
        ValueError: If material_behavior is not recognized.
    """
    # Handle case where element dimension is missing (single element)
    if sigma.dim() == 3:
        # (num_increments, dim, dim) -> (num_increments, 1, dim, dim)
        sigma = sigma.unsqueeze(1)
    if def_grad.dim() == 3:
        # (num_increments, dim, dim) -> (num_increments, 1, dim, dim)
        def_grad = def_grad.unsqueeze(1)

    if material_behavior in ['elastic', 'elastoplastic_nlh', 
                             'elastoplastic_lh']:
        # Compute strain tensor from deformation gradient
        # Small strain: epsilon = 0.5 * (F + F^T) - I
        I = torch.eye(dim, device=def_grad.device, dtype=def_grad.dtype)
        I = I.expand_as(def_grad)
        epsilon = 0.5 * (def_grad + def_grad.transpose(-2, -1)) - I

        # Strain energy density: W = 0.5 * sigma : epsilon
        # Using Einstein summation for tensor contraction
        strain_energy_density = 0.5 * torch.einsum(
            '...ij,...ij->...', sigma, epsilon)

    elif material_behavior == 'hyperelastic':
        # For hyperelastic materials, compute from deformation gradient
        # W = psi(F) where psi is the strain energy function
        # For Neo-Hookean:
        # W = mu/2 * (tr(C) - 3) - mu*ln(J) + lambda/2 * (ln(J))^2
        # where C = F^T * F is right Cauchy-Green tensor

        C = torch.matmul(def_grad.transpose(-2, -1), def_grad)
        J = torch.det(def_grad)
        log_J = torch.log(torch.clamp(J, min=1e-8))

        # Material parameters (should ideally come from material
        # definition). Using typical values for Neo-Hookean material
        if dim == 2:
            # For plane strain, simplified calculation for 2D
            trace_C = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
            strain_energy_density = 0.1 * (trace_C - 2.0)
        else:
            trace_C = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
            # Neo-Hookean parameters (should match material definition)
            mu = 1000.0
            lambda_param = 1000.0
            strain_energy_density = (
                (mu / 2.0) * (trace_C - 3.0) - mu * log_J +
                (lambda_param / 2.0) * log_J**2)
    else:
        raise ValueError(f"Unknown material behavior: {material_behavior}")

    return strain_energy_density
# =============================================================================
