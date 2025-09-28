import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import sys
import pathlib
import pickle as pkl

# Add graphorge to sys.path
graphorge_path = str(
    pathlib.Path(__file__).parents[2] / "graphorge_material_patches" / "src")
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path) 

import torch

from torchfem import Solid, Planar
from torchfem.materials import (IsotropicElasticityPlaneStrain, 
                                IsotropicElasticityPlaneStress,
                                IsotropicPlasticityPlaneStrain,
                                IsotropicPlasticityPlaneStress,
                                IsotropicElasticity3D,
                                IsotropicPlasticity3D,
                                IsotropicHencky3D,
                                Hyperelastic3D,
                                IsotropicHenckyPlaneStrain)
from torchfem.mesh import cube_hexa, rect_quad
from torchfem.elements import linear_to_quadratic


from utils import prescribe_disps_by_coords, plot_displacement_field, \
    compute_strain_energy_density

torch.set_default_dtype(torch.float64)

def run_simulation(
    element_type = 'quad4',
    material_behavior = 'elastic',
    patch_idx = 0,
    num_increments = 1,
    mesh_nx = 3, mesh_ny = 3, mesh_nz = 3, 
    filepath = f'/Users/rbarreira/Desktop/machine_learning/' + \
            f'material_patches/_data/'):
    
    if element_type in ['quad4', 'tri3', 'tetra4', 'hex8']:
        elem_order = 1
    elif element_type in ['quad8', 'tri6', 'tetra10', 'hex20']:
        elem_order = 2

    if element_type in ['quad4', 'tri3', 'quad8', 'tri6']:
        dim = 2
    elif element_type in ['tetra4', 'hex8', 'tetra10', 'hex20']:
        dim = 3
        
    #%% ---------------------- Read material patch input ----------------------
    out_filepath = f'/Users/rbarreira/Desktop/machine_learning/' + \
        f'material_patches/'
    if dim == 2:
        dir_path = out_filepath + f'_data/{material_behavior}/{dim}d/' + \
            f'{element_type}/mesh_{mesh_nx}x{mesh_ny}/ninc{num_increments}/'
    elif dim == 3:
        dir_path = out_filepath + f'_data/{material_behavior}/{dim}d/' + \
            f'{element_type}/mesh{mesh_nx}x{mesh_ny}x{mesh_nz}/' + \
                f'ninc{num_increments}/'
    os.makedirs(dir_path, exist_ok=True)

    if dim == 2:
        input_filename = filepath + \
            f'material_patches_generation_{dim}d_' + \
            f'{element_type}_mesh_{mesh_nx}x{mesh_ny}/' + \
            f'material_patch_{patch_idx}/material_patch/' + \
            f'material_patch_attributes.pkl'
        
        output_filename = dir_path + f'matpatch_idx{patch_idx}.pkl'
        
    elif dim == 3: 
        if element_type not in ['tetra4', 'hex8', 'tetra10', 'hex20']:
            raise ValueError(f"Wrong element type for {dim}d problem!")
        input_filename = filepath + \
            f'material_patches_generation_{dim}d_' + \
            f'{element_type}_mesh{mesh_nx}x{mesh_ny}x{mesh_nz}/' + \
            f'material_patch_{patch_idx}/material_patch/' + \
            f'material_patch_attributes.pkl'
        
        output_filename = dir_path + f'matpatch_idx{patch_idx}.pkl'

    with open(input_filename, 'rb') as file:
        matpatch = pkl.load(file)

    #%% -------------------------- Constitutive law ---------------------------
    if material_behavior == 'elastic':

        e_young = 110000
        nu = 0.33
        
        if dim == 2:
            material = IsotropicElasticityPlaneStrain(E=e_young, nu=nu)
        elif dim == 3:
            material = IsotropicElasticity3D(E=e_young, nu=nu)
    elif material_behavior == 'hyperelastic':

        e_young = 20000
        nu = 0.33

        lmbda =  e_young * nu / ((1. + nu) * (1. - 2. * nu))
        mu =  e_young / (2. * (1. + nu))

        def psi(F):
            """
            Neo-Hookean strain energy density function.
            """
            # Compute the right Cauchy-Green deformation tensor
            C = F.transpose(-1, -2) @ F
            # Stable computation of the logarithm of the determinant
            logJ = 0.5 * torch.logdet(C)
            return mu / 2 * (torch.trace(C) - 3.0) - mu * logJ + \
                lmbda / 2 * logJ**2
        
        if dim == 2:
            material = IsotropicHenckyPlaneStrain(E=e_young, nu=nu)
        elif dim == 3:
            # material = IsotropicHencky3D(E=e_young, nu=nu, n_state=0)
            material = Hyperelastic3D(psi)

    elif material_behavior == 'elastoplastic':

        e_young = 210000
        nu = 0.33

        sigma_y = 100.0
        hardening_modulus = 100.0
        
        # Hardening function
        def sigma_f(q):
            return sigma_y + hardening_modulus * q

        # Derivative of the hardening function
        def sigma_f_prime(q):
            return hardening_modulus
        
        if dim == 2:
            # Remember:
            # state (Tensor): Internal state variables, here: 
            # equivalent plastic strain and stress in the third direction. 
            # Shape: `(..., 2)`.
            material = IsotropicPlasticityPlaneStrain(
                E=e_young, nu=nu, sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)
        elif dim ==3:
            material = IsotropicPlasticity3D(
                E=e_young, nu=nu, sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)

    #%% --------------------------- Geometry & mesh ---------------------------
    if dim == 2:
        # cube_hexa takes in as arguments the number of nodes per direction
        nodes, elements = rect_quad(mesh_nx + 1, mesh_ny + 1)
        if elem_order == 2:
            nodes, elements = linear_to_quadratic(nodes, elements)

        domain = Planar(nodes, elements, material)
    elif dim == 3:
        # cube_hexa takes in as arguments the number of nodes per direction
        nodes, elements = cube_hexa(mesh_nx + 1, mesh_ny + 1, mesh_nz + 1)
        if elem_order == 2:
            nodes, elements = linear_to_quadratic(nodes, elements)

        domain = Solid(nodes, elements, material)

    #%% --------------------------- Data structure ----------------------------
    # Initialize the main data structure to save
    simulation_data = {
        'bd_nodes_coords': {},
        'bd_nodes_disps_time_series': {},
        'bd_nodes_forces_time_series': {},
        'stress_avg': {},
        'strain_energy_density': {},
    }

    # Save interpolated coordinates and displacements
    # if material_behavior == 'elastic' and \
    #     mesh_nx == 1 and mesh_ny == 1 and element_type == 'quad4':
    #     simulation_data['def_grad_avg'] = {}
    #     simulation_data['interpolated_coords'] = None
    #     simulation_data['interpolated_displacement'] = None
    if material_behavior == 'elastoplastic':
        simulation_data['epsilon_pl_eq'] = {}


    node_label_to_torchfem_idx = {}
    # Build mapping for ALL nodes, not just boundary nodes
    for node_label in matpatch['mesh_nodes_coords_ref'].keys():
        # Node coords in material patch global system 
        node_coords_matp = matpatch['mesh_nodes_coords_ref'][node_label]
        ref_point = torch.tensor(node_coords_matp)
        # Find the global idx of the corresponding node in the torch-fem mesh
        distances = torch.sqrt(torch.sum((nodes - ref_point)**2, axis=1))
        closest_idx = torch.argmin(distances).item()
        # Ensure distance is small; if it is large, break!
        if distances[closest_idx] >= 1e-6:
            break
        node_label_to_torchfem_idx[int(node_label)] = closest_idx
    
    # Initialize boundary node coordinates and time series arrays
    for node_label in matpatch['mesh_boundary_nodes_disps'].keys():
        closest_idx = node_label_to_torchfem_idx[int(node_label)]
        # Store coordinates in simulation framework, not material patch
        simulation_data['bd_nodes_coords'][closest_idx] = matpatch[
            'mesh_nodes_coords_ref'][node_label]
        # Initialize time series arrays for displacements and forces
        simulation_data['bd_nodes_disps_time_series'][closest_idx] = []
        simulation_data['bd_nodes_forces_time_series'][closest_idx] = []

    # Copy all matpatch data except excluded fields
    excluded_fields = {'mesh_boundary_nodes_disps', 
                       'load_factor_time_series', 
                       'mesh_nodes_coords_ref',
                       'mesh_boundary_nodes_disps_time'}
    for key, value in matpatch.items():
        if key not in excluded_fields:
            if key == 'mesh_nodes_matrix':
                # Create proper mesh nodes matrix based on mesh structure
                # 2D: Rows = vertical (i-direction),
                # Columns = horizontal (j-direction)
                # 3D: First dimension = i-direction,
                # Second = j-direction,
                # Third = k-direction
                if dim == 2:
                    if elem_order == 1:
                        # Linear elements:
                        # (mesh_nx+1)x(mesh_ny+1) nodes
                        mesh_nodes_matrix = np.zeros(
                            (mesh_nx + 1, mesh_ny + 1), dtype=int)
                        node_idx = 0
                        for i in range(mesh_nx + 1):
                            for j in range(mesh_ny + 1):
                                mesh_nodes_matrix[i, j] = node_idx
                                node_idx += 1
                    else:  # elem_order == 2
                        # Quadratic elements:
                        # (2*mesh_nx+1)x(2*mesh_ny+1) nodes
                        mesh_nodes_matrix = np.zeros(
                            (2*mesh_nx + 1, 2*mesh_ny + 1), dtype=int)
                        node_idx = 0
                        for i in range(2*mesh_nx + 1):
                            for j in range(2*mesh_ny + 1):
                                mesh_nodes_matrix[i, j] = node_idx
                                node_idx += 1               
                elif dim == 3:
                    if elem_order == 1:
                        # Linear elements:
                        # (mesh_nx+1)x(mesh_ny+1)x(mesh_nz+1) nodes
                        mesh_nodes_matrix = np.zeros(
                            (mesh_nx + 1, mesh_ny + 1, mesh_nz + 1), dtype=int)
                        node_idx = 0
                        for i in range(mesh_nx + 1):
                            for j in range(mesh_ny + 1):
                                for k in range(mesh_nz + 1):
                                    mesh_nodes_matrix[i, j, k] = node_idx
                                    node_idx += 1
                    else:  # elem_order == 2
                        # Quadratic elements:
                        # (2*mesh_nx+1)x(2*mesh_ny+1)x(2*mesh_nz+1) nodes
                        mesh_nodes_matrix = np.zeros(
                            (2*mesh_nx + 1, 2*mesh_ny + 1, 2*mesh_nz + 1),
                            dtype=int)
                        node_idx = 0
                        for i in range(2*mesh_nx + 1):
                            for j in range(2*mesh_ny + 1):
                                for k in range(2*mesh_nz + 1):
                                    mesh_nodes_matrix[i, j, k] = node_idx
                                    node_idx += 1
                simulation_data['mesh_nodes_matrix'] = torch.tensor(
                    mesh_nodes_matrix)
            else:
                simulation_data[key] = value
    #%% ------------------------- Boundary conditions -------------------------
    num_applied_disps, nodes_constrained = prescribe_disps_by_coords(
        domain=domain, data=matpatch, dim=dim)

    #%% -------------------------- Solve the system ---------------------------
    if num_increments == 1:
        increments = torch.linspace(0.0, 1.0, 2)
    elif num_increments > 1:
        increments = torch.tensor(matpatch['load_factor_time_series'])

    u_disp, f_int, sigma_out, def_grad, alpha_out, vol_elem = domain.solve(
        increments=increments, return_intermediate=True,
        aggregate_integration_points=True, return_volumes=True)
    

    # if torch.mean(alpha_out) > 0:
    #     print(f'{idx}: plastic strain: {torch.mean(alpha_out[-1,:,:])}')

    # Volume-weighted averaging instead of simple mean
    # Shape: (num_increments, 1)
    total_volume = vol_elem.sum(dim=1, keepdim=True) 
    # Shape: (num_increments, num_elem)
    vol_weights = vol_elem / total_volume
    # Volume-weighted average for sigma_out
    # sigma_out shape: (num_increments, num_elem, num_stress, num_stress) 
    # averaged over an element's integ. points;
    # num_stress = 3 for 3D
    # vol_weights shape: (num_increments, num_elem)
    vol_weights_expanded = vol_weights.unsqueeze(-1).unsqueeze(-1) 
    # sigma_out_avg shape: (num_increments, num_stress, num_stress) 
    if num_increments == 1 and dim == 2:
        sigma_out_avg = (sigma_out[-1,:,:] * vol_weights_expanded)
    elif num_increments == 1 and dim == 3:
        sigma_out_avg = (sigma_out[-1,:,:] * vol_weights_expanded)
    else:
        sigma_out_avg = (sigma_out * vol_weights_expanded)
        #.sum(dim=1)
    # def_grad shape: (num_increments, num_elem, num_stress, num_stress)
    
    # def_grad_avg shape: (num_increments, num_stress, num_stress)
    def_grad_avg = (def_grad * vol_weights_expanded) #.sum(dim=1)
    if material_behavior == 'elastoplastic':
        # def_grad shape: (num_increments, num_elem, num_state), 
        # num_state = 2 for plane strain, 1 for pure plasticity
        alpha_out_avg = (alpha_out[:,:,0] * vol_weights).sum(dim=1)

    #%% -------------------- Strain Energy Density Calculation ----------------
    # Compute strain energy density for each element at each time step
    strain_energy_density = compute_strain_energy_density(
        sigma_out, def_grad, material_behavior, dim)
    
    # Volume-weighted total strain energy
    total_strain_energy = (strain_energy_density * vol_weights).sum(dim=1)
    #%% ------------------- Interpolate displacement field --------------------
    # For 1x1 quad4 mesh, interpolate displacement field at 50x50 grid points
    # if material_behavior == 'elastic' and \
    #     mesh_nx == 1 and mesh_ny == 1 and element_type == 'quad4':
    #     # Create 50x50 grid in natural coordinates [-1, 1] x [-1, 1]
    #     xi_grid = torch.linspace(-1, 1, 51)
    #     eta_grid = torch.linspace(-1, 1, 51)
    #     xi_mesh, eta_mesh = torch.meshgrid(xi_grid, eta_grid, indexing='ij')
        
    #     # Flatten for easier processing
    #     xi_flat = xi_mesh.flatten()
    #     eta_flat = eta_mesh.flatten()
    #     # Shape: (2500, 2)
    #     xi_points = torch.stack([xi_flat, eta_flat], dim=-1)  
        
    #     # Evaluate shape functions at grid points
    #     # Shape: (2500, 4)
    #     N_grid = domain.etype.N(xi_points)  
        
    #     # Get node coordinates and displacements for the single element
    #     # Shape: (4,)
    #     element_nodes = domain.elements[0]  
    #     # Shape: (4, 2)
    #     node_coords = domain.nodes[element_nodes]  
        
    #     # Interpolate coordinates at grid points
    #     # Shape: (2500, 2)
    #     coords_interp = torch.einsum('ij,jk->ik', N_grid, node_coords)  
        
    #     # Initialize displacement interpolation arrays
    #     if num_increments == 1:
    #         # Only final displacements
    #         # Shape: (4, 2)
    #         u_final = u_disp[-1, element_nodes, :]  
    #         # Shape: (2500, 2)
    #         u_interp = torch.einsum('ij,jk->ik', N_grid, u_final)  
            
    #         # Reshape back to 50x50 grid
    #         u_interp_x = u_interp[:, 0].reshape(51, 51)
    #         u_interp_y = u_interp[:, 1].reshape(51, 51)
    #         coords_interp_x = coords_interp[:, 0].reshape(51, 51)
    #         coords_interp_y = coords_interp[:, 1].reshape(51, 51)
                    
    #         # Stack along last dimension to get (50, 50, 2) shape
    #         # interpolated_coords: Shape (50, 50, 2): 
    #         # Physical coordinates at each grid point
    #         # [:, :, 0] = x-coordinates
    #         # [:, :, 1] = y-coordinates
    #         # interpolated_displacement: Shape (50, 50, 2):
    #         # Displacement field at each grid point
    #         # [:, :, 0] = x-displacement
    #         # [:, :, 1] = y-displacement
    #         # Create arrays with proper shape (50, 50, 2)
    #         coords_combined = np.zeros((51, 51, 2))
    #         coords_combined[:, :, 0] = coords_interp_x.detach().cpu().numpy()
    #         coords_combined[:, :, 1] = coords_interp_y.detach().cpu().numpy()
        
    #         u_combined = np.zeros((51, 51, 2))
    #         u_combined[:, :, 0] = u_interp_x.detach().cpu().numpy()
    #         u_combined[:, :, 1] = u_interp_y.detach().cpu().numpy()
            
    #         simulation_data['interpolated_coords'] = coords_combined
    #         simulation_data['interpolated_displacement'] = u_combined
            
            # # Plot displacement field
            # plot_displacement_field(
            #     coords_interp_x.detach().cpu().numpy(),
            #     coords_interp_y.detach().cpu().numpy(),
            #     u_interp_x.detach().cpu().numpy(),
            #     u_interp_y.detach().cpu().numpy(),
            #     node_coords.detach().cpu().numpy(),
            #     u_final.detach().cpu().numpy(),
            #     output_filename.replace('.pkl', '_displacement_field.png')
            # )

            # -----------------------------------------------------------------
            # #  Plot using the built-in planar.py plot function
            # # Compute displacement magnitude at nodes
            # u_magnitude = torch.sqrt(u_disp[-1, :, 0]**2 + u_disp[-1, :, 1]**2)
            
            # # Plot displacement magnitude using domain.plot()
            # domain.plot(
            #     # Final displacement field (deformed config.)
            #     u=u_disp[-1], 
            #     # Displacement magnitude at nodes
            #     node_property=u_magnitude,  
            #     title='Displacement Magnitude',
            #     colorbar=True,
            #     figsize=(6, 6),
            #     cmap='viridis'
            # )
            # plt.savefig(output_filename.replace(
            #     '.pkl', '_torchfem_magnitude.png'), 
            #            dpi=300, bbox_inches='tight')
            # plt.close()
            
            # # Plot X-displacement using domain.plot()
            # domain.plot(
            #     # Final displacement field (deformed config)
            #     u=u_disp[-1],  
            #     # X-displacement at nodes
            #     node_property=u_disp[-1, :, 0],  
            #     title='X-Displacement',
            #     colorbar=True,
            #     figsize=(6, 6),
            #     cmap='RdBu_r'
            # )
            # plt.savefig(output_filename.replace(
            #     '.pkl', '_torchfem_x_displacement.png'), 
            #            dpi=300, bbox_inches='tight')
            # plt.close()
            
            # # Plot Y-displacement using domain.plot()
            # domain.plot(
            #     # Final displacement field (deformed config)
            #     u=u_disp[-1],
            #     # Y-displacement at nodes
            #     node_property=u_disp[-1, :, 1], 
            #     title='Y-Displacement',
            #     colorbar=True,
            #     figsize=(6, 6),
            #     cmap='RdBu_r'
            # )
            # plt.savefig(output_filename.replace(
            #     '.pkl', '_torchfem_y_displacement.png'), 
            #            dpi=300, bbox_inches='tight')
            # plt.close()
            # # -----------------------------------------------------------------

            # # Plot shape functions over the element
            # # Create a higher resolution grid
            # xi_fine = torch.linspace(-1, 1, 100)
            # eta_fine = torch.linspace(-1, 1, 100)
            # xi_mesh_fine, eta_mesh_fine = torch.meshgrid(
            #     xi_fine, eta_fine, indexing='ij')
            
            # # Flatten for shape function evaluation
            # xi_flat_fine = xi_mesh_fine.flatten()
            # eta_flat_fine = eta_mesh_fine.flatten()
            # xi_points_fine = torch.stack([xi_flat_fine, eta_flat_fine], dim=-1)
            
            # # Evaluate shape functions at grid points
            # # Shape: (10000, 4)
            # N_fine = domain.etype.N(xi_points_fine)  
            
            # # Interpolate physical coordinates
            # coords_fine = torch.einsum('ij,jk->ik', N_fine, node_coords)
            # coords_fine_x = coords_fine[:, 0].reshape(100, 100)
            # coords_fine_y = coords_fine[:, 1].reshape(100, 100)
            
            # # Plot all 4 shape functions
            # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            # axes = axes.flatten()
            
            # shape_func_names = ['N₁', 'N₂', 'N₃', 'N₄']
            
            # for i in range(4):
            #     # Reshape shape function values to grid
            #     N_i = N_fine[:, i].reshape(100, 100)
                
            #     im = axes[i].contourf(coords_fine_x.detach().cpu().numpy(), 
            #                          coords_fine_y.detach().cpu().numpy(), 
            #                          N_i.detach().cpu().numpy(), 
            #                          levels=20, cmap='viridis')
                
            #     axes[i].set_title(f'Shape Function {shape_func_names[i]}')
            #     axes[i].set_xlabel('X')
            #     axes[i].set_ylabel('Y')
            #     axes[i].set_aspect('equal')
        
            #     plt.colorbar(im, ax=axes[i])
                
            #     # Plot element boundary
            #     element_x = [node_coords[0,0], node_coords[1,0],
            #                  node_coords[2,0], node_coords[3,0],
            #                  node_coords[0,0]]
            #     element_y = [node_coords[0,1], node_coords[1,1],
            #                  node_coords[2,1], node_coords[3,1],
            #                  node_coords[0,1]]
            #     axes[i].plot(element_x, element_y, 'k-', linewidth=2)
                
            #     # Highlight the node where shape function = 1
            #     axes[i].scatter(node_coords[i,0], node_coords[i,1], 
            #                    c='red', s=100, zorder=5, 
            #                    edgecolors='black', linewidth=2)
                
            #     # Add other nodes
            #     other_nodes = [j for j in range(4) if j != i]
            #     axes[i].scatter(
            #         node_coords[other_nodes,0], node_coords[other_nodes,1], 
            #                    c='white', s=50, zorder=5, edgecolors='black')
            
            # plt.tight_layout()
            # plt.savefig(output_filename.replace('.pkl', '_shape_functions.png'), 
            #            dpi=300, bbox_inches='tight')
            # plt.close()
        
    
    #%% ----------------------------- Save output -----------------------------
    # Save displacements, internal forces of the boundary nodes (the ones 
    # constrained), volume-averaged stress, and strain
    for idx_node in range(u_disp.shape[1]):
        if idx_node in nodes_constrained:
            if num_increments == 1:
                simulation_data['bd_nodes_disps_time_series'][
                    idx_node] = u_disp[-1, idx_node, :]
                simulation_data['bd_nodes_forces_time_series'][
                    idx_node] = f_int[-1, idx_node, :]
            else:
                simulation_data['bd_nodes_disps_time_series'][
                    idx_node] = u_disp[:, idx_node, :]
                simulation_data['bd_nodes_forces_time_series'][
                    idx_node] = f_int[:, idx_node, :]
            
            # Print for user validation at last time step
            # print(f'Node: {idx_node}: u {u_disp[-1,idx_node,:]}')
            # print(f'          force {f_int[-1, idx_node, :]}')
    if num_increments == 1:
        simulation_data['stress_avg'] = sigma_out_avg[
            -1, 0]
        simulation_data['strain_energy_density'] = \
            total_strain_energy[-1]
        # simulation_data['def_grad_avg'] = def_grad_avg[
        #     -1, :, :]
        if material_behavior == 'elastoplastic':
                simulation_data['epsilon_pl_eq'] = alpha_out_avg[
                -1]
    else: 
        for idx_time in range(num_increments + 1):
            # simulation_data['stress_avg'][idx_time] = sigma_out_avg[
            #     idx_time, :, :]
            # simulation_data['def_grad_avg'][idx_time] = def_grad_avg[
            #     idx_time, :, :]
            simulation_data['strain_energy_density'][idx_time] = \
                total_strain_energy[idx_time]
            if material_behavior == 'elastoplastic':
                simulation_data['epsilon_pl_eq'][idx_time] = (alpha_out_avg[
                idx_time])
                # print(alpha_out_avg[idx_time, :, 0])
            

    # Save the complete simulation data to a pickle file
    # try:
    #     with open(output_filename, 'wb') as f:
    #         pkl.dump(simulation_data, f, protocol=pkl.HIGHEST_PROTOCOL)
    # except Exception as excp:
    #     print(f"Error saving simulation data: {excp}")

if __name__ == '__main__':

    # filepath = f'/Volumes/T7/material_patches/'
    filepath = f'/Users/rbarreira/Desktop/machine_learning/' + \
            f'material_patches/_input_material_patches/'
    
    for idx in range(0, 1):
        if idx % 100 == 0:
            print(f'     {idx}')
            
        run_simulation(
            element_type = 'quad4', material_behavior = 'elastic', 
            num_increments = 1, patch_idx=idx, filepath=filepath,
            mesh_nx=1, mesh_ny=1, mesh_nz=1)