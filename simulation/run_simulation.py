import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

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

torch.set_default_dtype(torch.float64)

#%% ------------------------------- User inputs -------------------------------
dim = 2
material_behavior = 'elastoplastic'
e_young = 110000
nu = 0.33
sigma_y = 100.0
hardening_modulus = 100.0

element_type = 'quad4'
patch_idx = 0
num_increments = 1
# number of elements per direction
mesh_nx = 3
mesh_ny = 3
mesh_nz = 3
filepath = f'/Users/rbarreira/Desktop/machine_learning/' + \
            f'material_patches/2025_06_05/'

#%% ------------------------ Read material patch input ------------------------
dir_path = filepath + f'_data/{material_behavior}_{dim}_' + \
    f'{element_type}_ninc{num_increments}/'
os.makedirs(dir_path, exist_ok=True)


if element_type in ['quad4', 'tri3', 'tetra4', 'hex8']:
    elem_order = 1
elif element_type in ['quad8', 'tri6', 'tetra10', 'hex20']:
    elem_order = 2

if dim == 2:
    if element_type not in ['quad4', 'tri3', 'quad8', 'tri6']:
        raise ValueError(f"Wrong element type for {dim}d problem!")
    input_filename = filepath + \
        f'material_patches_generation_{dim}d_' + \
        f'{element_type}_mesh_{mesh_nx}x{mesh_ny}/' + \
        f'material_patch_{patch_idx}/material_patch/' + \
        f'material_patch_attributes.pkl'
    
    output_filename = dir_path + f'matpatch_{dim}d_' + \
        f'mesh{mesh_nx}x{mesh_ny}_{element_type}_idx{patch_idx}.pkl'
    
elif dim == 3: 
    if element_type not in ['tetra4', 'hex8', 'tetra10', 'hex20']:
        raise ValueError(f"Wrong element type for {dim}d problem!")
    input_filename = filepath + \
        f'material_patches_generation_{dim}d_' + \
        f'{element_type}_mesh_{mesh_nx}x{mesh_ny}x{mesh_nz}/' + \
        f'material_patch_{patch_idx}/material_patch/' + \
        f'material_patch_attributes.pkl'
    
    output_filename = dir_path + f'matpatch_{dim}d_' + \
        f'mesh{mesh_nx}x{mesh_ny}x{mesh_nz}_{element_type}_' + \
            f'idx{patch_idx}.pkl'

with open(input_filename, 'rb') as file:
    matpatch = pkl.load(file)

#%% ---------------------------- Constitutive law -----------------------------
if material_behavior == 'elastic':
    if dim == 2:
        material = IsotropicElasticityPlaneStrain(E=e_young, nu=nu)
    elif dim == 3:
        material = IsotropicElasticity3D(E=e_young, nu=nu)
elif material_behavior == 'hyperelastic':
    if dim == 2:
        material = IsotropicHenckyPlaneStrain(E=e_young, nu=nu, n_state=0)
    elif dim == 3:
        material = IsotropicHencky3D(E=e_young, nu=nu, n_state=0)
elif material_behavior == 'elastoplastic':
    # Hardening function
    def sigma_f(q):
        return sigma_y + hardening_modulus * q

    # Derivative of the hardening function
    def sigma_f_prime(q):
        return hardening_modulus
    
    if dim == 2:
        material = IsotropicPlasticityPlaneStrain(
            E=e_young, nu=nu, sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)
    elif dim ==3:
        material = IsotropicPlasticity3D(
            E=e_young, nu=nu, sigma_f=sigma_f, sigma_f_prime=sigma_f_prime)

#%% ----------------------------- Geometry & mesh -----------------------------
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

#%% ----------------------------- Data structure ------------------------------
# Initialize the main data structure to save
simulation_data = {
    'boundary_nodes_coords': {},
    'boundary_nodes_disps_time_series': {},
    'boundary_nodes_forces_time_series': {},
}


# Initialize boundary node coordinates from fe_space coordinates:
for node_label in matpatch['mesh_boundary_nodes_disps'].keys():

    # Node coords in material patch global system 
    node_coords_matp = matpatch['mesh_nodes_coords_ref'][node_label]
    ref_point = torch.tensor(node_coords_matp)
    # Find the global idx of the corresponding node in the torch-fem mesh
    distances = torch.sqrt(torch.sum((nodes - ref_point)**2, axis=1))
    closest_idx = torch.argmin(distances)
    
    # Ensure distance is small; if it is large, break!
    if distances[closest_idx] >= 1e-6:
        break
    
    # Store coordinates in simulation framework, not material patch
    simulation_data['boundary_nodes_coords'][closest_idx] = matpatch[
        'mesh_nodes_coords_ref'][node_label]
    # Initialize time series arrays for displacements and forces
    simulation_data['boundary_nodes_disps_time_series'][closest_idx] = []
    simulation_data['boundary_nodes_forces_time_series'][closest_idx] = []



#%% --------------------------- Boundary conditions ---------------------------

def apply_displacements_by_coordinates(domain, data, dim):
    """
    Apply displacements by matching mesh coordinates with 
    reference coordinates read from material patch file.
    """

    mesh_coords_ref = data.get('mesh_nodes_coords_ref', {})
    boundary_disps = data.get('mesh_boundary_nodes_disps', {})
    
    nodes = domain.nodes.numpy()
    tolerance = 1e-6
    
    num_applied_disps = 0

    nodes_constrained = []
    
    for node_str, ref_coord in mesh_coords_ref.items():
        if node_str in boundary_disps:
            ref_point = np.array(ref_coord)
            
            # Find matching node in the mesh
            distances = np.sqrt(np.sum((nodes - ref_point)**2, axis=1))
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < tolerance:
                disp_data = boundary_disps[node_str]

                # Syntax: displacement at [Node_ID, DOF]
                # Constrained displacement at [Node_IDs, DOFs]
                if dim == 2:
                    # Apply displacement for first time step
                    domain.displacements[closest_idx, 0] = float(
                        disp_data[0])
                    domain.displacements[closest_idx, 1] = float(
                        disp_data[1])
                    
                    # Set constraints
                    domain.constraints[closest_idx, 0] = True
                    domain.constraints[closest_idx, 1] = True
                    
                    num_applied_disps += 1

                    nodes_constrained.append(closest_idx)

                elif dim == 3:
                    # Apply displacement for first time step
                    domain.displacements[closest_idx, 0] = float(
                        disp_data[0])
                    domain.displacements[closest_idx, 1] = float(
                        disp_data[1])
                    domain.displacements[closest_idx, 2] = float(
                        disp_data[2])
                    
                    # Set constraints
                    domain.constraints[closest_idx, 0] = True
                    domain.constraints[closest_idx, 1] = True
                    domain.constraints[closest_idx, 2] = True
                    
                    num_applied_disps += 1

                    nodes_constrained.append(closest_idx)

    if num_applied_disps == len(boundary_disps.keys()):
        return num_applied_disps, np.array(nodes_constrained)
    else:
        raise RuntimeError("Wrong number of boundary displacements!")


num_applied_disps, nodes_constrained = apply_displacements_by_coordinates(
    domain=domain, data=matpatch, dim=dim)

#%% ---------------------------- Solve the system -----------------------------
increments = torch.tensor(matpatch['load_factor_time_series'])
# increments = torch.linspace(0.0, 1.0, 2) #
                    # torch.cat((torch.linspace(0.0, 1.0, 20),
                    #     torch.linspace(1.0, 0.0, 20)))
u_disp, f_int, sigma_out, def_grad, alpha_out = domain.solve(
    increments=increments, return_intermediate=True)

#stress shape: 
#    [time_increments, num_elem, num_stress, num_stress] mean over integ. points
#     num_stress = 3 for 3D
#%% ------------------------------- Save output -------------------------------
for idx_time in range(len(increments)):
    for idx_node in range(u_disp[-1,:,:].shape[0]):
        if idx_node in nodes_constrained:
            simulation_data['boundary_nodes_disps_time_series'][idx_node] = [
                u_disp[idx_time, idx_node,:]]
            simulation_data['boundary_nodes_forces_time_series'][idx_node] = [
                f_int[idx_time, idx_node,:]]
            # Print for user validation at last time step
            if idx_time == len(increments) - 1:
                print(f'Node: {idx_node}: u {u_disp[-1,idx_node,:]}')
                print(f'          force {f_int[-1, idx_node, :]}')


        # Extract and save: avg matpatch def_grad (volume average for the patch)
        # to avg matpatch sigma_out (volume average for the patch)

