"""User script: Run FEM simulation with material patch surrogate models."""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import pathlib
import pickle as pkl
import psutil
import time
import signal
import cProfile
import pstats
import scalene

# Add graphorge to sys.path
graphorge_path = str(pathlib.Path(__file__).parents[2] \
                     / "graphorge_material_patches" / "src")
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)

# Third-party
import torch
import matplotlib.pyplot as plt
import numpy as np

# Local
from torchfem import Solid, Planar
from torchfem.materials import (
    IsotropicElasticityPlaneStrain, 
    IsotropicPlasticityPlaneStrain,
    IsotropicElasticity3D,
    IsotropicPlasticity3D,
    Hyperelastic3D,
    IsotropicHenckyPlaneStrain
)
from torchfem.mesh import cube_hexa, rect_quad
from torchfem.elements import linear_to_quadratic

from utils import prescribe_disps_by_coords

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.titlesize": 16,
    "figure.dpi": 360,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.figsize": (6, 6),
    "lines.linewidth": 1.5
})
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
#

torch.set_default_dtype(torch.float64)

def run_simulation_surrogate(
    element_type='quad4',
    material_behavior='elastic',
    mesh_nx=1, mesh_ny=1, mesh_nz=1,
    model_path=None):
    """
    Run simulation using solve_matpatch function with Graphorge surrogate model
    """
    
    # Monitor memory and time
    process = psutil.Process(os.getpid())
    start_time = time.time()
    
    def print_status(location):
        current_time = time.time()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"[{location}] Time: {current_time - start_time:.2f}s, "
              f"Memory: {memory_mb:.1f}MB")
    
    print_status("START")
    
    # Set up signal handler to catch termination
    def signal_handler(signum, frame):
        print(f"\n[SIGNAL] Caught signal {signum}")
        print_status("SIGNAL_CAUGHT")
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    # Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)  
    
    # Determine element order and dimension
    if element_type in ['quad4', 'tri3', 'tetra4', 'hex8']:
        elem_order = 1
    elif element_type in ['quad8', 'tri6', 'tetra10', 'hex20']:
        elem_order = 2

    if element_type in ['quad4', 'tri3', 'quad8', 'tri6']:
        dim = 2
    elif element_type in ['tetra4', 'hex8', 'tetra10', 'hex20']:
        dim = 3
    
    # Default model path if not provided
    if model_path is None:
        if material_behavior == 'elastoplastic':
            model_path = (
                "/Users/rbarreira/Desktop/machine_learning/material_patches/"
                "graphorge_material_patches/src/graphorge/projects/"
                "material_patches/elastoplastic/2d/quad4/mesh1x1/ninc100/"
                "scaler_minmax/reference/3_model"
                # "/Users/rbarreira/Desktop/machine_learning/material_patches/"
                # "graphorge_material_patches/src/graphorge/projects/"
                # "material_patches/elastic/2d/quad4/mesh1x1/ninc1/"
                # "long_training_rnn/reference/3_model"
                # '/Users/rbarreira/Desktop/machine_learning/'
                #     'material_patches/graphorge_material_patches/src/'
                #     'graphorge/projects/material_patches/elastic/'
                #     '2d/quad4/mesh1x1/ninc100/27_force_equilibrium_rnn/'
                #     'reference/3_model'
            )
        elif material_behavior == 'elastic':
            model_path = (
                "/Users/rbarreira/Desktop/machine_learning/material_patches/"
                "graphorge_material_patches/src/graphorge/projects/"
                "material_patches/elastic/2d/quad4/mesh1x1/ninc1/"
                "long_training_rnn/reference/3_model"
                # '/Users/rbarreira/Desktop/machine_learning/'
                #     'material_patches/graphorge_material_patches/src/'
                #     'graphorge/projects/material_patches/elastic/'
                #     '2d/quad4/mesh1x1/ninc100/27_force_equilibrium_rnn/'
                #     'reference/3_model'
            )
    
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

        lmbda = e_young * nu / ((1. + nu) * (1. - 2. * nu))
        mu = e_young / (2. * (1. + nu))

        def psi(F):
            """
            Neo-Hookean strain energy density function.
            """
            # Compute the right Cauchy-Green deformation tensor
            C = F.transpose(-1, -2) @ F
            # Stable computation of the logarithm of the determinant
            logJ = 0.5 * torch.logdet(C)
            return (mu / 2 * (torch.trace(C) - 3.0) - mu * logJ +
                    lmbda / 2 * logJ**2)
        
        if dim == 2:
            material = IsotropicHenckyPlaneStrain(E=e_young, nu=nu)
        elif dim == 3:
            # material = IsotropicHencky3D(
            #     E=e_young, nu=nu, n_state=0)
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
                E=e_young, nu=nu, sigma_f=sigma_f, 
                sigma_f_prime=sigma_f_prime)
        elif dim == 3:
            material = IsotropicPlasticity3D(
                E=e_young, nu=nu, sigma_f=sigma_f, 
                sigma_f_prime=sigma_f_prime)

    #%% ----------------------- Geometry & mesh -------------------------------
    if dim == 2:
        # rect_quad takes number of nodes per direction
        nodes, elements = rect_quad(mesh_nx + 1, mesh_ny + 1)
        if elem_order == 2:
            nodes, elements = linear_to_quadratic(nodes, elements)

        domain = Planar(nodes, elements, material)
    elif dim == 3:
        # cube_hexa takes number of nodes per direction
        nodes, elements = cube_hexa(mesh_nx + 1, mesh_ny + 1, mesh_nz + 1)
        if elem_order == 2:
            nodes, elements = linear_to_quadratic(nodes, elements)

        domain = Solid(nodes, elements, material)
    
    # Define material patch flag - use surrogate for all elements
    num_elements = elements.shape[0]
    # Each element gets its own unique material patch ID (0, 1, 2, ...)
    is_mat_patch = torch.arange(num_elements, dtype=torch.int)
    
    #%% ----------------------- Boundary conditions ---------------------------
    nodes_constrained = []
    num_applied_disps = 0
    
    if dim == 2:
        # Simple tension test for 2D
        # Fix bottom edge (y=0), apply displacement to top edge (y=1)
        for i, node_coord in enumerate(nodes):
            # Fix bottom edge (y=0)
            if torch.abs(node_coord[1]) < 1e-6:
                domain.displacements[i, 0] = 0.0
                domain.displacements[i, 1] = 0.0
                domain.constraints[i, 0] = True
                domain.constraints[i, 1] = True
                nodes_constrained.append(i)
                num_applied_disps += 1
            
            # Apply displacement to top edge (y=1)
            elif torch.abs(node_coord[1] - 1.0) < 1e-6:
                domain.displacements[i, 0] = 0.0
                domain.displacements[i, 1] = 0.01
                domain.constraints[i, 0] = True
                domain.constraints[i, 1] = True
                nodes_constrained.append(i)
                num_applied_disps += 1
    elif dim == 3:
        # Simple tension test for 3D
        # Fix bottom face (z=0), apply displacement to top face (z=1)
        for i, node_coord in enumerate(nodes):
            # Fix bottom face (z=0)
            if torch.abs(node_coord[2]) < 1e-6:
                domain.displacements[i, 0] = 0.0
                domain.displacements[i, 1] = 0.0
                domain.displacements[i, 2] = 0.0
                domain.constraints[i, 0] = True
                domain.constraints[i, 1] = True
                domain.constraints[i, 2] = True
                nodes_constrained.append(i)
                num_applied_disps += 1
            
            # Apply displacement to top face (z=1)
            elif torch.abs(node_coord[2] - 1.0) < 1e-6:
                domain.displacements[i, 0] = 0.0
                domain.displacements[i, 1] = 0.0
                domain.displacements[i, 2] = 0.1
                domain.constraints[i, 0] = True
                domain.constraints[i, 1] = True
                domain.constraints[i, 2] = True
                nodes_constrained.append(i)
                num_applied_disps += 1
    
    # print(f"Applied boundary conditions to {num_applied_disps} nodes")
    #%% ------------------------------- Solver --------------------------------
    # Create more increments for elastoplastic simulation
    if material_behavior == 'elastoplastic':
        # Use 50 loading steps for elastoplastic simulation
        increments = torch.linspace(0.0, 1.0, 5)
        # RNN-like behavior
        is_stepwise = True 
    else:
        # One step for elastic sim
        # increments = torch.tensor([0.0, 1.0])
        # is_stepwise = False
        increments = torch.linspace(0.0, 1.0, 5)
        # RNN-like behavior
        is_stepwise = False 
    
    print_status("BEFORE_SOLVE")
    # Start profiling
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    # Profile solve method
    profiler_solve = cProfile.Profile()
    profiler_solve.enable()
    u_ref, f_ref, _, _, _ = domain.solve(increments=increments, rtol=1e-8)
    profiler_solve.disable()
    print("\n=== SOLVE METHOD PROFILE ===")
    stats_solve = pstats.Stats(profiler_solve)
    stats_solve.sort_stats('cumulative').print_stats(15)
    print_status("AFTER_SOLVE")

    print_status("BEFORE_SOLVE_MATPATCH")
    # Profile solve_matpatch method
    profiler_matpatch = cProfile.Profile()
    profiler_matpatch.enable()
    u, f, _, _, _ = domain.solve_matpatch(
        is_mat_patch=is_mat_patch,
        increments=increments,
        max_iter=100,
        rtol=1e-8,
        verbose=True,
        return_intermediate=True,
        return_volumes=False,
        is_stepwise=is_stepwise,
        model_directory=model_path
    )
    profiler_matpatch.disable()
    print("\n=== SOLVE_MATPATCH METHOD PROFILE ===")
    stats_matpatch = pstats.Stats(profiler_matpatch)
    stats_matpatch.sort_stats('cumulative').print_stats(15)
    # Stop profiling and print results
    # profiler.disable()
    print_status("AFTER_SOLVE_MATPATCH")
    
    # print("\n" + "="*60)
    # print("PROFILING RESULTS - TOP 15 BOTTLENECKS")
    # print("="*60)
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative').print_stats(15)

    #%% ------------------------------- Outputs -------------------------------
    # Create directory structure for outputs
    mesh_str = f"{mesh_nx}x{mesh_ny}"
    if dim == 3:
        mesh_str += f"x{mesh_nz}"
    
    # Determine number of increments for output directory
    # WHY: # Subtract 1 for initial condition
    n_increments = len(increments) - 1  
    
    output_dir = (
        f"results/{material_behavior}/{dim}d/{element_type}/"
        f"mesh_{mesh_str}/n_time_inc_{n_increments}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results = {
        'displacements': u.detach().cpu().numpy(),
        'forces': f.detach().cpu().numpy(),
        'model_path': model_path,
        'material_patch_ids': is_mat_patch.detach().cpu().numpy()}
    
    output_file = os.path.join(output_dir, "results.pkl")
    with open(output_file, 'wb') as file_handle:
        pkl.dump(results, file_handle)
    
    print(f"Results saved to {output_file}")
    

    # ----------------------- Plot boundary conditions ------------------------
    domain.plot()

    plot_path = os.path.join(output_dir, "reference_configuration.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ----------------------- Plot displacement field -------------------------
    # Get final displacement field
    # Shape: (n_nodes, n_dim)
    u_final = u[-1]  

    domain.plot(
        u=u_final,
        node_property=torch.norm(u_final, dim=1),
        title=r'Displacement Magnitude $||\mathbf{u}||_{2}$',
        colorbar=True,
        cmap='viridis', vmin=0.0, vmax=0.01
    )

    plot_path = os.path.join(output_dir, "displacement_field.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    

    u_diff = torch.abs(u[-1] - u_ref[-1]) 

    domain.plot(
        u=u_diff,
        node_property=torch.norm(u_diff, dim=1),
        title=r'$||\mathbf{u}_{pred} - \mathbf{u}_{ref}||_{2}$',
        colorbar=True,
        cmap='viridis', vmin=0.0, vmax=0.01
    )

    plot_path = os.path.join(output_dir, "displacement_field_difference.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------- Plot force field ------------------------------
    # Get final force field
    # Shape: (n_nodes, n_dim)
    f_final = f[-1]
    
    domain.plot(
        u=u_final,
        node_property=torch.norm(f_final, dim=1),
        title=r'Force Magnitude $||\mathbf{f}||_{2}$',
        colorbar=True,
        cmap='plasma'
    )
    
    force_plot_path = os.path.join(output_dir, "force_field.png")
    plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    f_diff = torch.abs(f[-1] - f_ref[-1]) 

    domain.plot(
        u=u_diff,
        node_property=torch.norm(f_diff, dim=1),
        title=r'$||\mathbf{f}_{pred} - \mathbf{f}_{ref}||_{2}$',
        colorbar=True,
        cmap='viridis'
    )

    plot_path = os.path.join(output_dir, "force_field_difference.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run_simulation_surrogate(
        element_type='quad4',
        material_behavior='elastoplastic',
        mesh_nx=1,
        mesh_ny=1,
        mesh_nz=1)