"""User script: Run FEM simulation with material patch surrogate models."""
#
#                                                                       Modules
# =============================================================================
import os
import sys
import pathlib
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add graphorge to sys.path
graphorge_path = str(
    pathlib.Path(__file__).parents[2] /
    "graphorge_material_patches" / "src")
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import torchfem modules
from torchfem import Solid, Planar
from torchfem.materials import (
    IsotropicElasticityPlaneStrain,
    IsotropicElasticityPlaneStress,
    IsotropicPlasticityPlaneStrain,
    IsotropicPlasticityPlaneStress,
    IsotropicElasticity3D,
    IsotropicPlasticity3D,
    IsotropicHencky3D,
    Hyperelastic3D,
    IsotropicHenckyPlaneStrain
)
from torchfem.mesh import cube_hexa, rect_quad
from torchfem.elements import linear_to_quadratic, Hexa1r, Quad1r
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import local utilities
from utils.boundary_conditons import (
    prescribe_disps_by_coords
)
from utils.mechanical_quantities import (
    compute_strain_energy_density
)
from utils.plotting import (
    plot_displacement_field,
    plot_domain_displacements,
    plot_shape_functions
)
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.set_default_dtype(torch.float64)
# =============================================================================
class Simulation:
    """Material patch finite element simulation.

    Manages FEM simulation workflow for material patches including mesh
    generation, material definition, boundary condition application,
    system solving, and result post-processing.
    """

    def __init__(
        self,
        element_type='quad4',
        material_behavior='elastic',
        patch_idx=0,
        num_increments=1,
        mesh_nx=3,
        mesh_ny=3,
        mesh_nz=3,
        is_red_int=False,
        is_save=False,
        filepath='/Users/rbarreira/Desktop/machine_learning/'
                 'material_patches/_data/'
    ):
        """Initialize simulation parameters.

        Args:
            element_type (str): Element type ('quad4', 'hex8', etc.)
            material_behavior (str): Material model ('elastic', etc.)
            patch_idx (int): Material patch index
            num_increments (int): Number of load increments
            mesh_nx (int): Elements in x-direction
            mesh_ny (int): Elements in y-direction
            mesh_nz (int): Elements in z-direction (3D only)
            is_red_int (bool): Use reduced integration for hex8
                elements (default: False)
            filepath (str): Path to material patch data
        """
        self.element_type = element_type
        self.material_behavior = material_behavior
        self.patch_idx = patch_idx
        self.num_increments = num_increments
        self.mesh_nx = mesh_nx
        self.mesh_ny = mesh_ny
        self.mesh_nz = mesh_nz
        self.is_red_int = is_red_int
        self.is_save = is_save
        self.filepath = filepath

        # Determine element order and dimension
        self._element_properties()

        # Setup file paths
        self._setup_paths()

        # Initialize data structures
        self.matpatch = None
        self.domain = None
        self.simulation_data = None
        self.nodes_constrained = None

    # -------------------------------------------------------------------------
    def _element_properties(self):
        """Determine element order and problem dimension."""
        if self.element_type in ['quad4', 'tri3', 'tetra4', 'hex8']:
            self.elem_order = 1
        elif self.element_type in ['quad8', 'tri6', 'tetra10', 'hex20']:
            self.elem_order = 2

        if self.element_type in ['quad4', 'tri3', 'quad8', 'tri6']:
            self.dim = 2
        elif self.element_type in ['tetra4', 'hex8', 'tetra10', 'hex20']:
            self.dim = 3

    # -------------------------------------------------------------------------
    def _setup_paths(self):
        """Setup input and output file paths."""
        out_filepath = (
            '/Users/rbarreira/Desktop/machine_learning/'
            'material_patches/')

        if self.dim == 2:
            self.dir_path = (
                f'{out_filepath}_data/{self.material_behavior}/'
                f'{self.dim}d/{self.element_type}/'
                f'mesh_{self.mesh_nx}x{self.mesh_ny}/'
                f'ninc{self.num_increments}/')
            self.input_filename = (
                f'{self.filepath}material_patches_generation_'
                f'{self.dim}d_{self.element_type}_mesh_'
                f'{self.mesh_nx}x{self.mesh_ny}/'
                f'material_patch_{self.patch_idx}/material_patch/'
                f'material_patch_attributes.pkl')
        elif self.dim == 3:
            if self.element_type not in [
                    'tetra4', 'hex8', 'tetra10', 'hex20']:
                raise ValueError(
                    f"Invalid element type for {self.dim}d problem!")
            self.dir_path = (
                f'{out_filepath}_data/{self.material_behavior}/'
                f'{self.dim}d/{self.element_type}/'
                f'mesh{self.mesh_nx}x{self.mesh_ny}x{self.mesh_nz}/'
                f'ninc{self.num_increments}/')
            self.input_filename = (
                f'{self.filepath}material_patches_generation_'
                f'{self.dim}d_{self.element_type}_'
                f'mesh{self.mesh_nx}x{self.mesh_ny}x{self.mesh_nz}/'
                f'material_patch_{self.patch_idx}/material_patch/'
                f'material_patch_attributes.pkl')

        self.output_filename = (
            f'{self.dir_path}matpatch_idx{self.patch_idx}.pkl')
        os.makedirs(self.dir_path, exist_ok=True)

    # -------------------------------------------------------------------------
    def load_material_patch(self):
        """Load material patch data from file."""
        with open(self.input_filename, 'rb') as file:
            self.matpatch = pkl.load(file)

    # -------------------------------------------------------------------------
    def create_material(self):
        """Create material model based on material_behavior.

        Returns:
            Material: Material model instance
        """
        if self.material_behavior == 'elastic':
            e_young = 110000
            nu = 0.33

            if self.dim == 2:
                return IsotropicElasticityPlaneStrain(E=e_young, nu=nu)
            elif self.dim == 3:
                return IsotropicElasticity3D(E=e_young, nu=nu)

        elif self.material_behavior == 'hyperelastic':
            e_young = 20000
            nu = 0.33

            lmbda = e_young * nu / ((1. + nu) * (1. - 2. * nu))
            mu = e_young / (2. * (1. + nu))

            def psi(F):
                """Neo-Hookean strain energy density function."""
                C = F.transpose(-1, -2) @ F
                logJ = 0.5 * torch.logdet(C)
                return (mu / 2 * (torch.trace(C) - 3.0) - mu * logJ +
                        lmbda / 2 * logJ**2)

            if self.dim == 2:
                return IsotropicHenckyPlaneStrain(E=e_young, nu=nu)
            elif self.dim == 3:
                return Hyperelastic3D(psi)

        elif self.material_behavior == 'elastoplastic_lh':
            e_young = 70000
            nu = 0.33

            sigma_y = 100.0
            hardening_modulus = 100.0

            def sigma_f(eps_pl):
                """Linear hardening function."""
                return sigma_y + hardening_modulus * eps_pl

            def sigma_f_prime(_eps_pl):
                """Derivative of linear hardening function."""
                return hardening_modulus

            if self.dim == 2:
                return IsotropicPlasticityPlaneStrain(
                    E=e_young, nu=nu, sigma_f=sigma_f,
                    sigma_f_prime=sigma_f_prime)
            elif self.dim == 3:
                return IsotropicPlasticity3D(
                    E=e_young, nu=nu, sigma_f=sigma_f,
                    sigma_f_prime=sigma_f_prime)

        elif self.material_behavior == 'elastoplastic_nlh':
            e_young = 70000
            nu = 0.33

            # Swift-Voce hardening parameters for AA2024
            a_s = 798.56  # MPa
            epsilon_0 = 0.0178
            n = 0.202
            k_0 = 363.84  # MPa
            q_v = 240.03  # MPa
            beta = 10.533
            omega = 0.368

            def k_s(eps_pl):
                """Swift hardening component."""
                return a_s * (epsilon_0 + eps_pl)**n

            def k_v(eps_pl):
                """Voce hardening component."""
                return k_0 + q_v * (1.0 - torch.exp(-beta * eps_pl))

            def sigma_f(eps_pl):
                """Combined Swift-Voce hardening function."""
                return omega * k_s(eps_pl) + (1.0 - omega) * k_v(eps_pl)

            def sigma_f_prime(eps_pl):
                """Derivative of Swift-Voce hardening function."""
                dks_deps_pl = a_s * n * (epsilon_0 + eps_pl)**(n - 1.0)
                dkv_deps_pl = q_v * beta * torch.exp(-beta * eps_pl)
                return omega * dks_deps_pl + (1.0 - omega) * dkv_deps_pl

            if self.dim == 2:
                return IsotropicPlasticityPlaneStrain(
                    E=e_young, nu=nu, sigma_f=sigma_f,
                    sigma_f_prime=sigma_f_prime)
            elif self.dim == 3:
                return IsotropicPlasticity3D(
                    E=e_young, nu=nu, sigma_f=sigma_f,
                    sigma_f_prime=sigma_f_prime)

        else:
            raise ValueError(
                f"Unknown material behavior: {self.material_behavior}")
    # -------------------------------------------------------------------------
    def create_mesh(self, material):
        """Create finite element mesh and domain.

        Args:
            material: Material model instance

        Returns:
            Domain: FEM domain (Planar or Solid)
        """
        if self.dim == 2:
            nodes, elements = rect_quad(
                self.mesh_nx + 1, self.mesh_ny + 1)
            if self.elem_order == 2:
                nodes, elements = linear_to_quadratic(nodes, elements)
            domain = Planar(nodes, elements, material)

            # Apply reduced integration if requested
            if self.is_red_int and self.element_type == 'quad4':
                domain.etype = Quad1r()
                domain.n_int = len(domain.etype.iweights())

            return domain
        elif self.dim == 3:
            nodes, elements = cube_hexa(
                self.mesh_nx + 1, self.mesh_ny + 1, self.mesh_nz + 1)
            if self.elem_order == 2:
                nodes, elements = linear_to_quadratic(nodes, elements)
            domain = Solid(nodes, elements, material)

            # Apply reduced integration if requested
            if self.is_red_int and self.element_type == 'hex8':
                domain.etype = Hexa1r()
                domain.n_int = len(domain.etype.iweights())

            return domain
    # -------------------------------------------------------------------------
    def initialize_data_structures(self):
        """Initialize simulation data storage structures."""
        self.simulation_data = {
            'bd_nodes_coords': {},
            'bd_nodes_disps_time_series': {},
            'bd_nodes_forces_time_series': {},
            'stress_avg': {},
            'strain_energy_density': {},
        }

        if self.material_behavior in ['elastoplastic_lh', 'elastoplastic_nlh']:
            self.simulation_data['epsilon_pl_eq'] = {}

        self._build_node_mapping()
        self._initialize_boundary_nodes()
        self._copy_data()
    # -------------------------------------------------------------------------
    def _build_node_mapping(self):
        """Build mapping from material patch node labels to mesh indices."""
        self.node_label_to_torchfem_idx = {}
        nodes = self.domain.nodes

        for node_label in self.matpatch['mesh_nodes_coords_ref'].keys():
            node_coords_matp = self.matpatch[
                'mesh_nodes_coords_ref'][node_label]
            ref_point = torch.tensor(node_coords_matp)
            distances = torch.sqrt(
                torch.sum((nodes - ref_point)**2, axis=1))
            closest_idx = torch.argmin(distances).item()

            if distances[closest_idx] >= 1e-6:
                break
            self.node_label_to_torchfem_idx[int(node_label)] = closest_idx
    # -------------------------------------------------------------------------
    def _initialize_boundary_nodes(self):
        """Initialize boundary node data structures."""
        for node_label in self.matpatch['mesh_boundary_nodes_disps'].keys():
            closest_idx = self.node_label_to_torchfem_idx[int(node_label)]
            self.simulation_data['bd_nodes_coords'][closest_idx] = (
                self.matpatch['mesh_nodes_coords_ref'][node_label])
            self.simulation_data['bd_nodes_disps_time_series'][
                closest_idx] = []
            self.simulation_data['bd_nodes_forces_time_series'][
                closest_idx] = []
    # -------------------------------------------------------------------------
    def _copy_data(self):
        """Copy material patch data to simulation_data."""
        excluded_fields = {
            'mesh_boundary_nodes_disps',
            'load_factor_time_series',
            'mesh_nodes_coords_ref',
            'mesh_boundary_nodes_disps_time'}

        for key, value in self.matpatch.items():
            if key not in excluded_fields:
                if key == 'mesh_nodes_matrix':
                    self.simulation_data['mesh_nodes_matrix'] = (
                        self._create_mesh_nodes_matrix())
                else:
                    self.simulation_data[key] = value
    # -------------------------------------------------------------------------
    def _create_mesh_nodes_matrix(self):
        """Create mesh nodes matrix based on mesh structure."""
        if self.dim == 2:
            """Create 2D mesh nodes matrix."""
            if self.elem_order == 1:
                mesh_nodes_matrix = np.zeros(
                    (self.mesh_nx + 1, self.mesh_ny + 1), dtype=int)
                node_idx = 0
                for i in range(self.mesh_nx + 1):
                    for j in range(self.mesh_ny + 1):
                        mesh_nodes_matrix[i, j] = node_idx
                        node_idx += 1
            else:  # elem_order == 2
                mesh_nodes_matrix = np.zeros(
                    (2*self.mesh_nx + 1, 2*self.mesh_ny + 1), dtype=int)
                node_idx = 0
                for i in range(2*self.mesh_nx + 1):
                    for j in range(2*self.mesh_ny + 1):
                        mesh_nodes_matrix[i, j] = node_idx
                        node_idx += 1

            return torch.tensor(mesh_nodes_matrix)
        elif self.dim == 3:
            """Create 3D mesh nodes matrix."""
            if self.elem_order == 1:
                mesh_nodes_matrix = np.zeros(
                    (self.mesh_nx + 1, self.mesh_ny + 1, self.mesh_nz + 1),
                    dtype=int)
                node_idx = 0
                for i in range(self.mesh_nx + 1):
                    for j in range(self.mesh_ny + 1):
                        for k in range(self.mesh_nz + 1):
                            mesh_nodes_matrix[i, j, k] = node_idx
                            node_idx += 1
            else:  # elem_order == 2
                mesh_nodes_matrix = np.zeros(
                    (2*self.mesh_nx + 1, 2*self.mesh_ny + 1,
                    2*self.mesh_nz + 1), dtype=int)
                node_idx = 0
                for i in range(2*self.mesh_nx + 1):
                    for j in range(2*self.mesh_ny + 1):
                        for k in range(2*self.mesh_nz + 1):
                            mesh_nodes_matrix[i, j, k] = node_idx
                            node_idx += 1

            return torch.tensor(mesh_nodes_matrix)
    # -------------------------------------------------------------------------
    def apply_boundary_conditions(self):
        """Apply boundary conditions from material patch data."""
        _, self.nodes_constrained = (
            prescribe_disps_by_coords(
                domain=self.domain, data=self.matpatch, dim=self.dim))

    # -------------------------------------------------------------------------
    def solve(self):
        """Solve the FEM system.

        Returns:
            tuple: (u_disp, f_int, sigma_out, def_grad, alpha_out,
                    vol_elem)
        """
        if self.num_increments == 1:
            increments = torch.linspace(0.0, 1.0, 2)
        elif self.num_increments > 1:
            increments = torch.tensor(
                self.matpatch['load_factor_time_series'])

        return self.domain.solve(
            increments=increments,
            return_intermediate=True,
            aggregate_integration_points=True,
            return_volumes=True)

    # -------------------------------------------------------------------------
    def postprocess_results(
            self, u_disp, f_int, sigma_out, def_grad,
            alpha_out, vol_elem):
        """Postprocess and store simulation results.

        Args:
            u_disp: Nodal displacements
            f_int: Internal forces
            sigma_out: Stress tensor
            def_grad: Deformation gradient
            alpha_out: Internal state variables
            vol_elem: Element volumes
        """
        # Compute volume-weighted quantities
        vol_weights = self._compute_volume_weights(vol_elem)
        sigma_out_avg = self._compute_stress_avg(
            sigma_out, vol_weights)
        alpha_out_avg = self._compute_plastic_strain_avg(
            alpha_out, vol_weights)

        # Compute strain energy
        strain_energy_density = compute_strain_energy_density(
            sigma_out, def_grad, self.material_behavior, self.dim)
        total_strain_energy = (
            strain_energy_density * vol_weights).sum(dim=1)

        # Store results
        self._store_results(
            u_disp, f_int, sigma_out_avg, alpha_out_avg,
            total_strain_energy)

    # -------------------------------------------------------------------------
    def _compute_volume_weights(self, vol_elem):
        """Compute volume-weighted normalization factors."""
        total_volume = vol_elem.sum(dim=1, keepdim=True)
        return vol_elem / total_volume

    # -------------------------------------------------------------------------
    def _compute_stress_avg(self, sigma_out, vol_weights):
        """Compute volume-weighted stress average."""
        vol_weights_expanded = vol_weights.unsqueeze(-1).unsqueeze(-1)

        if self.num_increments == 1:
            return sigma_out[-1, :, :] * vol_weights_expanded
        else:
            return sigma_out * vol_weights_expanded

    # -------------------------------------------------------------------------
    def _compute_plastic_strain_avg(self, alpha_out, vol_weights):
        """Compute volume-weighted plastic strain average."""
        if self.material_behavior in ['elastoplastic_lh', 'elastoplastic_nlh']:
            return (alpha_out[:, :, 0] * vol_weights).sum(dim=1)
        return None

    # -------------------------------------------------------------------------
    def _store_results(
            self, u_disp, f_int, sigma_out_avg, alpha_out_avg,
            total_strain_energy):
        """Store all simulation results.

        Args:
            u_disp: Nodal displacements
            f_int: Internal forces
            sigma_out_avg: Volume-weighted stress average
            alpha_out_avg: Volume-weighted plastic strain average
            total_strain_energy: Total strain energy
        """
        # Store boundary node results
        for idx_node in range(u_disp.shape[1]):
            if idx_node in self.nodes_constrained:
                if self.num_increments == 1:
                    self.simulation_data['bd_nodes_disps_time_series'][
                        idx_node] = u_disp[-1, idx_node, :]
                    self.simulation_data['bd_nodes_forces_time_series'][
                        idx_node] = f_int[-1, idx_node, :]
                else:
                    self.simulation_data['bd_nodes_disps_time_series'][
                        idx_node] = u_disp[:, idx_node, :]
                    self.simulation_data['bd_nodes_forces_time_series'][
                        idx_node] = f_int[:, idx_node, :]

        # Store averaged results
        if self.num_increments == 1:
            self.simulation_data['stress_avg'] = sigma_out_avg[-1, 0]
            self.simulation_data['strain_energy_density'] = (
                total_strain_energy[-1])
            if self.material_behavior in [
                    'elastoplastic_lh', 'elastoplastic_nlh']:
                self.simulation_data['epsilon_pl_eq'] = alpha_out_avg[-1]
        else:
            for idx_time in range(self.num_increments + 1):
                self.simulation_data['strain_energy_density'][
                    idx_time] = total_strain_energy[idx_time]
                if self.material_behavior in [
                        'elastoplastic_lh', 'elastoplastic_nlh']:
                    self.simulation_data['epsilon_pl_eq'][idx_time] = (
                        alpha_out_avg[idx_time])

    # -------------------------------------------------------------------------
    def save_results(self):
        """Save simulation results to pickle file."""
        try:
            with open(self.output_filename, 'wb') as f:
                pkl.dump(
                    self.simulation_data, f,
                    protocol=pkl.HIGHEST_PROTOCOL)
        except Exception as excp:
            print(f"Error saving simulation data: {excp}")

    # -------------------------------------------------------------------------
    def run(self):
        """Execute complete simulation workflow."""
        self.load_material_patch()
        material = self.create_material()
        self.domain = self.create_mesh(material)
        self.initialize_data_structures()
        self.apply_boundary_conditions()

        results = self.solve()
        self.postprocess_results(*results)

        # Save results
        if self.is_save:
            self.save_results()
# =============================================================================
def run_simulation(
    element_type='quad4',
    material_behavior='elastic',
    patch_idx=0,
    num_increments=1,
    mesh_nx=3,
    mesh_ny=3,
    mesh_nz=3,
    is_red_int=False,
    is_save=False,
    filepath='/Users/rbarreira/Desktop/machine_learning/'
             'material_patches/_data/'
):
    """Run finite element simulation for material patch analysis.

    Convenience function wrapping Simulation class for backward
    compatibility.

    Args:
        element_type (str): Finite element type
        material_behavior (str): Material constitutive model
        patch_idx (int): Material patch identifier index
        num_increments (int): Number of load increments
        mesh_nx (int): Number of elements in x-direction
        mesh_ny (int): Number of elements in y-direction
        mesh_nz (int): Number of elements in z-direction
        is_red_int (bool): Use reduced integration for hex8 elements
        is_save (bool): Save simulation output
        filepath (str): Base path to material patch input data
    """
    sim = Simulation(
        element_type=element_type,
        material_behavior=material_behavior,
        patch_idx=patch_idx,
        num_increments=num_increments,
        mesh_nx=mesh_nx,
        mesh_ny=mesh_ny,
        mesh_nz=mesh_nz,
        is_red_int=is_red_int,
        is_save=is_save,
        filepath=filepath
    )
    sim.run()
# =============================================================================
if __name__ == '__main__':
    # filepath = (
    #     '/Users/rbarreira/Desktop/machine_learning/material_patches/'
    #     '_input_material_patches/')
    filepath = ('/Volumes/T7/material_patches_data/')

    for idx in range(0, 10000):
        if idx % 100 == 0:
            print(f'     {idx}')

        run_simulation(
            element_type='quad4',
            material_behavior='elastoplastic_nlh',
            num_increments=100,
            patch_idx=idx,
            filepath=filepath,
            mesh_nx=1,
            mesh_ny=1,
            mesh_nz=1,
            is_save=True,
            is_red_int=True)
