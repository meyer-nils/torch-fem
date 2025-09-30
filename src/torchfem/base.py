from abc import ABC, abstractmethod
from typing import Literal, Tuple
import numpy as np
import torch
from torch import Tensor
import copy
import cProfile
import pstats

from .elements import Element, Quad1, Quad2, Hexa1, Hexa2
from .materials import Material
from .sparse import CachedSolve, sparse_solve

import torch_geometric.data as pyg_data
from graphorge.gnn_base_model.model.gnn_model import GNNEPDBaseModel
from graphorge.gnn_base_model.data.graph_data import GraphData
from graphorge.projects.material_patches.gnn_model_tools.gen_graphs_files \
    import (get_elem_size_dims, get_mesh_connected_nodes)
from graphorge.projects.material_patches.gnn_model_tools.features import (
    GNNPatchFeaturesGenerator)
from graphorge.gnn_base_model.model.custom_layers import (
    compute_stiffness_matrix, extract_forces,
    extract_displacements, compute_edge_features,
    reconstruct_graph_with_displacements)
import torch.func as torch_func
import functools


class FEM(ABC):
    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize a general finite element problem.

        Args:
            nodes (Tensor): Nodal coordinates of shape (n_nodes, n_dim).
            elements (Tensor): Element connectivity of shape
                (n_elements, n_nodes_per_element).
            material (Material): Material model instance.

        Note:
            Automatically vectorizes non-vectorized materials for efficient
            computation across all elements.
        """
        # Store nodes and elements
        self.nodes = nodes
        self.elements = elements
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute problem size
        self.n_dofs = torch.numel(self.nodes)
        self.n_nod = nodes.shape[0]
        self.n_dim = nodes.shape[1]
        self.n_elem = len(self.elements)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize load variables
        self._forces = torch.zeros_like(nodes)
        self._displacements = torch.zeros_like(nodes)
        self._constraints = torch.zeros_like(nodes, dtype=torch.bool)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute mapping from local to global indices
        idx = (self.n_dim * self.elements).unsqueeze(-1) + \
            torch.arange(self.n_dim)
        self.idx = idx.reshape(self.n_elem, -1).to(torch.int32)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Vectorize material
        if material.is_vectorized:
            self.material = material
        else:
            self.material = material.vectorize(self.n_elem)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize types
        self.n_stress: int
        self.n_int: int
        self.ext_strain: Tensor
        self.etype: Element
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Cached solve for sparse linear systems
        self.cached_solve = CachedSolve()
    # -------------------------------------------------------------------------
    @property
    def forces(self) -> Tensor:
        """Get nodal forces.

        Returns:
            Tensor: Nodal forces of shape (n_nodes, n_dim).
        """
        return self._forces
    # -------------------------------------------------------------------------
    @forces.setter
    def forces(self, value: Tensor):
        """Set nodal forces with validation.

        Args:
            value (Tensor): Nodal forces tensor of shape (n_nodes, n_dim).
                Must be floating-point type.

        Raises:
            ValueError: If shape doesn't match nodes.
            TypeError: If not floating-point type.
        """
        if not value.shape == self.nodes.shape:
            raise ValueError("Forces must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Forces must be a floating-point tensor.")
        self._forces = value.to(self.nodes.device)
    # -------------------------------------------------------------------------
    @property
    def displacements(self) -> Tensor:
        """Get nodal displacements.

        Returns:
            Tensor: Nodal displacements of shape (n_nodes, n_dim).
        """
        return self._displacements
    # -------------------------------------------------------------------------
    @displacements.setter
    def displacements(self, value: Tensor):
        """Set nodal displacements with validation.

        Args:
            value (Tensor): Nodal displacements tensor of shape
                (n_nodes, n_dim). Must be floating-point type.

        Raises:
            ValueError: If shape doesn't match nodes.
            TypeError: If not floating-point type.
        """
        if not value.shape == self.nodes.shape:
            raise ValueError("Displacements must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Displacements must be a floating-point tensor.")
        self._displacements = value.to(self.nodes.device)
    # -------------------------------------------------------------------------
    @property
    def constraints(self) -> Tensor:
        """Get boundary condition constraints.

        Returns:
            Tensor: Boolean tensor of shape (n_nodes, n_dim) where True
                indicates constrained degrees of freedom.
        """
        return self._constraints
    # -------------------------------------------------------------------------
    @constraints.setter
    def constraints(self, value: Tensor):
        """Set boundary condition constraints with validation.

        Args:
            value (Tensor): Boolean tensor of shape (n_nodes, n_dim) where
                True indicates constrained degrees of freedom.

        Raises:
            ValueError: If shape doesn't match nodes.
            TypeError: If not boolean type.
        """
        if not value.shape == self.nodes.shape:
            raise ValueError("Constraints must have the same shape as nodes.")
        if value.dtype != torch.bool:
            raise TypeError("Constraints must be a boolean tensor.")
        self._constraints = value.to(self.nodes.device)
    # -------------------------------------------------------------------------
    @abstractmethod
    def eval_shape_functions(
        self, xi: Tensor, u: Tensor | float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate element shape functions and derivatives.

        Args:
            xi (Tensor): Natural coordinates for evaluation points.
            u (Tensor | float): Displacement field. Defaults to 0.0.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Shape functions, derivatives,
                and Jacobian determinant.
        """
        raise NotImplementedError
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        """Compute element stiffness matrix.

        Args:
            detJ (Tensor): Jacobian determinant at integration points.
            BCB (Tensor): B^T * C * B matrix product.

        Returns:
            Tensor: Element stiffness matrix.
        """
        raise NotImplementedError
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        """Compute element internal force vector.

        Args:
            detJ (Tensor): Jacobian determinant at integration points.
            B (Tensor): Strain-displacement matrix.
            S (Tensor): Stress tensor.
        """
        raise NotImplementedError
    # -------------------------------------------------------------------------
    @abstractmethod
    def plot(self, u: float | Tensor = 0.0, **kwargs):
        """Plot finite element mesh and results.

        Args:
            u (float | Tensor): Displacement field for deformed
                configuration. Defaults to 0.0.
            **kwargs: Additional plotting parameters.
        """
        raise NotImplementedError
    # -------------------------------------------------------------------------
    def compute_B(self) -> Tensor:
        """
        Compute null space matrix representing rigid body modes.
        """
        if self.n_dim == 3:
            B = torch.zeros((self.n_dofs, 6))
            B[0::3, 0] = 1
            B[1::3, 1] = 1
            B[2::3, 2] = 1
            B[1::3, 3] = -self.nodes[:, 2]
            B[2::3, 3] = self.nodes[:, 1]
            B[0::3, 4] = self.nodes[:, 2]
            B[2::3, 4] = -self.nodes[:, 0]
            B[0::3, 5] = -self.nodes[:, 1]
            B[1::3, 5] = self.nodes[:, 0]
        else:
            B = torch.zeros((self.n_dofs, 3))
            B[0::2, 0] = 1
            B[1::2, 1] = 1
            B[1::2, 2] = -self.nodes[:, 0]
            B[0::2, 2] = self.nodes[:, 1]
        return B
    # -------------------------------------------------------------------------
    def k0(self) -> Tensor:
        """Compute element stiffness matrix for zero strain.
        
        Returns:
            Tensor: Element stiffness matrices of shape (n_elem, n_dof_elem, 
                n_dof_elem).
        """
        u = torch.zeros_like(self.nodes)
        F = torch.zeros(2, self.n_int, self.n_elem, self.n_stress, 
                        self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        F[:, :, :, :, :] = torch.eye(self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        s = torch.zeros(2, self.n_int, self.n_elem, self.n_stress, 
                        self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        a = torch.zeros(2, self.n_int, self.n_elem, self.material.n_state)
        du = torch.zeros_like(self.nodes)
        de0 = torch.zeros(self.n_elem, self.n_stress, self.n_stress)
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        k, _ = self.integrate_material(u, F, s, a, 1, du, de0, False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return k
    # -------------------------------------------------------------------------
    def integrate_material(
        self,
        u: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        n: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix.
        
        Args:
            u (Tensor): Displacement history of shape (n_increments, n_nodes, 
                n_dim).
            F (Tensor): Deformation gradient history of shape (n_increments, 
                n_int, n_elem, n_stress, n_stress).
            stress (Tensor): Stress history of shape (n_increments, n_int, 
                n_elem, n_stress, n_stress).
            state (Tensor): Material state variable history of shape 
                (n_increments, n_int, n_elem, n_state).
            n (int): Current increment number.
            du (Tensor): Displacement increment vector of shape (n_dofs,).
            de0 (Tensor): External strain increment of shape (n_elem, 
                n_stress, n_stress).
            nlgeom (bool): Whether to use nonlinear geometry.
            
        Returns:
            Tuple[Tensor, Tensor]: Element stiffness matrices of shape 
                (n_elem, n_dof_elem, n_dof_elem) and internal force vectors
                of shape (n_elem, n_dof_elem).
        """
        # Compute updated configuration
        u_trial = u[n - 1] + du.view((-1, self.n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape displacement increment
        du = du.view(-1, self.n_dim)[self.elements].reshape(
            self.n_elem, -1, self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize nodal force and stiffness
        n_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dim * n_nod)
        k = torch.zeros((self.n_elem, self.n_dim * n_nod, self.n_dim * n_nod))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i, (w, xi) in enumerate(zip(self.etype.iweights(), 
                                        self.etype.ipoints())):
            # Compute gradient operators
            _, B0, detJ0 = self.eval_shape_functions(xi)
            if nlgeom:
                # Compute updated gradient operators in deformed configuration
                _, B, detJ = self.eval_shape_functions(xi, u_trial)
            else:
                # Use initial gradient operators
                B = B0
                detJ = detJ0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute displacement gradient increment
            H_inc = torch.einsum("...ij,...jk->...ik", B0, du)
            # Update deformation gradient
            F[n, i] = F[n - 1, i] + H_inc
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Evaluate material response
            stress[n, i], state[n, i], ddsdde = self.material.step(
                H_inc, F[n - 1, i], stress[n - 1, i], state[n - 1, i], de0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element internal forces
            force_contrib = self.compute_f(detJ, B, stress[n, i].clone())
            f += w * force_contrib.reshape(-1, self.n_dim * n_nod)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                # Material stiffness
                BCB = torch.einsum(
                    "...ijpq,...qk,...il->...ljkp", ddsdde, B, B)
                BCB = BCB.reshape(-1, self.n_dim * n_nod, self.n_dim * n_nod)
                k += w * self.compute_k(detJ, BCB)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if nlgeom:
                # Geometric stiffness
                BSB = torch.einsum(
                    "...iq,...qk,...il->...lk", stress[n, i].clone(), B, B
                )
                zeros = torch.zeros_like(BSB)
                kg = torch.stack([BSB] + (self.n_dim - 1) * [zeros], dim=-1)
                kg = kg.reshape(-1, n_nod, self.n_dim * n_nod).unsqueeze(-2)
                zeros = torch.zeros_like(kg)
                kg = torch.stack([kg] + (self.n_dim - 1) * [zeros], dim=-2)
                kg = kg.reshape(-1, self.n_dim * n_nod, self.n_dim * n_nod)
                k += w * self.compute_k(detJ, kg)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return k, f
    # ------------------------------------------------------------------------- 
    def _load_Graphorge_model(self, model_directory: str, 
                              device_type: str = 'cpu'):
        """Load and configure Graphorge material patch model.
        
        Args:
            model_directory (str): Path to the directory containing the trained 
                Graphorge model files.
            device_type (str, optional): Device type for model execution 
                ('cpu' or 'cuda'). Defaults to 'cpu'.
                
        Returns:
            GNNEPDBaseModel: Loaded and configured Graphorge model ready for 
                inference with material patch predictions.
                
        Example:
            >>> model = fem_instance._load_Graphorge_model(
            ...     model_directory='/path/to/trained/model',
            ...     device_type='cpu')
        """     
        # Initialize model from directory
        model = GNNEPDBaseModel.init_model_from_file(model_directory)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device
        model.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load best model state
        _ = model.load_model_state(
            load_model_state='best', is_remove_posterior=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set to evaluation mode
        model.eval()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return model
    # -------------------------------------------------------------------------  
    def integrate_field(self, field: Tensor | None = None) -> Tensor:
        """Integrate scalar field over elements.
        
        Args:
            field (Tensor, optional): Scalar field values at nodes of shape 
                (n_nodes,). If None, integrates unity to compute volumes. 
                Defaults to None.
                
        Returns:
            Tensor: Integrated values for each element of shape (n_elem,).
                If field is None, returns element volumes.
        """
        # Default field is ones to integrate volume
        if field is None:
            field = torch.ones(self.n_nod)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Integrate
        res = torch.zeros(len(self.elements))
        for w, xi in zip(self.etype.iweights(), self.etype.ipoints()):
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Evalute shale functions
            N, B, detJ = self.eval_shape_functions(xi)
            f = field[self.elements, None].squeeze() @ N
            res += w * f * detJ
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return res
    # -------------------------------------------------------------------------
    def assemble_stiffness(self, k: Tensor, con: Tensor) -> Tensor:
        """Assemble global stiffness matrix from element contributions.
        Args:
            k (Tensor): Element stiffness matrices of shape 
                (n_elem, n_dof_elem, n_dof_elem).
            con (Tensor): Global DOF indices of constrained degrees of freedom
                of shape (n_constraints,).
                
        Returns:
            Tensor: Assembled sparse global stiffness matrix of shape 
                (n_dofs, n_dofs) in COO format.
        """
        # Initialize sparse matrix
        size = (self.n_dofs, self.n_dofs)
        K = torch.empty(size, layout=torch.sparse_coo)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build matrix in chunks to prevent excessive memory usage
        chunks = 4
        for idx, k_chunk in zip(torch.chunk(self.idx, chunks), 
                                torch.chunk(k, chunks)):
            # Ravel indices and values
            chunk_size = idx.shape[0]
            col = idx.unsqueeze(1).expand(chunk_size, self.idx.shape[1], 
                                          -1).ravel()
            row = idx.unsqueeze(-1).expand(chunk_size, -1, 
                                           self.idx.shape[1]).ravel()
            indices = torch.stack([row, col], dim=0)
            values = k_chunk.ravel()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Eliminate and replace constrained dofs
            ci = torch.isin(idx, con)
            mask_col = ci.unsqueeze(1).expand(chunk_size, self.idx.shape[1], 
                                              -1).ravel()
            mask_row = (
                ci.unsqueeze(-1).expand(chunk_size, -1, 
                                        self.idx.shape[1]).ravel()
            )
            mask = ~(mask_col | mask_row)
            diag_index = torch.stack((con, con), dim=0)
            diag_value = torch.ones_like(con, dtype=k.dtype)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Concatenate
            indices = torch.cat((indices[:, mask], diag_index), dim=1)
            values = torch.cat((values[mask], diag_value), dim=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble stiffness matrix as a sparse coo tensor
            K += torch.sparse_coo_tensor(indices, values, size=size).coalesce()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return K.coalesce()
    # -------------------------------------------------------------------------
    def assemble_force(self, f: Tensor) -> Tensor:
        """Assemble global force vector from element contributions.
        
        Args:
            f (Tensor): Element force vectors of shape (n_elem, n_dof_elem).
            
        Returns:
            Tensor: Assembled global force vector of shape (n_dofs,).
        """

        # Initialize force vector
        F = torch.zeros((self.n_dofs))

        # Ravel indices and values
        indices = self.idx.ravel()
        values = f.ravel()

        return F.index_add_(0, indices, values)
    # -------------------------------------------------------------------------
    def solve(
        self,
        increments: Tensor = torch.tensor([0.0, 1.0]),
        max_iter: int = 100,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        verbose: bool = False,
        method: Literal["spsolve", "minres", "cg", "pardiso"] | None = None,
        device: str | None = None,
        return_intermediate: bool = True,
        aggregate_integration_points: bool = True,
        use_cached_solve: bool = False,
        nlgeom: bool = False,
        return_volumes: bool = False,
        return_resnorm: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Solve the FEM problem with the Newton-Raphson method.

        Args:
            increments (Tensor): Load increment stepping.
            max_iter (int): Maximum number of iterations during Newton-Raphson.
            rtol (float): Relative tolerance for Newton-Raphson convergence.
            atol (float): Absolute tolerance for Newton-Raphson convergence.
            stol (float): Solver tolerance for iterative methods.
            verbose (bool): Print iteration information.
            method (str): Method for linear solve 
                ('spsolve','minres','cg','pardiso').
            device (str): Device to run the linear solve on.
            return_intermediate (bool): Return intermediate values if True.
            aggregate_integration_points (bool): Aggregate integration 
                points if True.
            use_cached_solve (bool): Use cached solve, e.g. in topology 
                optimization.
            nlgeom (bool): Use nonlinear geometry if True.
            return_volumes (bool): Return element volumes for each 
                increment if True.
            return_resnorm (bool): Return residual norm history for each 
                increment if True.

        Returns:
                Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Final 
                    displacements,
                    internal forces, stress, deformation gradient, and 
                    material state.
                If return_volumes=True, also returns element volumes with shape
                (num_increments, num_elem, 1).
                If return_resnorm=True, also returns residual history dict with
                increment numbers as keys and lists of residual norms as values.

        """
        # Number of increments
        N = len(increments)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Null space rigid body modes for AMG preconditioner
        B = self.compute_B()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize variables to be computed
        u = torch.zeros(N, self.n_nod, self.n_dim)
        f = torch.zeros(N, self.n_nod, self.n_dim)
        stress = torch.zeros(N, self.n_int, self.n_elem,
                             self.n_stress, self.n_stress)
        defgrad = torch.zeros(N, self.n_int, self.n_elem,
                              self.n_stress, self.n_stress)
        defgrad[:, :, :, :, :] = torch.eye(self.n_stress)
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize volumes if requested
        if return_volumes:
            volumes = torch.zeros(N, self.n_elem)
            # Compute initial volume for increment 0
            # Default field is ones to integrate volume
            volumes[0] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize residual norm history if requested
        if return_resnorm:
            residual_history = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize global stiffness matrix
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Incremental loading
        for n in range(1, N):
            # Increment size
            inc = increments[n] - increments[n - 1]
            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            de0 = inc * self.ext_strain
            # Newton-Raphson iterations
            # Initialize residual list for this increment if requested
            if return_resnorm:
                residual_history[n] = []
            
            for i in range(max_iter):
                du[con] = DU[con]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Element-wise integration
                # profiler = cProfile.Profile()
                # profiler.enable()
                k, f_i = self.integrate_material(
                    u, defgrad, stress, state, n, du, de0, nlgeom
                )
                # profiler.disable()
                # print(f"\n=== integrate_material PROFILE (increment {n}) ===")
                # stats = pstats.Stats(profiler)
                # stats.sort_stats('cumulative').print_stats(10)
                if self.K.numel() == 0 or not self.material.n_state == 0 or \
                    nlgeom:
                    self.K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_i)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save residual norm to history if requested
                if return_resnorm:
                    residual_history[n].append(res_norm.item())
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save initial residual
                if i == 0:
                    res_norm0 = res_norm
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Print iteration information
                if verbose:
                    print(f"Increment {n} | Iteration {i+1} | "
                          f"Residual: {res_norm:.5e}")
                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol:
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Use cached solve from previous iteration if available
                if i == 0 and use_cached_solve:
                    cached_solve = self.cached_solve
                else:
                    cached_solve = CachedSolve()
                # Only update cache on first iteration
                update_cache = i == 0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve for displacement increment
                du -= sparse_solve(
                    self.K,
                    residual,
                    B,
                    stol,
                    device,
                    method,
                    None,
                    cached_solve,
                    update_cache,
                )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check convergence
            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise Exception("Newton-Raphson iteration did not converge.")
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update increment
            f[n] = F_int.reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element volumes if requested
            if return_volumes:
                volumes[n] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Aggregate integration points as mean
        if aggregate_integration_points:
            defgrad = defgrad.mean(dim=1)
            stress = stress.mean(dim=1)
            state = state.mean(dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Squeeze outputs
        stress = stress.squeeze()
        defgrad = defgrad.squeeze()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        result = [u, f, stress, defgrad, state]
        # Return intermediate states
        if not return_intermediate:
            result = [x[-1] for x in result]
        # Return volumes
        if return_volumes:
            result.append(volumes if return_intermediate else volumes[-1])
        # Return residual norm
        if return_resnorm:
            result.append(residual_history)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(result)
    # -------------------------------------------------------------------------
    def integrate_material_msc(
        self,
        msc_model,
        u: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        n: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool,
        scaler_hydrostatic: float,
        scaler_deviatoric: float,
    ) -> Tuple[Tensor, Tensor]:
        """Perform numerical integrations using MSC surrogate model.
        
        Args:
            msc_model: Loaded MSC PyTorch model.
            u (Tensor): Displacement history of shape (n_increments, n_nodes, 
                n_dim).
            F (Tensor): Deformation gradient history of shape (n_increments, 
                n_int, n_elem, n_stress, n_stress).
            stress (Tensor): Stress history of shape (n_increments, n_int, 
                n_elem, n_stress, n_stress).
            state (Tensor): Material state variable history of shape 
                (n_increments, n_int, n_elem, n_state).
            n (int): Current increment number.
            du (Tensor): Displacement increment vector of shape (n_dofs,).
            de0 (Tensor): External strain increment of shape (n_elem, 
                n_stress, n_stress).
            nlgeom (bool): Whether to use nonlinear geometry.
            scaler_hydrostatic (float): Scaling factor for hydrostatic stress.
            scaler_deviatoric (float): Scaling factor for deviatoric stress.
            
        Returns:
            Tuple[Tensor, Tensor]: Element stiffness matrices of shape 
                (n_elem, n_dof_elem, n_dof_elem) and internal force vectors
                of shape (n_elem, n_dof_elem).
        """
        
        # Compute updated configuration
        u_trial = u[n - 1] + du.view((-1, self.n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape displacement increment
        du = du.view(-1, self.n_dim)[self.elements].reshape(
            self.n_elem, -1, self.n_stress)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize nodal force and stiffness
        n_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dim * n_nod)
        k = torch.zeros((self.n_elem, self.n_dim * n_nod, self.n_dim * n_nod))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i, (w, xi) in enumerate(zip(self.etype.iweights(), 
                                        self.etype.ipoints())):
            # Compute gradient operators
            _, B0, detJ0 = self.eval_shape_functions(xi)
            if nlgeom:
                # Compute updated gradient operators in deformed configuration
                _, B, detJ = self.eval_shape_functions(xi, u_trial)
            else:
                # Use initial gradient operators
                B = B0
                detJ = detJ0
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute displacement gradient increment
            H_inc = torch.einsum("...ij,...jk->...ik", B0, du)
            # Update deformation gradient
            F[n, i] = F[n - 1, i] + H_inc
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute strain tensor from displacement gradient
            # For small strain: epsilon = 0.5*(H + H^T)
            eps = 0.5 * (H_inc + H_inc.transpose(-1, -2))
            
            # Prepare strains in torch-fem order for gradient computation
            eps_torchfem = torch.zeros(self.n_elem, self.n_stress)
            eps_torchfem[:, 0] = eps[:, 0, 0]  # eps_11
            eps_torchfem[:, 1] = eps[:, 0, 1]  # eps_12
            if self.n_dim == 3:
                eps_torchfem[:, 2] = eps[:, 0, 2]  # eps_13
                eps_torchfem[:, 3] = eps[:, 1, 1]  # eps_22
                eps_torchfem[:, 4] = eps[:, 1, 2]  # eps_23
                eps_torchfem[:, 5] = eps[:, 2, 2]  # eps_33
            else:
                eps_torchfem[:, 2] = eps[:, 1, 1]  # eps_22
            
            # Enable gradients for torch-fem ordered strains
            eps_torchfem_grad = eps_torchfem.clone().requires_grad_(True)
            
            # Convert to MSC order for model input
            eps_msc_grad = torch.zeros(self.n_elem, 6)
            eps_msc_grad[:, 0] = eps_torchfem_grad[:, 0]  # eps_11
            eps_msc_grad[:, 1] = eps_torchfem_grad[:, 1]  # eps_12
            if self.n_dim == 3:
                eps_msc_grad[:, 2] = eps_torchfem_grad[:, 2]  # eps_13
                eps_msc_grad[:, 3] = eps_torchfem_grad[:, 3]  # eps_22
                eps_msc_grad[:, 4] = eps_torchfem_grad[:, 4]  # eps_23
                eps_msc_grad[:, 5] = eps_torchfem_grad[:, 5]  # eps_33
            else:
                eps_msc_grad[:, 2] = 0.0
                eps_msc_grad[:, 3] = eps_torchfem_grad[:, 2]  # eps_22
                eps_msc_grad[:, 4] = 0.0
                eps_msc_grad[:, 5] = 0.0
            
            # Reshape for MSC input (seq_len=1, features=6)
            eps_msc_input = eps_msc_grad.unsqueeze(1)
            
            # Extract hidden states from previous step
            # state[n-1, i] stores states per integration point
            hidden_states = state[n - 1, i]
            
            # Single forward pass with gradients enabled
            sigma_pred, new_hidden_states = msc_model(
                eps_msc_input, hidden_states)
            
            # Store new hidden states
            # state[n, i] stores states per integration point
            state[n, i] = new_hidden_states
            
            # Apply denormalization to get stress in MSC order
            sigma_denorm = sigma_pred.clone()
            sigma_denorm[:, :, 0] *= scaler_hydrostatic
            sigma_denorm[:, :, 1:] *= scaler_deviatoric
            
            # Reorder denormalized stress to torch-fem order
            sigma_torchfem = torch.zeros(self.n_elem, 1, self.n_stress)
            sigma_torchfem[:, :, 0] = (sigma_denorm[:, :, 1] + 
                                     sigma_denorm[:, :, 0])  # s11
            sigma_torchfem[:, :, 1] = sigma_denorm[:, :, 3]  # s12
            if self.n_dim == 3:
                sigma_torchfem[:, :, 2] = sigma_denorm[:, :, 4]  # s13
                sigma_torchfem[:, :, 3] = (sigma_denorm[:, :, 2] + 
                                         sigma_denorm[:, :, 0])  # s22
                sigma_torchfem[:, :, 4] = sigma_denorm[:, :, 5]  # s23
                sigma_torchfem[:, :, 5] = (3*sigma_denorm[:, :, 0] - 
                                         sigma_torchfem[:, :, 0] - 
                                         sigma_torchfem[:, :, 3])  # s33
            else:
                sigma_torchfem[:, :, 2] = (sigma_denorm[:, :, 2] + 
                                         sigma_denorm[:, :, 0])  # s22
            
            # Update stress tensor for this integration point
            if self.n_dim == 3:
                stress[n, i, :, 0, 0] = sigma_torchfem[:, 0, 0]  # s11
                stress[n, i, :, 0, 1] = sigma_torchfem[:, 0, 1]  # s12
                stress[n, i, :, 0, 2] = sigma_torchfem[:, 0, 2]  # s13
                stress[n, i, :, 1, 0] = sigma_torchfem[:, 0, 1]  # s21
                stress[n, i, :, 1, 1] = sigma_torchfem[:, 0, 3]  # s22
                stress[n, i, :, 1, 2] = sigma_torchfem[:, 0, 4]  # s23
                stress[n, i, :, 2, 0] = sigma_torchfem[:, 0, 2]  # s31
                stress[n, i, :, 2, 1] = sigma_torchfem[:, 0, 4]  # s32
                stress[n, i, :, 2, 2] = sigma_torchfem[:, 0, 5]  # s33
            else:
                stress[n, i, :, 0, 0] = sigma_torchfem[:, 0, 0]  # s11
                stress[n, i, :, 0, 1] = sigma_torchfem[:, 0, 1]  # s12
                stress[n, i, :, 1, 0] = sigma_torchfem[:, 0, 1]  # s21
                stress[n, i, :, 1, 1] = sigma_torchfem[:, 0, 2]  # s22
            
            # Compute ddsdde via autograd from single forward pass
            ddsdde = torch.zeros(self.n_elem, self.n_stress, self.n_stress)
            for i_stress in range(self.n_stress):
                grad_output = torch.zeros_like(sigma_torchfem)
                grad_output[:, 0, i_stress] = 1.0
                
                grads = torch.autograd.grad(outputs=sigma_torchfem, 
                                          inputs=eps_torchfem_grad,
                                          grad_outputs=grad_output,
                                          retain_graph=True,
                                          create_graph=False)[0]
                ddsdde[:, i_stress, :] = grads
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element internal forces
            force_contrib = self.compute_f(detJ, B, stress[n, i].clone())
            f += w * force_contrib.reshape(-1, self.n_dim * n_nod)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                # Material stiffness
                BCB = torch.einsum(
                    "...ijpq,...qk,...il->...ljkp", ddsdde, B, B)
                BCB = BCB.reshape(-1, self.n_dim * n_nod, self.n_dim * n_nod)
                k += w * self.compute_k(detJ, BCB)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if nlgeom:
                # Geometric stiffness
                BSB = torch.einsum(
                    "...iq,...qk,...il->...lk", stress[n, i].clone(), B, B
                )
                zeros = torch.zeros_like(BSB)
                kg = torch.stack([BSB] + (self.n_dim - 1) * [zeros], dim=-1)
                kg = kg.reshape(-1, n_nod, self.n_dim * n_nod).unsqueeze(-2)
                zeros = torch.zeros_like(kg)
                kg = torch.stack([kg] + (self.n_dim - 1) * [zeros], dim=-2)
                kg = kg.reshape(-1, self.n_dim * n_nod, self.n_dim * n_nod)
                k += w * self.compute_k(detJ, kg)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return k, f
    # -------------------------------------------------------------------------
    def solve_msc(
        self,
        model_path: str,
        scaler_hydrostatic: float = 1375.297984380115,
        scaler_deviatoric: float = 324.7645473652983,
        increments: Tensor = torch.tensor([0.0, 1.0]),
        max_iter: int = 100,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        verbose: bool = False,
        method: Literal["spsolve", "minres", "cg", "pardiso"] | None = None,
        device: str | None = None,
        return_intermediate: bool = True,
        aggregate_integration_points: bool = True,
        use_cached_solve: bool = False,
        nlgeom: bool = False,
        return_volumes: bool = False,
        return_resnorm: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Solve the FEM problem using the MSC surrogate model.

        Args:
            model_path (str): Path to the MSC PyTorch model file.
            scaler_hydrostatic (float): Scaling factor for hydrostatic stress.
            scaler_deviatoric (float): Scaling factor for deviatoric stress.
            increments (Tensor): Load increment stepping.
            max_iter (int): Maximum number of iterations during Newton-Raphson.
            rtol (float): Relative tolerance for Newton-Raphson convergence.
            atol (float): Absolute tolerance for Newton-Raphson convergence.
            stol (float): Solver tolerance for iterative methods.
            verbose (bool): Print iteration information.
            method (str): Method for linear solve 
                ('spsolve','minres','cg','pardiso').
            device (str): Device to run the linear solve on.
            return_intermediate (bool): Return intermediate values if True.
            aggregate_integration_points (bool): Aggregate integration 
                points if True.
            use_cached_solve (bool): Use cached solve, e.g. in topology 
                optimization.
            nlgeom (bool): Use nonlinear geometry if True.
            return_volumes (bool): Return element volumes for each 
                increment if True.
            return_resnorm (bool): Return residual norm history for each 
                increment if True.

        Returns:
                Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Final 
                    displacements,
                    internal forces, stress, deformation gradient, and 
                    material state.
                If return_volumes=True, also returns element volumes with shape
                (num_increments, num_elem, 1).
                If return_resnorm=True, also returns residual history dict with
                increment numbers as keys and lists of residual norms as values.

        """
        # Load MSC model
        msc_model = torch.load(model_path, weights_only=True)

        # model.load_state_dict(torch.load('./trained_lmsc_u7_d4_w25_e1500_batch64_lrinit0.005_lrrate0.03_lrpower0.5', weights_only=True))
        msc_model.eval()
        
        # Number of increments
        N = len(increments)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Null space rigid body modes for AMG preconditioner
        B = self.compute_B()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize variables to be computed
        u = torch.zeros(N, self.n_nod, self.n_dim)
        f = torch.zeros(N, self.n_nod, self.n_dim)
        stress = torch.zeros(N, self.n_int, self.n_elem,
                             self.n_stress, self.n_stress)
        defgrad = torch.zeros(N, self.n_int, self.n_elem,
                              self.n_stress, self.n_stress)
        defgrad[:, :, :, :, :] = torch.eye(self.n_stress)
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize volumes if requested
        if return_volumes:
            volumes = torch.zeros(N, self.n_elem)
            # Compute initial volume for increment 0
            # Default field is ones to integrate volume
            volumes[0] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize residual norm history if requested
        if return_resnorm:
            residual_history = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize global stiffness matrix
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Incremental loading
        for n in range(1, N):
            # Increment size
            inc = increments[n] - increments[n - 1]
            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            de0 = inc * self.ext_strain
            # Newton-Raphson iterations
            # Initialize residual list for this increment if requested
            if return_resnorm:
                residual_history[n] = []
            
            for i in range(max_iter):
                du[con] = DU[con]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Element-wise integration with MSC surrogate
                k, f_i = self.integrate_material_msc(
                    msc_model, u, defgrad, stress, state, n, du, de0, nlgeom,
                    scaler_hydrostatic, scaler_deviatoric
                )
                if self.K.numel() == 0 or not self.material.n_state == 0 or \
                    nlgeom:
                    self.K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_i)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save residual norm to history if requested
                if return_resnorm:
                    residual_history[n].append(res_norm.item())
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Save initial residual
                if i == 0:
                    res_norm0 = res_norm
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Print iteration information
                if verbose:
                    print(f"Increment {n} | Iteration {i+1} | "
                          f"Residual: {res_norm:.5e}")
                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol:
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Use cached solve from previous iteration if available
                if i == 0 and use_cached_solve:
                    cached_solve = self.cached_solve
                else:
                    cached_solve = CachedSolve()
                # Only update cache on first iteration
                update_cache = i == 0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve for displacement increment
                du -= sparse_solve(
                    self.K,
                    residual,
                    B,
                    stol,
                    device,
                    method,
                    None,
                    cached_solve,
                    update_cache,
                )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check convergence
            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise Exception("Newton-Raphson iteration did not converge.")
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update increment
            f[n] = F_int.reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element volumes if requested
            if return_volumes:
                volumes[n] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Aggregate integration points as mean
        if aggregate_integration_points:
            defgrad = defgrad.mean(dim=1)
            stress = stress.mean(dim=1)
            state = state.mean(dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Squeeze outputs
        stress = stress.squeeze()
        defgrad = defgrad.squeeze()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        result = [u, f, stress, defgrad, state]
        # Return intermediate states
        if not return_intermediate:
            result = [x[-1] for x in result]
        # Return volumes
        if return_volumes:
            result.append(volumes if return_intermediate else volumes[-1])
        # Return residual norm
        if return_resnorm:
            result.append(residual_history)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(result)
    # -------------------------------------------------------------------------
    def _build_graph_topology(self, patch_id: int):
        """Build and store graph topology for a material patch.

        Args:
            patch_id (int): Material patch identifier.
        """
        # Get material patch nodes
        elem_nodes = self.elements[patch_id]
        elem_coords_ref = self.nodes[elem_nodes]
        node_coords_init = elem_coords_ref.detach().numpy()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Extract data in format expected by Graphorge - patch dimensions
        dim = self.n_dim
        if dim == 2:
            patch_dim = [1.0, 1.0]
            n_elem_per_dim = [1, 1]
        elif dim == 3:
            patch_dim = [1.0, 1.0, 1.0]
            n_elem_per_dim = [1, 1, 1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create mesh matrix for single-element material patch
        if isinstance(self.etype, Quad1):
            elem_order = 1
            # mesh_nodes_matrix = np.array([[0, 1], [2, 3]])
        elif isinstance(self.etype, Quad2):
            elem_order = 2
        # Initialize mesh_nodes_matrix
        mesh_nx = mesh_ny = mesh_nz = 1
        n_nod = self.etype.nodes
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate GNN-based material patch graph data
        gnn_patch_data = GraphData(
            n_dim=dim, nodes_coords=node_coords_init)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set connectivity radius based on finite element size
        connect_radius = 4 * np.sqrt(np.sum([x**2 for x in 
            get_elem_size_dims(patch_dim, n_elem_per_dim, dim)]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get boundary node information
        # Note: for single element all nodes are boundary nodes
        bd_node_indices = list(range(n_nod))
        boundary_node_set = set(bd_node_indices)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create mapping from original to boundary indices
        original_to_boundary_idx = {node_id: position for position, 
                                    node_id in enumerate(bd_node_indices)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get finite element mesh edges for all nodes
        connected_nodes_all = get_mesh_connected_nodes(
            dim, mesh_nodes_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Filter connected_nodes to only include boundary node pairs
        connected_nodes_boundary = []
        for node1, node2 in connected_nodes_all:
            if node1 in boundary_node_set and node2 in boundary_node_set:
                boundary_node1 = original_to_boundary_idx[node1]
                boundary_node2 = original_to_boundary_idx[node2]
                connected_nodes_boundary.append(
                    (boundary_node1, boundary_node2))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Organize connected nodes as tuples
        connected_nodes = tuple(connected_nodes_boundary)
        edges_indexes_mesh = GraphData.get_edges_indexes_mesh(
            connected_nodes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GNN-based material patch graph edges
        gnn_patch_data.set_graph_edges_indexes(
            connect_radius=connect_radius,
            edges_indexes_mesh=edges_indexes_mesh)
        edges_indexes = torch.tensor(
            np.transpose(
                copy.deepcopy(gnn_patch_data.get_graph_edges_indexes())),
                dtype=torch.long)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._edges_indexes[f"patch_{patch_id}"] = edges_indexes
    # -------------------------------------------------------------------------
    def surrogate_integrate_material(
        self, model, u: Tensor, n: int, du: Tensor,
        is_stepwise: bool = False,
        patch_ids: Tensor = None,
        hidden_states: dict = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform surrogate integration using Graphorge material patch model.
        
        Args:
            model (GNNEPDBaseModel): Pre-loaded Graphorge material patch model 
                configured for inference.
            u (Tensor): Displacement history tensor of shape (n_increments, 
                n_nodes, n_dim). Only current displacement u[n] is used for 
                graph construction.
            n (int): Current increment number (0-indexed).
            du (Tensor): Displacement increment tensor of shape (n_dofs,). 
                Used to update current displacement configuration.
            is_stepwise (bool): Whether to use stepwise RNN mode with 
                hidden state tracking between steps. Defaults to False.
            
        Returns:
            Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, dict]: A tuple
                containing:
                - k (Tensor): Element stiffness matrices of shape (n_elem,
                  n_dof_elem, n_dof_elem).
                - f (Tensor): Element internal force vectors of shape (n_elem,
                  n_dof_elem).
                - hidden_states_out (dict, optional): Updated hidden states
                  for each patch (only returned in stepwise mode).
                  
        Note:
            This method constructs graphs using only the current step 
            displacement and coordinates (single-step approach), regardless of 
            the mode. The distinction between stepwise and non-stepwise modes 
            affects only how the GNN model processes the graph data internally.
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize patch-specific hidden states output
        hidden_states_out = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update current displacement
        u[n] = u[n - 1] + du.view((-1, self.n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize nodal force and stiffness
        n_nod = self.etype.nodes
        n_dof_elem = self.n_dim * n_nod
        f = torch.zeros(self.n_elem, n_dof_elem)
        k = torch.zeros((self.n_elem, n_dof_elem, n_dof_elem))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Determine which elements to process based on patch_ids
        if patch_ids is not None:
            # patch_ids contains the actual patch IDs to process
            # Convert to list for iteration
            patch_indices = patch_ids.tolist()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over element indices
        for idx_patch in patch_indices:
            # Get patch identifier for stepwise mode
            patch_id = f"patch_{idx_patch}"
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material patch nodes
            elem_nodes = self.elements[idx_patch]  
            # Current displacement at increment n
            elem_u_current = u[n, elem_nodes, :].clone()
            # Extract material patch nodal coordinates
            elem_coords_ref = self.nodes[elem_nodes]
            # Get pre-computed edges_indexes for this patch
            edges_indexes = self._edges_indexes[patch_id]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            def detach_hidden_states(states):
                if isinstance(states, dict):
                    return {k: detach_hidden_states(v) for k, v in states.items()}
                elif isinstance(states, list):
                    return [detach_hidden_states(item) for item in states]
                elif torch.is_tensor(states):
                    return states.detach()
                else:
                    return states
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Load patch-specific hidden states into model before inference
            if is_stepwise and hidden_states and patch_id in hidden_states:
                # patch_hidden = copy.deepcopy(hidden_states[patch_id])
                patch_hidden = copy.deepcopy(
                    detach_hidden_states(hidden_states[patch_id]))
                model._gnn_epd_model._hidden_states = patch_hidden
                # Set the model's hidden states to this patch's states
                if 'encoder' in patch_hidden:
                    model._gnn_epd_model._encoder._hidden_states = \
                        patch_hidden['encoder']
                if 'processor' in patch_hidden:
                    model._gnn_epd_model._processor._hidden_states = \
                        patch_hidden['processor']
                    for i, layer in enumerate(
                        model._gnn_epd_model._processor._processor):
                        layer_key = f'layer_{i}'
                        if layer_key in patch_hidden['processor']:
                            layer._hidden_states = \
                                patch_hidden['processor'][layer_key]
                if 'decoder' in patch_hidden:
                    model._gnn_epd_model._decoder._hidden_states = \
                        patch_hidden['decoder']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Model prediction: stepwise vs non-stepwise mode
            hidden_states_trial = None
            def forward(disp):
                """Forward function for gradient computation."""
                nonlocal hidden_states_trial
                if is_stepwise:
                    node_prediction, hidden_states_out = forward_graph(
                        model=model,
                        disps=disp, coords_ref=elem_coords_ref,
                        edges_indexes=edges_indexes, n_dim=self.n_dim)
                    hidden_states_trial = hidden_states_out
                else:
                    node_prediction = forward_graph(
                        model=model,
                        disps=disp, coords_ref=elem_coords_ref,
                        edges_indexes=edges_indexes, n_dim=self.n_dim)
                return node_prediction, node_prediction.detach()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute Jacobian: d(forces_t)/d(u)
            # Use torch.func.jacrev for vectorized Jacobian computation
            # jacrev computes J[i,j] = d(output[i])/d(input[j])
            (jacobian, node_output) = torch_func.jacfwd(
                forward, has_aux=True)(elem_u_current)
            node_forces_trial = node_output.flatten()
            # Reshape Jacobian to stiffness matrix format
            # jacobian shape: (n_nodes, n_dim, n_nodes, n_dim)
            # Reshape to: (n_nodes*n_dim, n_nodes*n_dim)
            stiffness_matrix = jacobian.view(n_dof_elem, n_dof_elem)

            # Store updated hidden states for stepwise mode
            if is_stepwise and hidden_states_trial is not None:
                hidden_states_out[patch_id] = hidden_states_trial

            # Store forces
            f[idx_patch] = node_forces_trial.flatten()
            # Store stiffness matrix
            k[idx_patch] = stiffness_matrix
            breakpoint()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
        if is_stepwise:
            return k, f, hidden_states_out
        else:    
            return k, f
    # -------------------------------------------------------------------------
    def solve_matpatch(
        self,
        is_mat_patch: Tensor,
        increments: Tensor = torch.tensor([0.0, 1.0]),
        max_iter: int = 100,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        verbose: bool = False,
        method: Literal["spsolve", "minres", "cg", "pardiso"] | None = None,
        device: str | None = None,
        return_intermediate: bool = True,
        aggregate_integration_points: bool = True,
        use_cached_solve: bool = False,
        nlgeom: bool = False,
        return_volumes: bool = False,
        is_stepwise: bool = False,
        model_directory: str | None = None,
        return_resnorm: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, dict] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Solve the FEM problem with material patches and the surrogate model.

        Args:
            is_mat_patch (Tensor): Element-wise flag indicating material patch 
                usage of shape (n_elem,). Values: 0 = use standard 
                integration, >0 = material patch ID for surrogate integration.
            increments (Tensor): Load increment stepping of shape 
                (n_increments,). Defaults to torch.tensor([0.0, 1.0]).
            max_iter (int): Maximum number of iterations during Newton-Raphson.
                Defaults to 100.
            rtol (float): Relative tolerance for Newton-Raphson convergence.
                Defaults to 1e-8.
            atol (float): Absolute tolerance for Newton-Raphson convergence.
                Defaults to 1e-6.
            stol (float): Solver tolerance for iterative methods. 
                Defaults to 1e-10.
            verbose (bool): Whether to print iteration information. 
                Defaults to False.
            method (str, optional): Method for linear solve ('spsolve', 
                'minres', 'cg', 'pardiso'). Defaults to None for automatic 
                selection.
            device (str, optional): Device to run the linear solve on. 
                Defaults to None.
            return_intermediate (bool): Whether to return intermediate values. 
                Defaults to True.
            aggregate_integration_points (bool): Whether to aggregate 
                integration points. Defaults to True.
            use_cached_solve (bool): Whether to use cached solve for 
                optimization. Defaults to False.
            nlgeom (bool): Whether to use nonlinear geometry. 
                Defaults to False.
            return_volumes (bool): Whether to return element volumes for each 
                increment. Defaults to False.
            is_stepwise (bool): Whether to use stepwise RNN mode for 
                surrogate integration. Defaults to False.
            model_directory (str, optional): Path to trained Graphorge model. 
                If None, uses default path. Defaults to None.
            return_resnorm (bool): Whether to return residual norm history. 
                Defaults to False.

        Returns:
            Tuple[Tensor, ...]: If return_volumes=False and return_resnorm=False, 
                returns 5-tuple of (displacements, forces, stress, 
                deformation_gradient, state). If return_volumes=True, returns 
                6-tuple with volumes added. If return_resnorm=True, returns 
                additional residual_history dict as last element.
                If return_intermediate=True, returns full history arrays.
                If return_intermediate=False, returns only final values.
                
        Raises:
            ValueError: If is_mat_patch shape doesn't match number of elements.
            Exception: If Newton-Raphson iteration fails to converge.
        """
        # Validate is_mat_patch tensor
        if is_mat_patch.shape[0] != self.n_elem:
            raise ValueError(f'is_mat_patch shape {is_mat_patch.shape} ' + \
                             f'must match number of elements {self.n_elem}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Number of increments
        N = len(increments)
        # Null space rigid body modes for AMG preconditioner
        # B = self.compute_B()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize variables to be computed
        u = torch.zeros(N, self.n_nod, self.n_dim)
        f = torch.zeros(N, self.n_nod, self.n_dim)
        stress = torch.zeros(N, self.n_int, self.n_elem,
                             self.n_stress, self.n_stress)
        defgrad = torch.zeros(N, self.n_int, self.n_elem,
                              self.n_stress, self.n_stress)
        defgrad[:, :, :, :, :] = torch.eye(self.n_stress)
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize volumes if requested
        if return_volumes:
            volumes = torch.zeros(N, self.n_elem)
            # Compute initial volume for increment 0
            volumes[0] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize residual norm history if requested
        if return_resnorm:
            residual_history = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize global stiffness matrix
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize material patch data structures for surrogate integration
        patch_mask = is_mat_patch >= 0
        model = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load surrogate model
        if torch.any(patch_mask):
            # Use provided model directory or default path
            if model_directory is None:
                model_directory = (
                    "/Users/rbarreira/Desktop/machine_learning/material_patches/"
                    "graphorge_material_patches/src/graphorge/projects/"
                    "material_patches/elastic/2d/quad4/mesh1x1/ninc1/"
                    "26.1_force_equilibrium_npath10000/reference/3_model")
            model = self._load_Graphorge_model(
                model_directory=model_directory,
                device_type=device if device else 'cpu')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Enable stepwise mode
            # Temporarily set attributes for single-step mode (_n_time_* = 1)
            if is_stepwise:
                # Store original attributes for stepwise data scaler access
                # orig_attrs = 
                model._save_time_series_attrs()
                # model._original_time_attrs = orig_attrs
                # Enable stepwise mode
                model.set_rnn_mode(is_stepwise=True)
            
            # Initialize patch_id dictionary with None values
            patch_ids = torch.unique(is_mat_patch[is_mat_patch >= 0])
            # Initialize edges_indexes dict
            self._edges_indexes = {}
            for pid in patch_ids:
                # Build graph topology for this material patch
                self._build_graph_topology(pid)
                # Initialize hidden states structure for GNN model
                # Same structure as graphorge:
                # - encoder: for encoding layers
                # - processor: for message passing layers (layer_0, layer_1, etc.)
                # - decoder: for decoding layers
                # Each layer can have node, edge, global hidden states
                
                # Get number of message passing steps from model if available
                n_message_steps = model._n_message_steps
                
                # Create processor hidden states for each layer
                processor_hidden = {}
                for layer_idx in range(n_message_steps):
                    processor_hidden[f'layer_{layer_idx}'] = {
                        'node': None,
                        'edge': None, 
                        'global': None
                    }
                
                hidden_states_dict = {
                    f"patch_{pid.item()}": {
                        'encoder': {
                            'node': None,
                            'edge': None,
                            'global': None
                        },
                        'processor': processor_hidden,
                        'decoder': {
                            'node': None,
                            'edge': None,
                            'global': None
                        }
                    }
                }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Incremental loading
        for n in range(1, N):
            # Increment size
            inc = increments[n] - increments[n - 1]
            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            de0 = inc * self.ext_strain
            # Initialize residual norm list for this increment if requested
            if return_resnorm:
                residual_history[n] = []
            # Newton-Raphson iterations
            for i in range(max_iter):
                du[con] = DU[con]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Element-wise integration with material patch support
                # Initialize element stiffness and force arrays
                n_nod = self.etype.nodes
                k = torch.zeros((self.n_elem, self.n_dim * n_nod,
                                 self.n_dim * n_nod))
                f_i = torch.zeros(self.n_elem, self.n_dim * n_nod)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # # Standard integration for elements with is_mat_patch == 0
                # std_mask = is_mat_patch == 0
                # if torch.any(std_mask):
                #     # Create temporary arrays for standard integration
                #     std_indices = torch.nonzero(
                #         std_mask, as_tuple=False).squeeze(-1)
                #     if std_indices.numel() > 0:
                #         k_std, f_std = self.integrate_material(
                #             u, defgrad, stress, state, n, du, de0, nlgeom
                #         )
                #         k[std_indices] = k_std[std_indices]
                #         f_i[std_indices] = f_std[std_indices]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Surrogate integration for elements with is_mat_patch >= 0
                patch_mask = is_mat_patch >= 0
                if torch.any(patch_mask):
                    # Get unique patch IDs and process all at once
                    patch_ids = torch.unique(is_mat_patch[patch_mask])
                    # Call surrogate integration
                    # profiler_surr = cProfile.Profile()
                    # profiler_surr.enable()
                    if is_stepwise:
                        k_surr, f_surr, hidden_state_out = \
                            self.surrogate_integrate_material(
                            model, u, n, du,
                            is_stepwise=is_stepwise,
                            patch_ids=patch_ids,
                            hidden_states=hidden_states_dict)
                    else:
                        k_surr, f_surr = \
                            self.surrogate_integrate_material(
                            model, u, n, du,
                            is_stepwise=is_stepwise,
                            patch_ids=patch_ids)
                    # profiler_surr.disable()
                    # print(f"\n=== surrogate_integrate_material PROFILE (increment {n}) ===")
                    # stats_surr = pstats.Stats(profiler_surr)
                    # stats_surr.sort_stats('cumulative').print_stats(10)
                    # Assemble results for all patches
                    k[patch_mask] = k_surr[patch_mask]
                    f_i[patch_mask] = f_surr[patch_mask]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble global stiffness matrix and internal force vector
                if self.K.numel() == 0 or not self.material.n_state == 0 or \
                    nlgeom:
                    self.K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_i)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)
                # Save residual norm to history if requested
                if return_resnorm:
                    residual_history[n].append(res_norm.item())
                # Save initial residual for relative error
                if i == 0:
                    res_norm0 = res_norm
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Print iteration information
                if verbose:
                    print(f"Increment {n} | Iteration {i+1} | "
                          f"Residual: {res_norm:.5e}")
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol:
                    # Update patch-specific hidden states with converged values
                    if is_stepwise:
                        # Update patch-specific hidden states 
                        # with converged values
                        # hidden_state_out contains the updated states for 
                        # processed patches
                        for pid in patch_ids:
                            patch_key = f"patch_{pid.item()}"
                            # Update the patch-specific hidden states
                            hidden_states_dict[patch_key] = hidden_state_out[
                                patch_key]
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Use cached solve from previous iteration if available
                if i == 0 and use_cached_solve:
                    cached_solve = self.cached_solve
                else:
                    cached_solve = CachedSolve()
                # Only update cache on first iteration
                update_cache = i == 0
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve for displacement increment
                du -= sparse_solve(
                    self.K,
                    residual,
                    None,
                    stol,
                    device,
                    method,
                    None,
                    cached_solve,
                    update_cache,
                )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Raise an Exception if the model did not converge
            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise Exception("Newton-Raphson iteration did not converge.")
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update increment
            f[n] = F_int.reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element volumes if requested
            if return_volumes:
                volumes[n] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Disable stepwise mode after computation
        if is_stepwise:
            model.set_rnn_mode(is_stepwise=False)
            model._restore_time_series_attrs()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Aggregate integration points as mean
        if aggregate_integration_points:
            defgrad = defgrad.mean(dim=1)
            stress = stress.mean(dim=1)
            state = state.mean(dim=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Squeeze outputs
        stress = stress.squeeze()
        defgrad = defgrad.squeeze()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        result = [u, f, stress, defgrad, state]
        # Return intermediate states
        if not return_intermediate:
            result = [x[-1] for x in result]
        # Return volumes
        if return_volumes:
            result.append(volumes if return_intermediate else volumes[-1])
        # Return residual norm
        if return_resnorm:
            result.append(residual_history)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(result)
# =============================================================================
def compute_edge_features(coords_hist, disps_hist, edges_indexes, 
                              n_dim):
        """Compute edge features following the logic of features.py.
        
        Computes edge features ('edge_vector', 'relative_disp').
        
        Parameters
        ----------
        coords_hist : torch.Tensor
            Node coordinates history (coord_hist feature),
            shape (n_nodes, n_time_steps*n_dim)
        disps_hist : torch.Tensor  
            Node displacements history (disp_hist feature),
            shape (n_nodes, n_time_steps*n_dim)
        edges_indexes : torch.Tensor
            Edge connectivity, shape (2, n_edges)
        n_dim : int
            Spatial dimensions
            
        Returns
        -------
        edge_features : torch.Tensor
            Edge features [edge_vector, relative_disp],
            shape (n_edges, 2*n_time_steps*n_dim)
        """
        # Get edge connections
        # Source nodes
        edge_sources = edges_indexes[0]
        # Target nodes
        edge_targets = edges_indexes[1]  
        n_edges = edge_sources.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute number of time steps
        n_coords_features = coords_hist.shape[1]
        n_time_steps = n_coords_features // n_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Edge features: [edge_vector, relative_disp]
        # Each edge has n_time_steps * n_dim components
        edge_vector_size = n_time_steps * n_dim
        relative_disp_size = n_time_steps * n_dim
        n_edge_features = edge_vector_size + relative_disp_size
        edge_features = torch.zeros(
            n_edges, n_edge_features, device=coords_hist.device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # coords_hist already contains current coordinates
        # (reference + displacements)
        current_coords_hist = coords_hist
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over edges
        for k in range(n_edges):
            # Source node
            i = edge_sources[k]
            # Target node
            j = edge_targets[k]  
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute edge_vector: current_coords[i] - current_coords[j]
            # See lines 443-445 in features.py
            for t in range(n_time_steps):
                # Set start and end indices for each time step
                start_idx = t * n_dim
                end_idx = (t + 1) * n_dim
                # Edge vector for time step t
                edge_vector_t = (current_coords_hist[i, start_idx:end_idx] - 
                               current_coords_hist[j, start_idx:end_idx])
                # Store in edge features
                edge_features[k, start_idx:end_idx] = edge_vector_t
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute relative_disp: disps[i] - disps[j]  
            # See lines 505-508 in features.py
            for t in range(n_time_steps):
                # Set start and end indices for each time step
                start_idx = t * n_dim
                end_idx = (t + 1) * n_dim
                # Relative displacement for time step t
                relative_disp_t = (disps_hist[i, start_idx:end_idx] - 
                                 disps_hist[j, start_idx:end_idx]) 
                # Store in edge features (after edge_vector section)
                edge_features[k,edge_vector_size + \
                              start_idx:edge_vector_size + end_idx] = \
                                relative_disp_t
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edge_features
# =============================================================================
def forward_graph(
        model, disps, coords_ref, edges_indexes, n_dim,
        batch_vector=None, global_features_in=None):
        """Forward pass through GNN model with graph data reconstruction.

        Args:
            model: Trained GNN model for material patch prediction.
            disps (Tensor): Nodal displacements of shape (n_nodes, n_dim).
            coords_ref (Tensor): Reference nodal coordinates of shape
                (n_nodes, n_dim).
            edges_indexes (Tensor): Edge connectivity of shape (2, n_edges).
            n_dim (int): Spatial dimension (2 or 3).
            batch_vector (Tensor, optional): Batch vector for multiple
                patches. Defaults to None.
            global_features_in (Tensor, optional): Global features.
                Defaults to None.

        Returns:
            Tensor: Predicted nodal forces of shape (n_nodes, n_dim).
        """
        coords = coords_ref + disps
        node_features_in = torch.cat([coords, disps], dim=1)
        # Recompute edge features based on updated data coords:
        edge_features_in = compute_edge_features(
            coords, disps, edges_indexes, n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Forward pass with updated features
        # Stepwise forward mode
        if model._is_stepwise:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize updated node features
            node_features_norm = model.stepwise_data_scaler_transform(
                tensor=node_features_in,
                features_type='node_features_in',
                mode='normalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize updated edge features
            edge_features_norm = model.stepwise_data_scaler_transform(
                tensor=edge_features_in,
                features_type='edge_features_in',
                mode='normalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Use stepwise forward method
            node_output_norm, _, _, hidden_states_out = model.step(
                node_features_in=node_features_norm,
                edge_features_in=edge_features_norm,
                global_features_in=global_features_in,
                edges_indexes=edges_indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Denormalize output to get real forces
            node_output_real = model.stepwise_data_scaler_transform(
                tensor=node_output_norm,
                features_type='node_features_out',
                mode='denormalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return node_output_real, hidden_states_out
        # Regular forward mode
        else:
            # Normalize updated node features
            node_features_norm = model.data_scaler_transform(
                tensor=node_features_in,
                features_type='node_features_in',
                mode='normalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize updated edge features
            edge_features_norm = model.data_scaler_transform(
                tensor=edge_features_in,
                features_type='edge_features_in',
                mode='normalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Use regular forward method
            node_output_norm, _, _ = model(
                node_features_in=node_features_norm,
                edge_features_in=edge_features_norm,
                global_features_in=global_features_in,
                edges_indexes=edges_indexes,
                batch_vector=batch_vector)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Denormalize output to get real forces
            node_output_real = model.data_scaler_transform(
                tensor=node_output_norm,
                features_type='node_features_out',
                mode='denormalize')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return node_output_real