from abc import ABC, abstractmethod
from typing import Literal, Tuple
import numpy as np
import torch
from torch import Tensor

from .elements import Element, Quad1, Quad2, Hexa1, Hexa2
from .materials import Material
from .sparse import CachedSolve, sparse_solve

import torch_geometric.data as pyg_data
from graphorge.gnn_base_model.model.gnn_model import GNNEPDBaseModel
from graphorge.gnn_base_model.data.graph_data import GraphData
from graphorge.projects.material_patches.gnn_model_tools.gen_graphs_files \
    import (
    get_elem_size_dims, get_mesh_connected_nodes, 
    extract_fem_data_from_pkl)
from graphorge.projects.material_patches.gnn_model_tools.features import (
    GNNPatchFeaturesGenerator)
from graphorge.gnn_base_model.model.custom_layers import (
    compute_stiffness_matrix)


class FEM(ABC):
    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize a general FEM problem."""
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
        return self._forces
    # -------------------------------------------------------------------------
    @forces.setter
    def forces(self, value: Tensor):
        if not value.shape == self.nodes.shape:
            raise ValueError("Forces must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Forces must be a floating-point tensor.")
        self._forces = value.to(self.nodes.device)
    # -------------------------------------------------------------------------
    @property
    def displacements(self) -> Tensor:
        return self._displacements
    # -------------------------------------------------------------------------
    @displacements.setter
    def displacements(self, value: Tensor):
        if not value.shape == self.nodes.shape:
            raise ValueError("Displacements must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Displacements must be a floating-point tensor.")
        self._displacements = value.to(self.nodes.device)
    # -------------------------------------------------------------------------
    @property
    def constraints(self) -> Tensor:
        return self._constraints
    # -------------------------------------------------------------------------
    @constraints.setter
    def constraints(self, value: Tensor):
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
        raise NotImplementedError
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        raise NotImplementedError
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        raise NotImplementedError
    # -------------------------------------------------------------------------
    @abstractmethod
    def plot(self, u: float | Tensor = 0.0, **kwargs):
        raise NotImplementedError
    # -------------------------------------------------------------------------
    def compute_B(self) -> Tensor:
        """Null space representing rigid body modes."""
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
                H_inc, F[n - 1, i], stress[n - 1, i], state[n - 1, i], de0
            )
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
    def surrogate_integrate_material(
        self, model, u: Tensor, n: int, du: Tensor,
        is_stateful_mode: bool = False,
        is_converged: bool = False,
        patch_data: dict = None
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
            is_stateful_mode (bool): Whether to use stateful RNN mode with 
                hidden state tracking between steps. Defaults to False.
            patch_data (dict, optional): Additional patch data for future 
                extensions. Currently unused. Defaults to None.
            
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - k (Tensor): Element stiffness matrices of shape (n_elem, 
                  n_dof_elem, n_dof_elem).
                - f (Tensor): Element internal force vectors of shape (n_elem, 
                  n_dof_elem).
                  
        Note:
            This method constructs graphs using only the current step 
            displacement and coordinates (single-step approach), regardless of 
            the mode. The distinction between stateful and non-stateful modes 
            affects only how the GNN model processes the graph data internally.
        """
        # Update current displacement
        u[n] = u[n - 1] + du.view((-1, self.n_dim))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize nodal force and stiffness
        n_nod = self.etype.nodes
        n_dof_elem = self.n_dim * n_nod
        f = torch.zeros(self.n_elem, n_dof_elem)
        k = torch.zeros((self.n_elem, n_dof_elem, n_dof_elem))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for idx_elem in range(self.n_elem):
            # Get material patch nodes
            elem_nodes = self.elements[idx_elem]  
            # Extract material patch nodal coordinates and displacements
            elem_coords_ref = self.nodes[elem_nodes]
            # Current displacement at increment n
            elem_u_current = u[n, elem_nodes, :]
            
            # Get patch identifier for stateful mode
            patch_id = f"patch_{idx_elem}"
            
            # Convert to numpy for GraphData (reference coordinates)
            node_coords_init = elem_coords_ref.detach().numpy()
        
            # For single time step, create artificial time series
            nodes_coords_hist = (elem_coords_ref + elem_u_current).detach(
                ).numpy()[:, :, np.newaxis]
            nodes_disps_hist = elem_u_current.detach().numpy(
                )[:, :, np.newaxis]
            
            # Forces history (not available, set to zeros)
            nodes_int_forces_hist = np.zeros_like(nodes_disps_hist)
            
            # Extract data in format expected by Graphorge
            dim = self.n_dim
            if dim == 2:
                patch_dim = [1.0, 1.0]
                n_elem_per_dim = [1, 1]
            elif dim == 3:
                patch_dim = [1.0, 1.0, 1.0]
                n_elem_per_dim = [1, 1, 1]

            # Create mesh matrix for single element
            if isinstance(self.etype, Quad1):
                mesh_nodes_matrix = np.array([[0, 1], [2, 3]])
            else:
                # For other elements, create connectivity based on nodes
                mesh_nodes_matrix = np.arange(n_nod).reshape(-1, 1)
            
            # Instantiate GNN-based material patch graph data
            gnn_patch_data = GraphData(
                n_dim=dim, nodes_coords=node_coords_init)
            
            # Set connectivity radius based on finite element size
            connect_radius = 4 * np.sqrt(np.sum([x**2 for x in 
                get_elem_size_dims(patch_dim, n_elem_per_dim, dim)]))
            
            # Get boundary node information (for single element, 
            # all nodes are boundary)
            bd_node_indices = list(range(n_nod))
            boundary_node_set = set(bd_node_indices)
            
            # Create mapping from original to boundary indices
            original_to_boundary_idx = {node_id: position for position, 
                                      node_id in enumerate(bd_node_indices)}
            
            # Get finite element mesh edges for ALL nodes
            connected_nodes_all = get_mesh_connected_nodes(
                dim, mesh_nodes_matrix)
            
            # Filter connected_nodes to only include boundary node pairs
            connected_nodes_boundary = []
            for node1, node2 in connected_nodes_all:
                if node1 in boundary_node_set and node2 in boundary_node_set:
                    boundary_node1 = original_to_boundary_idx[node1]
                    boundary_node2 = original_to_boundary_idx[node2]
                    connected_nodes_boundary.append(
                        (boundary_node1, boundary_node2))
            
            # Organize connected nodes as tuples
            connected_nodes = tuple(connected_nodes_boundary)
            edges_indexes_mesh = GraphData.get_edges_indexes_mesh(
                connected_nodes)
            
            # Set GNN-based material patch graph edges
            gnn_patch_data.set_graph_edges_indexes(
                connect_radius=connect_radius,
                edges_indexes_mesh=edges_indexes_mesh)
            
            # Create features generator following Graphorge approach
            features_generator = GNNPatchFeaturesGenerator(
                n_dim=self.n_dim,
                nodes_coords_hist=nodes_coords_hist,
                edges_indexes=gnn_patch_data.get_graph_edges_indexes(),
                nodes_disps_hist=nodes_disps_hist,
                nodes_int_forces_hist=nodes_int_forces_hist)
            
            # Build node and edge features as done in training
            n_time_steps = nodes_coords_hist.shape[2]
            node_features_matrix = \
                features_generator.build_nodes_features_matrix(
                features=('coord_hist', 'disp_hist'), 
                n_time_steps=n_time_steps)
            

            edge_features_matrix = \
                features_generator.build_edges_features_matrix(
                features=('edge_vector', 'relative_disp'), 
                n_time_steps=n_time_steps)
        

            # Set GNN-based material patch graph node and edges features
            gnn_patch_data.set_node_features_matrix(node_features_matrix)
            gnn_patch_data.set_edge_features_matrix(edge_features_matrix)
            
            # Get PyG homogeneous graph data object
            pyg_graph = gnn_patch_data.get_torch_data_object()
            
            # Get input features and normalize
            node_features_in, edge_features_in, global_features_in, \
                edges_indexes = (model.get_input_features_from_graph(
                    pyg_graph, is_normalized=True))




            # Model prediction: stateful vs non-stateful mode
            # Stateful mode: single step prediction with hidden state tracking
            if is_stateful_mode:
                # Step - forward pass for forces
                node_features_out, _, _ = model.step(
                    node_features_current=node_features_in,
                    edge_features_current=edge_features_in,
                    global_features_current=global_features_in,
                    edges_indexes=edges_indexes,
                    patch_id=patch_id,
                    is_iteration=not is_converged)
                print(f'model.step: {node_features_out}')
                # Compute stiffness matrix
                stiffness_matrix = compute_stiffness_matrix(
                    model=model,
                    node_features_in=node_features_in,
                    edge_features_in=edge_features_in,
                    global_features_in=global_features_in,
                    edges_indexes=edges_indexes,
                    batch_vector=None,
                    n_dim=self.n_dim)
                print(f'compute_stiffness_matrix: {stiffness_matrix}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Prediction without hidden state tracking (FFNNs)
            else:
                # Forward pass for forces
                node_features_out, _, _ = model(
                    node_features_in=node_features_in,
                    edge_features_in=edge_features_in,
                    global_features_in=global_features_in,
                    edges_indexes=edges_indexes,
                    batch_vector=None)
                # print(f'model: {node_features_out}')
                # Compute stiffness matrix
                stiffness_matrix = compute_stiffness_matrix(
                    model=model,
                    node_features_in=node_features_in,
                    edge_features_in=edge_features_in,
                    global_features_in=global_features_in,
                    edges_indexes=edges_indexes,
                    batch_vector=None,
                    n_dim=self.n_dim)
                # print(f'compute_stiffness_matrix: {stiffness_matrix}')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store stiffness matrix (already in correct torchFEM ordering)
            k[idx_elem] = stiffness_matrix
            # print(f'stiffness matrix: {k[idx_elem]}')
            # k[idx_elem] = torch.tensor([
            #     [ 68111.4551,  30406.8996, -47434.7634,   9730.2079, 
            #      -34055.7276, -30406.8996,  13379.0358,  -9730.2079],
            #     [ 30406.8996,  68111.4551,  -9730.2079,  13379.0358, 
            #      -30406.8996, -34055.7276,   9730.2079, -47434.7634],
            #     [-47434.7634,  -9730.2079,  68111.4551, -30406.8996, 
            #       13379.0358,   9730.2079, -34055.7276,  30406.8996],
            #     [  9730.2079,  13379.0358, -30406.8996,  68111.4551, 
            #       -9730.2079, -47434.7634,  30406.8996, -34055.7276],
            #     [-34055.7276, -30406.8996,  13379.0358,  -9730.2079, 
            #       68111.4551,  30406.8996, -47434.7634,   9730.2079],
            #     [-30406.8996, -34055.7276,   9730.2079, -47434.7634, 
            #       30406.8996,  68111.4551,  -9730.2079,  13379.0358],
            #     [ 13379.0358,   9730.2079, -34055.7276,  30406.8996, 
            #      -47434.7634,  -9730.2079,  68111.4551, -30406.8996],
            #     [ -9730.2079, -47434.7634,  30406.8996, -34055.7276, 
            #        9730.2079,  13379.0358, -30406.8996,  68111.4551]])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Denormalize output to get real forces
            node_forces_real = model.data_scaler_transform(
                tensor=node_features_out,
                features_type='node_features_out',
                mode='denormalize')
            # Store forces
            f[idx_elem] = node_forces_real.flatten() 
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
            load_model_state='best',
            is_remove_posterior=False)
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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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

        Returns:
                Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Final 
                    displacements,
                    internal forces, stress, deformation gradient, and 
                    material state.
                If return_volumes=True, also returns element volumes with shape
                (num_increments, num_elem, 1).

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
            for i in range(max_iter):
                du[con] = DU[con]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Element-wise integration
                k, f_i = self.integrate_material(
                    u, defgrad, stress, state, n, du, de0, nlgeom
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
        if return_intermediate:
            # Return all intermediate values
            if return_volumes:
                return u, f, stress, defgrad, state, volumes
            else:
                return u, f, stress, defgrad, state
        else:
            # Return only the final values
            if return_volumes:
                return u[-1], f[-1], stress[-1], defgrad[-1], state[-1], \
                    volumes[-1]
            else:
                return u[-1], f[-1], stress[-1], defgrad[-1], state[-1]
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
        is_stateful_mode: bool = False,
        model_directory: str | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            is_stateful_mode (bool): Whether to use stateful RNN mode for 
                surrogate integration. Defaults to False.
            model_directory (str, optional): Path to trained Graphorge model. 
                If None, uses default path. Defaults to None.

        Returns:
            Tuple[Tensor, ...]: If return_volumes=False, returns 5-tuple of
                (displacements, forces, stress, deformation_gradient, state).
                If return_volumes=True, returns 6-tuple with volumes added.
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
        # Initialize global stiffness matrix
        self.K = torch.empty(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize material patch data structures for surrogate integration
        patch_mask = is_mat_patch > 0
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
            # Enable stateful mode if requested
            # Temporarily set attributes for single-step mode (_n_time_* = 1)
            if is_stateful_mode:
                model.set_stateful_mode(True)
                orig_attrs = model._save_time_series_attrs()
                model._set_single_step_attrs()
                # Store original attributes for stateful data scaler access
                model._original_time_attrs = orig_attrs
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
                # Surrogate integration for elements with is_mat_patch > 0
                patch_mask = is_mat_patch > 0
                if torch.any(patch_mask):
                    # Get unique patch IDs
                    patch_ids = torch.unique(is_mat_patch[patch_mask])
                    # Loop over patches
                    for patch_id in patch_ids:
                        # print(f'patch_id: {patch_id}')
                        # Elements belonging to this patch
                        patch_elements = is_mat_patch == patch_id
                        patch_indices = torch.nonzero(
                            patch_elements, as_tuple=False).squeeze(-1)
                        if patch_indices.numel() > 0:
                            # Call surrogate integration for this patch
                            k_surr, f_surr = (
                                self.surrogate_integrate_material(
                                    model, u, n, du,
                                    is_stateful_mode=is_stateful_mode,
                                    is_converged=False))
                            # Assemble 
                            k[patch_indices] = k_surr[patch_indices]
                            f_i[patch_indices] = f_surr[patch_indices]
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
            # Final step to update converged hidden states (stateful mode)
            if is_stateful_mode:
                patch_mask = is_mat_patch > 0
                if torch.any(patch_mask):
                    # Get unique patch IDs  
                    patch_ids = torch.unique(is_mat_patch[patch_mask])
                    # Loop over patches
                    for patch_id in patch_ids:
                        # Elements belonging to this patch
                        patch_elements = is_mat_patch == patch_id
                        patch_indices = torch.nonzero(
                            patch_elements, as_tuple=False).squeeze(-1)
                        if patch_indices.numel() > 0:
                            # Final converged call to update hidden states
                            _, _ = self.surrogate_integrate_material(
                                model, u, n, du,
                                is_stateful_mode=is_stateful_mode,
                                is_converged=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update increment
            f[n] = F_int.reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element volumes if requested
            if return_volumes:
                volumes[n] = self.integrate_field()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Disable stateful mode after computation
        if is_stateful_mode:
            model._restore_time_series_attrs(orig_attrs)
            model.set_stateful_mode(False)
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
        if return_intermediate:
            # Return all intermediate values
            if return_volumes:
                return u, f, stress, defgrad, state, volumes
            else:
                return u, f, stress, defgrad, state
        else:
            # Return only the final values
            if return_volumes:
                return u[-1], f[-1], stress[-1], defgrad[-1], state[-1], \
                    volumes[-1]
            else:
                return u[-1], f[-1], stress[-1], defgrad[-1], state[-1]
