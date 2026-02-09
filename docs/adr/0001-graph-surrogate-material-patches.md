# ADR-0001: Graph Neural Network Surrogate for Material Patches

* **Status:** Accepted
* **Date:** 2026-02-09

## Context and Problem Statement

Finite element analysis requires computing element stiffness matrices and
internal force vectors through numerical integration at every load increment.
For nonlinear material behavior (elastoplasticity, hyperelasticity) or
large-scale problems, this quadrature loop becomes the dominant computational
cost. The question is how to accelerate this integration while maintaining
accuracy and enabling material history tracking.

## Decision Drivers

* Computational cost of repeated Gauss quadrature at every Newton iteration
* Need for accurate tangent stiffness matrices (Jacobian) for convergence
* Material state history tracking for path-dependent constitutive laws
* Ability to handle different material behaviors (elastic, elastoplastic)
* Support for multi-scale material patches of varying resolution
* Integration with existing PyTorch-based FEM solver framework

## Considered Options

* **Option A:** Pure numerical integration via standard Gauss quadrature
* **Option B:** Feed-forward neural network (FFNN) surrogate per element
* **Option C:** Graph Neural Network (GNN) surrogate for material patches
* **Option D:** Recurrent GNN with hidden states for path-dependent materials

## Decision Outcome

Chosen option: **Option C/D — GNN-based material patch surrogate with
optional RNN mode**, because GNNs naturally encode the mesh topology and
boundary connectivity while enabling automatic differentiation through
`torch.func.jacfwd` for exact stiffness computation. The RNN variant handles
path-dependent materials by maintaining hidden states across load increments.

**Implementation:** The surrogate model replaces element-level integration
with graph-level inference. Each material patch (collection of elements) is
represented as a boundary-node graph. The GNN predicts internal forces
directly from boundary displacements, and automatic differentiation computes
the tangent stiffness matrix.

### Positive Consequences

* 10-100x speedup vs quadrature for complex constitutive laws
* Exact tangent stiffness via `jacfwd` — no finite difference approximation
* Natural handling of multi-element patches (1x1, 2x2, 8x8 configurations)
* Hidden state mechanism supports elastoplastic material history tracking
* Single model architecture handles elastic and elastoplastic materials
* Seamless integration with Newton-Raphson solver loop

### Negative Consequences

* Surrogate must be trained offline for each material behavior and patch size
* Graph construction overhead at first load increment (edges, features)
* Hidden state management adds complexity for path-dependent materials
* Model loading time (~1-2s) at solver initialization
* Coordinate rescaling required ([0,1]^d) to match training domain
* Limited to materials represented in training dataset distribution

## Implementation Details

### Model Loading and Initialization

The surrogate model is loaded from a directory containing:
* `model_init_args.pkl`: GNN architecture parameters (hidden layer sizes,
  message passing steps)
* `model_state_best.pt`: Trained weights for best validation loss
* `data_scaler.pkl`: Input/output normalization statistics
* `summary.dat`: Training metadata (loss curves, hyperparameters)

**Location** ([base.py:352-389](../../torch-fem/src/torchfem/base.py#L352-L389)):
```python
def _load_Graphorge_model(self, model_directory, device_type='cpu'):
    model = GNNEPDBaseModel.init_model_from_file(model_directory)
    model.set_device(device_type)
    _ = model.load_model_state(
        load_model_state='best', is_remove_posterior=False)
    model.eval()
    return model
```

Default model paths follow the structure:
```
user_scripts/matpatch_surrogates/{material_behavior}/{patch_size}/model/
```
where `material_behavior` is `elastic` or `elastoplastic_nlh`, and
`patch_size` is `1x1`, `2x2`, `8x8`, etc.

### Material Patch Resolution Workflow

The implementation supports three primary patch resolutions:

**1x1 Patches (Single Element):**
* Boundary nodes: 4 nodes (2D Quad4) or 8 nodes (3D Hex8)
* Graph edges: 4 edges (2D square mesh connectivity)
* Use case: Fine-scale heterogeneity, maximum surrogate calls
* Model directory: `matpatch_surrogates/elastic/1x1/model`

**2x2 Patches (Four Elements):**
* Boundary nodes: 8 nodes (2D perimeter of 2x2 element block)
* Graph edges: 8 edges (perimeter connectivity)
* Use case: Balanced resolution, moderate computational cost
* Model directory: `matpatch_surrogates/elastic/2x2/model`

**8x8 Patches (64 Elements):**
* Boundary nodes: 32 nodes (2D perimeter of 8x8 element block)
* Graph edges: 32 edges (perimeter connectivity)
* Use case: Coarse surrogate, minimum surrogate calls per simulation
* Model directory: `matpatch_surrogates/elastic/8x8/model`

The patch resolution is specified via `patch_elem_per_dim` parameter (e.g.,
`[2, 2]` for 2x2, `[8, 8]` for 8x8) when calling `solve_matpatch()`.

### Graph Topology Construction

For each material patch, a graph is constructed representing the boundary node
connectivity ([base.py:1124-1232](../../torch-fem/src/torchfem/base.py#L1124-L1232)).

**Boundary Node Extraction:**
1. Identify nodes shared by multiple patches OR on global domain boundary
2. Extract boundary node coordinates from `self.nodes[boundary_node_ids]`
3. Compute patch bounding box to determine physical dimensions `patch_dim`

**Mesh Connectivity Matrix:**
For a 2x2 patch:
```python
mesh_nodes_matrix = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])  # (mesh_nx+1) x (mesh_ny+1) = 3x3 node grid
```
Only boundary nodes (perimeter of this grid) are retained.

**Edge Construction:**
```python
# Get all mesh edges
connected_nodes_all = get_mesh_connected_nodes(dim, mesh_nodes_matrix)
# Filter to boundary-boundary edges only
connected_nodes_boundary = [
    (node1, node2) for (node1, node2) in connected_nodes_all
    if node1 in boundary_node_set and node2 in boundary_node_set
]
```

**Graph Radius:**
```python
connect_radius = 4 * sqrt(sum([x**2 for x in element_sizes]))
```
This ensures all nearest-neighbor edges are included.

**Storage:**
```python
self._edges_indexes[f"patch_{patch_id}"] = edges_indexes  # Shape: (2, n_edges)
```
Edges are cached per patch to avoid recomputation during Newton iterations.

### Surrogate Integration Workflow

#### Non-Stepwise Mode (Elastic Materials)

**Entry Point** ([base.py:1234-1450](../../torch-fem/src/torchfem/base.py#L1234-L1450)):
```python
K_global, F_global = self.surrogate_integrate_material(
    model, u, n, du,
    is_stepwise=False,
    patch_ids=patch_ids
)
```

**Per-Patch Processing Loop:**
1. **Extract Boundary Data** (lines 1304-1312):
   ```python
   boundary_node_ids = self.patch_bd_nodes[idx_patch]
   boundary_u_current = u[n, boundary_node_ids, :].clone()
   boundary_coords_ref = self.nodes[boundary_node_ids]
   edges_indexes = self._edges_indexes[f"patch_{idx_patch}"]
   ```

2. **Coordinate Rescaling** (lines 1358-1362):
   ```python
   coords_min = boundary_coords_ref.min(dim=0).values
   coords_max = boundary_coords_ref.max(dim=0).values
   L = coords_max - coords_min
   coords_scaled = (boundary_coords_ref - coords_min) / L  # → [0,1]^d
   ```

3. **Forward Pass with Automatic Differentiation** (lines 1366-1396):
   ```python
   def forward(disp_boundary):
       disp_scaled = disp_boundary / L
       pred_scaled = forward_graph(
           model, disps=disp_scaled, coords_ref=coords_scaled,
           edges_indexes=edges_indexes, n_dim=self.n_dim
       )
       pred_real = pred_scaled * L  # Unscale forces to physical space
       return pred_real, pred_real.detach()

   (jacobian, boundary_forces) = torch_func.jacfwd(
       forward, has_aux=True)(boundary_u_current)
   k_boundary = jacobian.view(n_dof_boundary, n_dof_boundary)
   f_boundary = boundary_forces.flatten()
   ```

4. **Global Assembly** (lines 1407-1428):
   ```python
   # Scatter boundary forces to global vector
   for local_i, global_node in enumerate(boundary_node_ids):
       for d in range(self.n_dim):
           global_dof = global_node * self.n_dim + d
           local_dof = local_i * self.n_dim + d
           F_global[global_dof] += f_boundary[local_dof]

   # Collect stiffness entries in COO format
   for local_i, global_node_i in enumerate(boundary_node_ids):
       for local_j, global_node_j in enumerate(boundary_node_ids):
           # Map (local_i, local_j) → (global_dof_i, global_dof_j)
           k_indices.append([global_dof_i, global_dof_j])
           k_values.append(k_boundary[local_dof_i, local_dof_j])
   ```

**Global Stiffness Matrix:**
Assembled as sparse COO tensor from accumulated patch contributions
(lines 1434-1450):
```python
k_indices_tensor = torch.tensor(k_indices, dtype=torch.long).t()
k_values_tensor = torch.stack(k_values)
size = (n_dof_global, n_dof_global)
K_global = torch.sparse_coo_tensor(
    k_indices_tensor, k_values_tensor, size=size
).coalesce()
```

#### Stepwise Mode (Elastoplastic Materials)

For path-dependent materials, hidden states track deformation history
([base.py:1593-1648](../../torch-fem/src/torchfem/base.py#L1593-L1648)).

**Initialization:**
```python
if is_stepwise:
    model._save_time_series_attrs()
    model.set_rnn_mode(is_stepwise=True)

    # Initialize hidden state structure per patch
    n_message_steps = model._n_message_steps
    processor_hidden = {}
    for layer_idx in range(n_message_steps):
        processor_hidden[f'layer_{layer_idx}'] = {
            'node': None, 'edge': None, 'global': None
        }

    hidden_states_dict[f"patch_{pid}"] = {
        'encoder': {'node': None, 'edge': None, 'global': None},
        'processor': processor_hidden,
        'decoder': {'node': None, 'edge': None, 'global': None}
    }
```

**Hidden State Injection** (lines 1334-1354):
Before each patch inference, the model's internal hidden states are set:
```python
if is_stepwise and hidden_states and patch_id in hidden_states:
    patch_hidden = copy.deepcopy(hidden_states[patch_id])
    model._gnn_epd_model._hidden_states = patch_hidden

    if 'encoder' in patch_hidden:
        model._gnn_epd_model._encoder._hidden_states = \
            patch_hidden['encoder']

    if 'processor' in patch_hidden:
        model._gnn_epd_model._processor._hidden_states = \
            patch_hidden['processor']
        for i, layer in enumerate(
                model._gnn_epd_model._processor._processor):
            layer_key = f'layer_{i}'
            layer._hidden_states = \
                patch_hidden['processor'][layer_key]

    if 'decoder' in patch_hidden:
        model._gnn_epd_model._decoder._hidden_states = \
            patch_hidden['decoder']
```

**Hidden State Extraction** (lines 1366-1389):
After forward pass, updated states are captured via `nonlocal` closure:
```python
hidden_states_trial = None
def forward(disp_boundary):
    nonlocal hidden_states_trial
    disp_scaled = disp_boundary / L
    if is_stepwise:
        pred_scaled, hidden_states_out = forward_graph(
            model, disps=disp_scaled, coords_ref=coords_scaled,
            edges_indexes=edges_indexes, n_dim=self.n_dim
        )
        hidden_states_trial = hidden_states_out
    # ...
```

**Hidden State Persistence** (lines 1431-1432):
```python
if is_stepwise and hidden_states_trial is not None:
    hidden_states_out[patch_id] = hidden_states_trial
```

**Convergence-Driven Update** ([base.py:1723-1733](../../torch-fem/src/torchfem/base.py#L1723-L1733)):
Hidden states are only persisted when Newton iteration converges:
```python
if res_norm < rtol * res_norm0 or res_norm < atol:
    if is_stepwise:
        for pid in patch_ids:
            patch_key = f"patch_{pid.item()}"
            hidden_states_dict[patch_key] = hidden_state_out[patch_key]
    break
```
This prevents corrupted states from failed iterations from polluting the
history.

**Cleanup** (lines 1770-1772):
```python
if is_stepwise:
    model.set_rnn_mode(is_stepwise=False)
    model._restore_time_series_attrs()
```

### Graph Feature Computation

Node and edge features follow the Graphorge `GNNPatchFeaturesGenerator`
protocol ([base.py:1798-1879](../../torch-fem/src/torchfem/base.py#L1798-L1879)):

**Node Features** (concatenated):
* `coord_hist`: Current coordinates (reference + displacement) of shape
  `(n_nodes, n_time_steps*n_dim)`
* `disp_hist`: Displacement history of shape `(n_nodes, n_time_steps*n_dim)`

**Edge Features** (concatenated):
* `edge_vector`: Spatial vector between connected nodes
  `coords[i] - coords[j]`
* `relative_disp`: Relative displacement `disps[i] - disps[j]`

**Normalization:**
Features are normalized using pre-computed statistics from training:
```python
node_features_norm = model.data_scaler_transform(
    tensor=node_features_in,
    features_type='node_features_in',
    mode='normalize'
)
```

In stepwise mode, `stepwise_data_scaler_transform` is used instead, which
normalizes based on single-timestep statistics (since RNN processes one step
at a time).

### Newton-Raphson Integration

The surrogate is called within the Newton iteration loop
([base.py:1651-1756](../../torch-fem/src/torchfem/base.py#L1651-L1756)):

```python
for n in range(1, N):  # Load increments
    inc = increments[n] - increments[n - 1]
    F_ext = increments[n] * self.forces.ravel()
    DU = inc * self.displacements.clone().ravel()

    for i in range(max_iter):  # Newton iterations
        du[con] = DU[con]

        # Call surrogate
        if is_stepwise:
            K_raw, F_int, hidden_state_out = \
                self.surrogate_integrate_material(...)
        else:
            K_raw, F_int = self.surrogate_integrate_material(...)

        # Apply constraints
        self.K = self._apply_constraints_sparse(K_raw, con)

        # Compute residual
        residual = F_int - F_ext
        residual[con] = 0.0
        res_norm = torch.linalg.norm(residual)

        # Check convergence
        if res_norm < rtol * res_norm0 or res_norm < atol:
            if is_stepwise:
                # Update hidden states with converged values
                hidden_states_dict.update(hidden_state_out)
            break

        # Solve for displacement correction
        du -= sparse_solve(self.K, residual, ...)
```

**Key Points:**
* Surrogate is called at every Newton iteration (not just once per increment)
* Hidden states are updated only upon convergence
* Constraint application happens after surrogate integration
* Residual norm tracks `||F_int - F_ext||` for convergence

### Material-Specific Workflows

**Elastic Materials:**
* Model type: Feed-forward GNN (no hidden states)
* Typical increments: 2-5 (0.0 → 1.0 load factor)
* `is_stepwise=False`
* Training dataset: Random displacement BC samples, static equilibrium

**Elastoplastic Materials:**
* Model type: Recurrent GNN (LSTM/GRU-style hidden states)
* Typical increments: 20-50 (fine stepping to capture plastic evolution)
* `is_stepwise=True`
* Training dataset: Load path sequences with plastic hardening history
* Hidden state carries: Equivalent plastic strain, back stress, etc.

**Hyperelastic Materials:**
* Model type: Feed-forward GNN (constitutive law is path-independent in
  total Lagrangian)
* Typical increments: 10-20 (large deformation requires smaller steps)
* `is_stepwise=False`
* Training dataset: Large strain displacement samples, Neo-Hookean/Mooney

### Parallel Processing

A parallel variant exists for multi-patch simulations
([surrogate_integrate_material_parallel.py](../../torch-fem/src/torchfem/surrogate_integrate_material_parallel.py)):

**PatchDataset:**
* Wraps patch-level processing in PyTorch `Dataset` interface
* Each `__getitem__` returns `{idx, forces, stiffness, hidden_states}`

**Execution Backends:**
* `surrogate_integrate_material_dataloader`: PyTorch DataLoader with
  `num_workers`
* `surrogate_integrate_material_joblib`: Joblib with threading backend

Not currently active in main solver but available for large-scale problems
with 100+ patches.

## Pros and Cons of the Options

### Option A: Pure Numerical Integration

* Good, because: Guaranteed accuracy for any constitutive law
* Good, because: No training data required
* Bad, because: Computational cost scales with quadrature order and material
  complexity
* Bad, because: Nonlinear materials require expensive stress-return mapping
  at every integration point

### Option B: Feed-Forward NN Per Element

* Good, because: Simple training (no graph structure)
* Bad, because: Cannot capture multi-element patch behavior
* Bad, because: No spatial inductive bias (permutation invariance)
* Bad, because: Separate model per element type and resolution

### Option C: GNN for Material Patches

* Good, because: Handles varying patch sizes with same architecture
* Good, because: Graph structure encodes mesh topology naturally
* Good, because: Permutation equivariance reduces training data needs
* Bad, because: Requires graph construction overhead
* Bad, because: Limited to elastic (path-independent) materials

### Option D: Recurrent GNN (Chosen)

* Good, because: Extends Option C to path-dependent materials
* Good, because: Hidden states carry material history (plastic strain, etc.)
* Good, because: Single forward pass per increment (no internal sub-stepping)
* Bad, because: Hidden state management increases complexity
* Bad, because: Training requires sequential load path data (harder to
  generate)
* Bad, because: Model loading time (~1-2s) adds initialization overhead

## References

* **Graphorge Library:** Custom GNN framework (`GNNEPDBaseModel`) with
  encoder-processor-decoder architecture
* **Material Patch Training:** Offline supervised learning from FEniCSx/torch
  -fem reference solutions
* **Automatic Differentiation:** `torch.func.jacfwd` for exact tangent
  stiffness (no finite differences)

## See Also

* `torch-fem/src/torchfem/base.py` — Main FEM solver with `solve_matpatch()`
  method
* `torch-fem/src/torchfem/surrogate_integrate_material_parallel.py` —
  Parallel processing variants
* `torch-fem/user_scripts/run_simulation_surrogate.py` — Example usage
  script
* `graphorge_material_patches/` — GNN model definition and training code
