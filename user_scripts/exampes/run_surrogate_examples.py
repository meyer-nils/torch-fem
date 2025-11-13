import torch
import os
import sys
import pathlib
import pickle as pkl
# Add graphorge to sys.path
graphorge_path = str(pathlib.Path(__file__).parents[2] \
                     / "graphorge_material_patches" / "src")
if graphorge_path not in sys.path:
    sys.path.insert(0, graphorge_path)

import meshzoo
import numpy as np

from torchfem import Planar

from torchfem.examples import get_example_file
from torchfem.io import import_mesh
from torchfem.materials import IsotropicElasticityPlaneStress


import matplotlib.pyplot as plt


output_dir = ( f"results/examples/cantilever")
os.makedirs(output_dir, exist_ok=True)


# Set double precision
torch.set_default_dtype(torch.float32)

# Material model (plane stress)
material = IsotropicElasticityPlaneStress(E=110000.0, nu=0.33)

# Mesh
points, cells = meshzoo.rectangle_quad(
    np.linspace(0.0, 2.0, 13),
    np.linspace(0.0, 1.0, 7),
    cell_type="quad4",
)
nodes = torch.tensor(points, dtype=torch.get_default_dtype())
elements = torch.tensor(cells.tolist())

# Create model
cantilever = Planar(nodes, elements, material)

# Load at tip
tip = (nodes[:, 0] == 2.0) & (nodes[:, 1] == 0.5)
cantilever.forces[tip, 1] = -1.0

# Constrained displacement at left end
left = nodes[:, 0] == 0.0
cantilever.constraints[left, :] = True

# Thickness
cantilever.thickness[:] = 0.1

# Define material patch flag - use surrogate for all elements
num_elements = cantilever.elements.shape[0]
# All elements use material patch ID 1
is_mat_patch = torch.ones(num_elements, dtype=torch.int)


u, f, σ, F, α = cantilever.solve_matpatch( is_mat_patch=is_mat_patch,
        increments=torch.tensor([0.0, 1.0]),
        max_iter=100,
        rtol=1e-4,
        verbose=True,
        return_intermediate=True,
        return_volumes=False)


f_final = f[-1]  

cantilever.plot(
        u=f_final,
        node_property=torch.sqrt(f_final[:, 0]**2 + f_final[:, 1]**2),
        title='Displacement Magnitude',
        colorbar=True,
        figsize=(8, 6),
        cmap='viridis',
        node_markers=True
    )

plot_path = os.path.join(output_dir, "displacement_field.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()