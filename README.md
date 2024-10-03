[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-fem)
[![PyPI - Version](https://img.shields.io/pypi/v/torch-fem)](https://pypi.org/project/torch-fem/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/meyer-nils/torch-fem/HEAD)



# torch-fem: differentiable linear elastic finite elements

Simple finite element assemblers for linear elasticity with PyTorch. The advantage of using PyTorch is the ability to efficiently compute sensitivities and use them in structural optimization. 

## Basic examples
The subdirectory `examples->basic` contains a couple of Jupyter Notebooks demonstrating the use of torch-fem for trusses, planar problems, shells and solids. 

<img src="doc/cantilever_tria2.png" width="400"></br>
**Simple cantilever beam:** There are examples with linear and quadratic triangles and quads.

## Optimization examples
The subdirectory `examples->optimization` demonstrates the use of torch-fem for optimization of structures (e.g. topology optimization, composite orientation optimization).

<img src="doc/topopt_mbb.png" width="400"></br>
**Simple topology optimization of a MBB beam:** You can switch between analytical sensitivities and autograd sensitivities.

<img src="doc/topopt_3d.png" width="400"></br>
**Simple topology optimization of a 3D beam:** The model is exported to Paraview for visualization.

<img src="doc/fillet_shape_optimization.png" width="400"></br>
**Simple shape optimization of a fillet:** The shape is morphed with shape basis vectors and MMA + autograd is used to minimize the maximum stress.

<img src="doc/plate_hole_shape_optimization.png" width="400"></br>
**Simple fiber orientation optimization of a plate with a hole:** Compliance is minimized by optimizing the fiber orientation of an anisotropic material using automatic differentiation w.r.t. element-wise fiber angles.

## Installation
Your may install torch-fem via pip with
```
pip install torch-fem
```

## Minimal code
This is a minimal example of how to use torch-fem to solve a simple cantilever problem. 

```python
from torchfem import Planar
from torchfem.materials import IsotropicPlaneStress

# Define a (minimal) mesh 
nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
elements = torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]])

# Apply a load at the tip
tip = (nodes[:, 0] == 2.0) & (nodes[:, 1] == 1.0)
forces = torch.zeros_like(nodes)
forces[tip, 1] = -1.0

# Constrained displacement at left end
left = nodes[:, 0] == 0.0
displacements = torch.zeros_like(nodes)
constraints = torch.zeros_like(nodes, dtype=bool)
constraints[left, :] = True

# Thickness
thickness = torch.ones(len(elements))

# Material model (plane stress)
material = IsotropicPlaneStress(E=1000.0, nu=0.3)

# Create model
cantilever = Planar(nodes, elements, forces, displacements, constraints, thickness, material.C())
```
This creates a minimal planar FEM model:

![minimal](doc/minimal_example.png)

```python
# Solve
u, f = cantilever.solve()

# Plot
cantilever.plot(u, node_property=torch.norm(u, dim=1), node_markers=True)
```
This solves the model and plots the result:

![minimal](doc/minimal_example_solved.png)

## Benchmarks 
The following benchmarks were performed on a cube subjected to a one dimensional extension. The cube is discretized with N x N x N linear hexahedral elements, has a side length of 1.0 and is made of a material with Young's modulus of 1000.0 and Poisson's ratio of 0.3. The cube is fixed at one end and a displacement of 0.1 is applied at the other end. The benchmark measures the forward time to assemble the stiffness matrix and the time to solve the linear system. In addition, it measures the backward time to compute the sensitivities of the sum of the displacements with respect to the forces.

#### Apple M1 Pro (10 cores, 16 GB RAM)
Python 3.10 with Apple Accelerate

|  N  |    DOFs | FWD Time |  FWD Memory | BWD Time |  BWD Memory |
| --- | ------- | -------- | ----------- | -------- | ----------- |
|  10 |    3000 |    0.76s |    14.08 MB |    0.65s |     2.69 MB |
|  20 |   24000 |    3.53s |   474.75 MB |    2.95s |   227.78 MB |
|  30 |   81000 |   70.67s |  2774.97 MB |   69.70s |  1998.42 MB |
|  40 |  192000 |  637.74s |  8962.11 MB |  626.16s |  8985.50 MB |