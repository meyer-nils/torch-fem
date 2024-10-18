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

# Material
material = IsotropicPlaneStress(E=1000.0, nu=0.3)

# Nodes and elements
nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
elements = torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]])

# Create model
cantilever = Planar(nodes, elements, material)

# Load at tip
cantilever.forces[5, 1] = -1.0

# Constrained displacement at left end
cantilever.constraints[[0, 3], :] = True

# Show model
cantilever.plot(node_markers="o", node_labels=True)
```
This creates a minimal planar FEM model:

![minimal](doc/minimal_example.png)

```python
# Solve
u, f = cantilever.solve()

# Plot
cantilever.plot(u, node_property=torch.norm(u, dim=1))
```
This solves the model and plots the result:

![minimal](doc/minimal_example_solved.png)

If we want to compute gradients through the FEM model, we simply need to define the variables that require gradients. Automatic differentiation is performed through the entire FE solver.
```python 
# Enable automatic differentiation
cantilever.thickness.requires_grad = True
u, f = cantilever.solve()

# Compute sensitivity
compliance = torch.inner(f.ravel(), u.ravel())
torch.autograd.grad(compliance, cantilever.thickness)[0]
```

## Benchmarks 
The following benchmarks were performed on a cube subjected to a one dimensional extension. The cube is discretized with N x N x N linear hexahedral elements, has a side length of 1.0 and is made of a material with Young's modulus of 1000.0 and Poisson's ratio of 0.3. The cube is fixed at one end and a displacement of 0.1 is applied at the other end. The benchmark measures the forward time to assemble the stiffness matrix and the time to solve the linear system. In addition, it measures the backward time to compute the sensitivities of the sum of displacements with respect to forces.

#### Apple M1 Pro (10 cores, 16 GB RAM)
Python 3.10 with Apple Accelerate

|  N  |    DOFs | FWD Time |  FWD Memory | BWD Time |  BWD Memory |
| --- | ------- | -------- | ----------- | -------- | ----------- |
|  10 |    3000 |    0.75s |    23.59 MB |    0.59s |     0.09 MB |
|  20 |   24000 |    3.37s |   439.23 MB |    2.82s |   227.05 MB |
|  30 |   81000 |    3.09s |   728.31 MB |    1.47s |     0.06 MB |
|  40 |  192000 |    7.84s |   807.41 MB |    3.99s |   217.56 MB |
|  50 |  375000 |   16.29s |  1211.27 MB |    9.50s |   433.30 MB |
|  60 |  648000 |   30.78s |  2638.23 MB |   19.17s |  1484.67 MB |
|  70 | 1029000 |   55.14s |  3546.22 MB |   34.41s |  1997.77 MB |
|  80 | 1536000 |   87.83s |  5066.81 MB |   56.23s |  3594.27 MB |
|  90 | 2187000 |  131.09s |  7795.83 MB |  107.40s |  5020.55 MB |


#### AMD Ryzen Threadripper PRO 5995WX (64 Cores, 512 GB RAM)
Python 3.12 with openBLAS

|  N  |    DOFs | FWD Time |   FWD Memory | BWD Time |   BWD Memory |
| --- | ------- | -------- | ------------ | -------- | ------------ |
|  10 |    3000 |    1.42s |     17.27 MB |    1.87s |      0.00 MB |
|  20 |   24000 |    1.30s |    160.49 MB |    0.98s |     62.64 MB |
|  30 |   81000 |    2.76s |    480.52 MB |    2.16s |    305.76 MB |
|  40 |  192000 |    6.68s |   1732.11 MB |    5.15s |    762.89 MB |
|  50 |  375000 |   12.51s |   3030.36 MB |   11.29s |   1044.85 MB |
|  60 |  648000 |   22.94s |   5813.95 MB |   25.54s |   3481.15 MB |
|  70 | 1029000 |   38.81s |   7874.30 MB |   45.06s |   4704.93 MB |
|  80 | 1536000 |   63.07s |  14278.46 MB |   63.70s |   8505.52 MB |
|  90 | 2187000 |   93.47s |  16803.27 MB |  142.94s |  10995.63 MB |

