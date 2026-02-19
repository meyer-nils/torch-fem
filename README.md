[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-fem)
[![PyPI - Version](https://img.shields.io/pypi/v/torch-fem)](https://pypi.org/project/torch-fem/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/meyer-nils/torch-fem/HEAD)


# torch-fem
Simple GPU accelerated differentiable finite elements for solid mechanics with PyTorch. 
PyTorch enables efficient computation of sensitivities via automatic differentiation and using them in optimization tasks.

## Installation
Your may install torch-fem via pip with
```
pip install torch-fem
```

*Optional*: For GPU support, install CUDA, PyTorch for CUDA, and the corresponding CuPy version.

For CUDA 11.8: 
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cupy-cuda11x # v11.2 - 11.8
```

For CUDA 12.6:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install cupy-cuda12x # v12.x
```

## Features
- Elements
  - 1D: Bar1, Bar2
  - 2D: Quad1, Quad2, Tria1, Tria2
  - 3D: Hexa1, Hexa2, Tetra1, Tetra2
  - Shell: Flat-facet triangle (linear only)
- Material models
  - Isotropic linear elasticity 
  - Orthotropic linear elasticity
  - Isotropic small strain plasticity
  - Isotropic small strain damage
  - Hyperelasticity (via automatic differentiation of their energy function)
  - Isotropic thermal conductivity
  - Orthotropic thermal conductivity
  - Custom user material interface

- Utilities
  - Homogenization of orthotropic elasticity for composites
  - Simple structured meshing
  - I/O to and from other mesh formats via meshio

## Basic examples
The subdirectory `examples->basic` contains a couple of Jupyter Notebooks demonstrating the use of torch-fem for trusses, planar problems, shells and solids. You may click on the examples to check out the notebooks online.

<table>
    <tbody>
        <tr>
            <td style="width: 30%;"><a href="https://meyer-nils.github.io/torch-fem/examples/basic/solid/gyroid.html"><img src="https://meyer-nils.github.io/torch-fem/images/gyroid.png"></a></td>
            <td style="width: 30%;"><a href="https://meyer-nils.github.io/torch-fem/examples/basic/solid/cubes.html"><img src="https://meyer-nils.github.io/torch-fem/images/cubes.png"></a></td>
            <td style="width: 30%;"><a href="https://meyer-nils.github.io/torch-fem/examples/basic/planar/cantilever.html"><img src="https://meyer-nils.github.io/torch-fem/images/cantilever_tria2.png"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Gyroid:</b> Support for voxel meshes and implicit surfaces.</td>
            <td align="center"><b>Solid cubes:</b> There are several examples with different element types rendered in PyVista.</td>
            <td align="center"><b>Planar cantilever beams:</b> There are several examples with different element types rendered in matplotlib.</td>
        </tr>
        <tr>
            <td colspan="3"><a href="https://meyer-nils.github.io/torch-fem/examples/basic/planar/plasticity.html"><img src="https://meyer-nils.github.io/torch-fem/images/plate_hole_plasticity.png"></a></td>
        </tr>
        <tr>
            <td colspan="3" align="center"><b>Plasticity in a plate with hole:</b> Isotropic linear hardening model for plane-stress or plane-strain.</td>
        </tr>
        <tr>
            <td colspan="3"><a href="https://meyer-nils.github.io/torch-fem/examples/basic/solid/finite_strain.html"><img src="https://meyer-nils.github.io/torch-fem/images/cantilever_finite_strain.png"></a></td>
        </tr>
        <tr>
            <td colspan="3" align="center"><b>Finite strain cantilever:</b> Hyperelastic model in Total Lagrangian Formulation.</td>
        </tr>
    </tbody>
</table>

## Optimization examples
The subdirectory `examples->optimization` demonstrates the use of torch-fem for optimization of structures (e.g. topology optimization, composite orientation optimization). You may click on the examples to check out the notebooks online.

<table>
    <tbody>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/truss/shape.html"><img src="https://meyer-nils.github.io/torch-fem/images/bridge.png"></a></td>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/shape.html"><img src="https://meyer-nils.github.io/torch-fem/images/fillet_shape_optimization.png"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Shape optimization of a truss:</b> The top nodes are moved and MMA + autograd is used to minimize the compliance.</td>
            <td align="center"><b>Shape optimization of a fillet:</b> The shape is morphed with shape basis vectors and MMA + autograd is used to minimize the maximum stress.</td>
        </tr>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/topology.html"><img src="https://meyer-nils.github.io/torch-fem/images/topopt_mbb.png"></a></td>
            <td style="width: 50%;"><img src="https://meyer-nils.github.io/torch-fem/images/topopt_3d.png"></td>
        </tr>
        <tr>
            <td align="center"><b>Topology optimization of a MBB beam:</b> You can switch between analytical and autograd sensitivities.</td>
            <td align="center"><b>Topology optimization of a jet engine bracket:</b> The 3D model is exported to Paraview for visualization.</td>
        </tr>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/topology+orientation.html"><img src="https://meyer-nils.github.io/torch-fem/images/topo+ori.png"></a></td>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/orientation.html"><img src="https://meyer-nils.github.io/torch-fem/images/plate_hole_shape_optimization.png"></a>
            </td>
        </tr>
        <tr>
            <td align="center"><b>Combined topology and orientation optimization:</b> Compliance is minimized by optimizing fiber orientation and density of an anisotropic material using automatic differentiation.</td>
            <td align="center"><b>Fiber orientation optimization of a plate with a hole</b> Compliance is minimized by optimizing the fiber orientation of an anisotropic material using automatic differentiation w.r.t. element-wise fiber angles.</td>
        </tr>
        <tr>
            <td colspan="2"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/topology_thermal_static.html"><img src="https://meyer-nils.github.io/torch-fem/images/topology_thermal_static.png"></a></td>
        </tr>
        <tr>
            <td colspan="2" align="center"><b>Heat sink:</b> Thermal topology optimization</td>
        </tr>
    </tbody>
</table>


## Minimal example
This is a minimal example of how to use torch-fem to solve a very simple planar cantilever problem. 

```python
import torch
from torchfem import Planar
from torchfem.materials import IsotropicElasticityPlaneStress

torch.set_default_dtype(torch.float64)

# Material
material = IsotropicElasticityPlaneStress(E=1000.0, nu=0.3)

# Nodes and elements
nodes = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [1., 1.], [2., 1.]])
elements = torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]])

# Create model
cantilever = Planar(nodes, elements, material)

# Load at tip [Node_ID, DOF]
cantilever.forces[5, 1] = -1.0

# Constrained displacement at left end [Node_IDs, DOFs]
cantilever.constraints[[0, 3], :] = True

# Show model
cantilever.plot(node_markers="o", node_labels=True)
```
This creates a minimal planar FEM model:

![minimal](https://meyer-nils.github.io/torch-fem/images/minimal_example.png)

```python
# Solve
u, f, σ, F, α = cantilever.solve()

# Plot displacement magnitude on deformed state
cantilever.plot(u, node_property=torch.norm(u, dim=1))
```
This solves the model and plots the result:

![minimal](https://meyer-nils.github.io/torch-fem/images/minimal_example_solved.png)

If we want to compute gradients through the FEM model, we simply need to define the variables that require gradients. Automatic differentiation is performed through the entire FE solver.
```python 
# Enable automatic differentiation
cantilever.thickness.requires_grad = True
u, f, _, _, _ = cantilever.solve()

# Compute sensitivity of compliance w.r.t. element thicknesses
compliance = torch.inner(f.ravel(), u.ravel())
torch.autograd.grad(compliance, cantilever.thickness)[0]
```

## Benchmarks 
The following benchmarks were performed on a cube subjected to a one-dimensional extension. The cube is discretized with N x N x N linear hexahedral elements, has a side length of 1.0 and is made of a material with Young's modulus of 1000.0 and Poisson's ratio of 0.3. The cube is fixed at one end and a displacement of 0.1 is applied at the other end. The benchmark measures the forward time to assemble the stiffness matrix and the time to solve the linear system. In addition, it measures the backward time to compute the sensitivities of the sum of displacements with respect to forces.

#### Apple M1 Pro (10 cores, 16 GB RAM)
Python 3.10, SciPy 1.14.1, Apple Accelerate, float64

|  N  |     DOFs |     Setup | FWD Solve | BWD Solve |   Peak RAM |
| --- | -------- | --------- | --------- | --------- | ---------- |
|  10 |     3000 |     0.02s |     0.18s |     0.14s |    448.2MB |
|  20 |    24000 |     0.15s |     0.78s |     0.21s |    847.1MB |
|  30 |    81000 |     0.56s |     2.86s |     0.67s |   1979.2MB |
|  40 |   192000 |     1.31s |     6.94s |     1.17s |   2988.7MB |
|  50 |   375000 |     2.69s |    15.36s |     2.59s |   4011.4MB |
|  60 |   648000 |     5.31s |    26.63s |     4.06s |   5760.9MB |
|  70 |  1029000 |     8.93s |    45.27s |     6.81s |   7885.4MB |
|  80 |  1536000 |    14.28s |    81.66s |    12.60s |   9360.8MB |


#### NVIDIA GeForce RTX 4090 (16,384 Cuda cores, 24 GB VRAM)
Python 3.12, CuPy 13.3.0, CUDA 11.8, float64

|  N  |     DOFs |  FWD Time |  BWD Time |   Peak RAM |
| --- | -------- | --------- | --------- | ---------- |
|  10 |     3000 |     0.68s |     0.17s |   1503.0MB |
|  20 |    24000 |     0.94s |     0.41s |   1495.1MB |
|  30 |    81000 |     1.15s |     0.54s |   1496.3MB |
|  40 |   192000 |     1.46s |     0.73s |   1489.9MB |
|  50 |   375000 |     1.97s |     1.02s |   1505.1MB |
|  60 |   648000 |     2.65s |     1.36s |   1506.7MB |
|  70 |  1029000 |     3.69s |     1.89s |   1496.4MB |

## Alternatives
There are many alternative FEM solvers in Python that you may also consider: 

- Non-differentiable 
  - [scikit-fem](https://github.com/kinnala/scikit-fem)
  - [nutils](https://github.com/evalf/nutils) 
  - [felupe](https://github.com/adtzlr/felupe)
- Differentiable 
  - [jaxfem](https://github.com/deepmodeling/jax-fem)
  - [PyTorch FEA](https://github.com/liangbright/pytorch_fea)