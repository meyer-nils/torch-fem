[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-fem)
[![PyPI - Version](https://img.shields.io/pypi/v/torch-fem)](https://pypi.org/project/torch-fem/)
[![Tests](https://github.com/meyer-nils/torch-fem/actions/workflows/python-package.yml/badge.svg)](https://github.com/meyer-nils/torch-fem/actions/workflows/python-package.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/meyer-nils/torch-fem/HEAD)
[![DOI](https://zenodo.org/badge/651011204.svg)](https://doi.org/10.5281/zenodo.20306384)

<p align="center">
  <a href="https://meyer-nils.github.io/torch-fem"><b>Documentation</b></a> ·
  <a href="https://meyer-nils.github.io/torch-fem/examples/"><b>Examples</b></a> ·
  <a href="https://github.com/meyer-nils/torch-fem/blob/main/CHANGELOG.md"><b>Changelog</b></a>
</p>

# torch-fem

*torch-fem* is a simple GPU-accelerated differentiable finite element solver for solid mechanics built on PyTorch. Automatic differentiation provides exact sensitivities of simulation results with respect to material parameters, geometry, loads, etc. without hand-derived adjoint formulations. It is aimed at researchers in computational mechanics who need gradients through FEM solvers for tasks such as optimization, inverse problems, and machine-learning-augmented simulation.

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
  - Composite laminates for shells
  - Simple structured meshing
  - I/O to and from other mesh formats via meshio

## Installation
You may install *torch-fem* via pip with

```
pip install torch-fem
```

To run the example notebooks, install with the `notebook` extra (`pip install torch-fem[notebook]`). For GPU acceleration, install PyTorch with CUDA support and the matching CuPy version - see the [installation guide](https://meyer-nils.github.io/torch-fem/installation/) for details.

## Minimal example
This is a minimal example of how to use *torch-fem* to solve a very simple planar cantilever problem. 

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

If we want to compute gradients through the FEM model, we simply need to define the variables that require gradients. Automatic differentiation is performed through the entire FE solver. Rather than differentiating through individual solver iterations or Newton iterations (this would explode in memory and autograd graph size) though, the *implicit function theorem* is used to formulate an adjoint backward for `solve()`.
```python 
# Enable automatic differentiation
cantilever.thickness.requires_grad = True
u, f, _, _, _ = cantilever.solve(differentiable_parameters=cantilever.thickness)

# Compute sensitivity of compliance w.r.t. element thicknesses
compliance = torch.inner(f.ravel(), u.ravel())
torch.autograd.grad(compliance, cantilever.thickness)[0]
```

## Basic examples
The subdirectory `examples/basic` contains a couple of Jupyter notebooks demonstrating the use of *torch-fem* for trusses, planar problems, shells, and solids. You may click on the examples to check out the notebooks online.

<table>
    <tbody>
        <tr>
            <td><a href="https://meyer-nils.github.io/torch-fem/examples/basic/planar/plasticity.html"><img src="https://meyer-nils.github.io/torch-fem/images/plate_hole_plasticity.png" alt="Planar plate with a hole plasticity example"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Plasticity in a plate with hole:</b> Isotropic linear hardening model for plane-stress or plane-strain.</td>
        </tr>
        <tr>
            <td><a href="https://meyer-nils.github.io/torch-fem/examples/basic/solid/finite_strain.html"><img src="https://meyer-nils.github.io/torch-fem/images/cantilever_finite_strain.png" alt="Finite-strain cantilever example"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Finite strain cantilever:</b> Hyperelastic model in Total Lagrangian Formulation.</td>
        </tr>
    </tbody>
</table>

## Optimization examples
The subdirectory `examples/optimization` demonstrates the use of *torch-fem* for optimization of structures (e.g. topology optimization, composite orientation optimization). You may click on the examples to check out the notebooks online.

<table>
    <tbody>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/truss/shape.html"><img src="https://meyer-nils.github.io/torch-fem/images/bridge.png" alt="Truss shape optimization example"></a></td>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/shape.html"><img src="https://meyer-nils.github.io/torch-fem/images/fillet_shape_optimization.png" alt="Planar fillet shape optimization example"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Shape optimization of a truss:</b> The top nodes are moved and MMA + autograd is used to minimize the compliance.</td>
            <td align="center"><b>Shape optimization of a fillet:</b> The shape is morphed with shape basis vectors and MMA + autograd is used to minimize the maximum stress.</td>
        </tr>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/topology.html"><img src="https://meyer-nils.github.io/torch-fem/images/topopt_mbb.png" alt="MBB beam topology optimization example"></a></td>
            <td style="width: 50%;"><img src="https://meyer-nils.github.io/torch-fem/images/topopt_3d.png" alt="3D jet engine bracket topology optimization result"></td>
        </tr>
        <tr>
            <td align="center"><b>Topology optimization of a MBB beam:</b> You can switch between analytical and autograd sensitivities.</td>
            <td align="center"><b>Topology optimization of a jet engine bracket:</b> The 3D model is exported to Paraview for visualization.</td>
        </tr>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/topology+orientation.html"><img src="https://meyer-nils.github.io/torch-fem/images/topo+ori.png" alt="Combined topology and orientation optimization example"></a></td>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/orientation.html"><img src="https://meyer-nils.github.io/torch-fem/images/plate_hole_shape_optimization.png" alt="Fiber orientation optimization example"></a>
            </td>
        </tr>
        <tr>
            <td align="center"><b>Combined topology and orientation optimization:</b> Compliance is minimized by optimizing fiber orientation and density of an anisotropic material using automatic differentiation.</td>
            <td align="center"><b>Fiber orientation optimization of a plate with a hole</b> Compliance is minimized by optimizing the fiber orientation of an anisotropic material using automatic differentiation w.r.t. element-wise fiber angles.</td>
        </tr>
    </tbody>
</table>

## Performance 
*torch-fem* solves problems with millions of degrees of freedom: a linear elastic hexahedral cube model with 1.5 million DOFs assembles and solves in about four seconds on a consumer GPU (RTX 4090, float64). Detailed CPU and GPU benchmarks for timing and memory are reported in the [performance documentation](https://meyer-nils.github.io/torch-fem/performance/) and can be reproduced with the scripts in `benchmarks/`.

## Citing torch-fem
If you use torch-fem in your research, please cite it as follows:

```bibtex
@software{torchfem,
    author = {Meyer, Nils},
    title  = {torch-fem: GPU accelerated differentiable finite elements for solid mechanics with PyTorch},
    doi    = {10.5281/zenodo.20306384},
    url    = {https://github.com/meyer-nils/torch-fem},
}
```

## Contributing
Contributions are welcome! Please check out the [contributing guide](https://github.com/meyer-nils/torch-fem/blob/main/CONTRIBUTING.md) for the development workflow. Bug reports, feature requests, and usage questions are all welcome in the [issue tracker](https://github.com/meyer-nils/torch-fem/issues) - see the [support guide](https://github.com/meyer-nils/torch-fem/blob/main/SUPPORT.md) for what to include.

## Alternatives
There are many alternative Python FEM tools that you may also consider, 
depending on your needs:

- General-purpose FEM/PDE frameworks
    - [FEniCSx (DOLFINx)](https://github.com/FEniCS/dolfinx) ![GitHub stars](https://img.shields.io/github/stars/FEniCS/dolfinx?style=flat-square)
    - [SfePy](https://github.com/sfepy/sfepy) ![GitHub stars](https://img.shields.io/github/stars/sfepy/sfepy?style=flat-square)
    - [scikit-fem](https://github.com/kinnala/scikit-fem) ![GitHub stars](https://img.shields.io/github/stars/kinnala/scikit-fem?style=flat-square)
    - [FElupe](https://github.com/adtzlr/felupe) ![GitHub stars](https://img.shields.io/github/stars/adtzlr/felupe?style=flat-square)
    - [Nutils](https://github.com/evalf/nutils) ![GitHub stars](https://img.shields.io/github/stars/evalf/nutils?style=flat-square) 
- Differentiable or adjoint-capable FEM
    - [JAX-FEM](https://github.com/deepmodeling/jax-fem) ![GitHub stars](https://img.shields.io/github/stars/deepmodeling/jax-fem?style=flat-square)
    - [dolfin-adjoint](https://github.com/dolfin-adjoint/pyadjoint) ![GitHub stars](https://img.shields.io/github/stars/dolfin-adjoint/pyadjoint?style=flat-square)
    - [PyTorch-FEA](https://github.com/liangbright/pytorch_fea) ![GitHub stars](https://img.shields.io/github/stars/liangbright/pytorch_fea?style=flat-square)
