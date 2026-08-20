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
  - Assembly of several models coupled by kinematic constraints
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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://meyer-nils.github.io/torch-fem/images/minimal_example/minimal_example_dark.png">
  <img alt="minimal" src="https://meyer-nils.github.io/torch-fem/images/minimal_example/minimal_example_light.png">
</picture>

```python
# Solve
u, f, σ, F, α = cantilever.solve()

# Plot displacement magnitude on deformed state
cantilever.plot(u, node_property=torch.norm(u, dim=1))
```
This solves the model and plots the result:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://meyer-nils.github.io/torch-fem/images/minimal_example/minimal_example_solved_dark.png">
  <img alt="minimal" src="https://meyer-nils.github.io/torch-fem/images/minimal_example/minimal_example_solved_light.png">
</picture>

If we want to compute gradients through the FEM model, we simply need to define the variables that require gradients. Automatic differentiation is performed through the entire FE solver. Rather than differentiating through individual solver iterations or Newton iterations (this would explode in memory and autograd graph size) though, the *implicit function theorem* is used to formulate an adjoint backward for `solve()`.
```python 
# Enable automatic differentiation
cantilever.thickness.requires_grad = True
u, f, _, _, _ = cantilever.solve(differentiable_parameters=cantilever.thickness)

# Compute sensitivity of compliance w.r.t. element thicknesses
compliance = torch.inner(f.ravel(), u.ravel())
torch.autograd.grad(compliance, cantilever.thickness)[0]
```
This returns the sensitivity of the compliance with respect to the thickness of each element:
```
tensor([-0.0208, -0.0053])
```
Both entries are negative, so adding material anywhere stiffens the structure, but the element at the clamped end is about four times as effective as the one at the tip.

## Basic examples
The subdirectory `examples/basic` contains a couple of Jupyter notebooks demonstrating the use of *torch-fem* for trusses, planar problems, shells, and solids. You may click on the examples to check out the notebooks online.

<table>
    <tbody>
        <tr>
            <td colspan="2"><a href="https://meyer-nils.github.io/torch-fem/examples/basic/planar/plasticity.html"><img src="https://meyer-nils.github.io/torch-fem/images/plate_hole_plasticity_light.png" alt="Planar plate with a hole plasticity example"></a></td>
        </tr>
        <tr>
            <td colspan="2" align="center"><b>Plasticity in a plate with hole:</b> Isotropic linear hardening model for plane-stress or plane-strain.</td>
        </tr>
        <tr>
            <td colspan="2"><a href="https://meyer-nils.github.io/torch-fem/examples/basic/solid/finite_strain.html"><img src="https://meyer-nils.github.io/torch-fem/images/cantilever_finite_strain.png" alt="Finite-strain cantilever example"></a></td>
        </tr>
        <tr>
            <td colspan="2" align="center"><b>Finite strain cantilever:</b> Hyperelastic model in Total Lagrangian Formulation.</td>
        </tr>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/basic/shell/modal.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/basic/shell/modal.png" alt="Shell modal analysis example"></a></td>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/basic/solid/gyroid.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/basic/solid/gyroid.png" alt="Implicit gyroid structure example"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Modal analysis of a clamped shell:</b> Natural frequencies and mode shapes of a fully clamped flat shell.</td>
            <td align="center"><b>Implicit gyroid structure:</b> A voxel mesh is carved into a triply periodic minimal surface with a signed distance function.</td>
        </tr>
    </tbody>
</table>

## Optimization examples
The subdirectory `examples/optimization` demonstrates the use of *torch-fem* for optimization of structures (e.g. topology optimization, composite orientation optimization). You may click on the examples to check out the notebooks online.

<table>
    <tbody>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/truss/shape.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/optimization/truss/shape.png" alt="Truss shape optimization example"></a></td>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/shape.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/optimization/planar/shape.png" alt="Planar fillet shape optimization example"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Shape optimization of a truss:</b> The top nodes are moved and MMA + autograd is used to minimize the compliance.</td>
            <td align="center"><b>Shape optimization of a fillet:</b> The shape is morphed with shape basis vectors and MMA + autograd is used to minimize the maximum stress.</td>
        </tr>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/solid/bracket.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/optimization/solid/bracket.png" alt="3D jet engine bracket topology optimization result"></a></td>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/topology+orientation.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/optimization/planar/topology+orientation.png" alt="Combined topology and orientation optimization example"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Topology optimization of a jet engine bracket:</b> The optimized part is cut out of the design space at an iso-value of the density.</td>
            <td align="center"><b>Combined topology and orientation optimization:</b> Compliance is minimized by optimizing fiber orientation and density of an anisotropic material.</td>
        </tr>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/orientation.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/optimization/planar/orientation.png" alt="Fiber orientation optimization example"></a></td>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/solid/topology_thermal.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/optimization/solid/topology_thermal.png" alt="3D heat sink topology optimization example"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Fiber orientation optimization of a plate with a hole</b> Compliance is minimized by optimizing the fiber orientation of an anisotropic material.</td>
            <td align="center"><b>Topology optimization of a 3D heat sink:</b> Conductive material is distributed in a cube with a homogeneous heat source to minimize thermal compliance.</td>
        </tr>
        <tr>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/planar/property_fields.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/optimization/planar/property_fields.png" alt="Property field recovery example"></a></td>
            <td style="width: 50%;"><a href="https://meyer-nils.github.io/torch-fem/examples/optimization/shell/pressure_vessel.html"><img src="https://meyer-nils.github.io/torch-fem/images/examples/optimization/shell/pressure_vessel.png" alt="Pressure vessel free size optimization example"></a></td>
        </tr>
        <tr>
            <td align="center"><b>Recovery of a property field:</b> A direct optimization and a neural field recover a graded elastic modulus from noisy observations of the displacement.</td>
            <td align="center"><b>Free size optimization of a pressure vessel:</b> Each element's shell thickness is a design variable and a fixed amount of material is redistributed to minimize compliance.</td>
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
*torch-fem* focuses on solid mechanics and thermal problems. It provides sensitivities through PyTorch autograd, which makes it easy to drop into optimization loops and ML pipelines. It is the natural choice if you are working in the PyTorch ecosystem. Depending on your needs, one of these Python FEM tools may serve you better:

| Library | Stars | Focus | Differentiable | Consider it over torch-fem when… |
|---|:---:|---|:---:|---|
| [FEniCSx (DOLFINx)](https://github.com/FEniCS/dolfinx) | ![stars](https://img.shields.io/github/stars/FEniCS/dolfinx?style=flat-square) | General PDEs, UFL weak forms, MPI | via [dolfin-adjoint](https://github.com/dolfin-adjoint/pyadjoint) | you need arbitrary weak forms or massively parallel distributed runs |
| [SfePy](https://github.com/sfepy/sfepy) | ![stars](https://img.shields.io/github/stars/sfepy/sfepy?style=flat-square) | General multiphysics, pure Python | — | you need a broad range of PDE applications on CPU |
| [JAX-FEM](https://github.com/deepmodeling/jax-fem) | ![stars](https://img.shields.io/github/stars/deepmodeling/jax-fem?style=flat-square) | Differentiable FEM, JAX / GPU | ✅ | your stack is built on JAX rather than PyTorch |
| [Firedrake](https://github.com/firedrakeproject/firedrake) | ![stars](https://img.shields.io/github/stars/firedrakeproject/firedrake?style=flat-square) | General PDEs, UFL weak forms | via [pyadjoint](https://github.com/dolfin-adjoint/pyadjoint) | you want a UFL form language with automated adjoints for multiphysics |
| [scikit-fem](https://github.com/kinnala/scikit-fem) | ![stars](https://img.shields.io/github/stars/kinnala/scikit-fem?style=flat-square) | Lightweight assembly, NumPy/SciPy | — | you want minimal dependencies and full control over custom forms |
| [FElupe](https://github.com/adtzlr/felupe) | ![stars](https://img.shields.io/github/stars/adtzlr/felupe?style=flat-square) | Finite-strain solid mechanics | partially via [tensortrax](https://github.com/adtzlr/tensortrax) | you work with hyperelastic / finite-strain solids |
| [Nutils](https://github.com/evalf/nutils) | ![stars](https://img.shields.io/github/stars/evalf/nutils?style=flat-square) | High-order / immersed methods | — | you research advanced or immersed discretizations including IGA |
| [PyTorch-FEA](https://github.com/liangbright/pytorch_fea) | ![stars](https://img.shields.io/github/stars/liangbright/pytorch_fea?style=flat-square) | Biomechanics, PyTorch | ✅ | you work on soft-tissue / inverse biomechanics |

Not sure which to pick? The [mosaic](https://github.com/pasteurlabs/mosaic) differentiable-physics benchmark suite compares several of these solvers on gradient accuracy and forward/adjoint performance under a common interface.
