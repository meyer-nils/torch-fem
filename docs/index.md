---
icon: lucide/binoculars
---

# About

*torch-fem* is a GPU-accelerated differentiable finite element solver for solid mechanics and heat transfer built on PyTorch. Automatic differentiation provides exact sensitivities of simulation results with respect to material parameters, geometry, loads, etc. without hand-derived adjoint formulations. It is aimed at researchers in computational mechanics who need gradients through FEM solvers for tasks such as optimization, inverse problems, and machine-learning-augmented simulation.

**Highlights**

- **Differentiable:** exact gradients of simulation results through efficient adjoint solves, for linear and nonlinear problems alike
- **GPU-accelerated:** a linear elastic model with a million degrees of freedom solves in seconds on a single GPU
- **Broad scope:** solid mechanics and heat transfer from trusses to shells and 3D solids, including nonlinear materials and large deformations
- **Extensible:** a new hyperelastic material needs only its energy function and a new element only its shape functions, as autograd derives the rest
- **PyTorch-native:** models are plain tensors, so FEM results plug directly into optimizers and neural networks