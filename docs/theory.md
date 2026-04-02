---
icon: lucide/pi
---

# Theory

This page summarizes the continuum and finite-element formulations implemented in *torch-fem*. The implementation supports solving mechanical problems with differentiable solution operators built on PyTorch autograd.

## Mechanics

### Kinematics

Let $\Omega_0 \subset \mathbb{R}^d$ denote the reference domain with material point $\mathbf{X}$ and displacement field $\mathbf{u}(\mathbf{X})$. The deformed configuration is

$$
\mathbf{x}(\mathbf{X}) = \mathbf{X} + \mathbf{u}(\mathbf{X}),
$$

and the deformation gradient is

$$
\mathbf{F} = \nabla \mathbf{x} = \mathbf{I} + \nabla \mathbf{u} = \mathbf{I} + \mathbf{H}
$$

with the displacement gradient $\mathbf{H} = \nabla \mathbf{u}$, where the operator $\nabla \mathbf{u} = \frac{\partial u_i}{\partial X_j}$ denotes the gradient w.r.t. the reference configuration.

### Momentum balance

In quasi-static settings (neglecting inertia) and without body forces (neglecting gravity), the local balance of linear momentum in the reference configuration reads

$$
\mathrm{div} \left( \mathbf{P} \right) = \mathbf{0},
$$

where $\mathbf{P}$ is the first Piola-Kirchhoff stress.

Boundary conditions are applied as:

- Dirichlet: prescribed displacements on $\Gamma_u$
- Neumann: prescribed tractions on $\Gamma_t$

with $\partial\Omega_0 = \Gamma_u \cup \Gamma_t$ and $\Gamma_u \cap \Gamma_t = \varnothing$.

### Material models

Material models describe how the stress in a material depends on its deformation. In a general sense, we can formulate this for an isothermal material abstractly as 

$$
 \mathbf{P} = \mathcal{F}(\mathbf{F}, \pmb{\alpha})
$$

with a function $\mathcal{F}$ that maps a deformation gradient $\mathbf{F}$ and material state vector $\pmb{\alpha}$ to a stress. 


!!! info
    
    See [Materials](materials/index.md) for more details.

## Finite element method

### Weak form

Let $\delta\mathbf{u}$ be an admissible virtual displacement. The weak form of quasi-static mechanics (without body forces) is

$$
\int_{\Omega_0} \mathbf{P} : \nabla\delta\mathbf{u}\,\mathrm{d}\Omega
=
\int_{\Gamma_t} \bar{\mathbf{t}}_0 \cdot \delta\mathbf{u}\,\mathrm{d}\Gamma.
$$

### Discretization

!!! warning 

    Work in progress

### Shape functions

!!! warning 

    Work in progress

### Assembly

!!! warning 

    Work in progress

### Solution procedure

!!! warning 

    Work in progress