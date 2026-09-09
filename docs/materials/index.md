---
icon: lucide/lens-concave
---

# Materials

Material models describe how the stress in a material depends on its deformation. In a general sense, we can formulate this for an isothermal material abstractly as 

$$
 \mathbf{P} = \mathcal{F}(\mathbf{F}, \pmb{\alpha})
$$

with a function $\mathcal{F}$ that maps a deformation gradient $\mathbf{F}$ and material state vector $\pmb{\alpha}$ to a stress. 

In general, this relation is non-linear and often path-dependent. To solve the global equilibrium equations, the material model must be implemented incrementally. Instead of a total mapping, we evaluate the material response over a discrete load step from time $t_n$ to $t_{n+1}$.

The material model is responsible for advancing the material state. Given the state at the beginning of the step ($\mathbf{F}_n,\mathbf{P}_n, \pmb{\alpha}_n$) and an increment of deformation $\Delta \mathbf{H}$, the model must determine the new stress and updated internal variables ($\mathbf{P}_{n+1}, \pmb{\alpha}_{n+1}$). In addition, it must provide an algorithmic tangent stiffness 
$$
\mathbb{C}_{n+1} = \frac{\partial \Delta \pmb{\sigma}}{\partial \Delta \mathbf{H}}
$$
for convergence speed of the underlying incremental Newton-Raphson solver.
In `torch-fem`, this logic is encapsulated in the `step()` method of each material. `Material` holds what every material has -- `vectorize()`, `rotate()`, the density, the state width and the spatial dimension -- and one of two bases adds the balance law.

## Which material for which model

A model takes only a material of its own physics and spatial dimension, and rejects
anything else when it is constructed. `PS` is plane stress and `PE` plane strain.

| Material | Truss | Planar | Solid | Shell | Laminate | PlanarHeat | SolidHeat |
|:---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Isotropic elasticity | 1D | PS · PE | 3D | PS · PE | PS · PE | — | — |
| Orthotropic elasticity | — | PS · PE | 3D | PS · PE | PS · PE | — | — |
| Transverse isotropic elasticity | — | PS · PE | 3D | PS · PE | PS · PE | — | — |
| Isotropic plasticity | 1D | PS · PE | 3D | PS · PE | PS · PE | — | — |
| Hyperelasticity | — | PS · PE | 3D | — | — | — | — |
| Isotropic damage | — | PS · PE | 3D | PS · PE | PS · PE | — | — |
| Isotropic conductivity | — | — | — | — | — | 2D | 3D |
| Orthotropic conductivity | — | — | — | — | — | 2D | 3D |

A hyperelastic material needs geometric nonlinearity, which `Shell` and `Truss` do not
support, so those cells stay empty whatever material is added.

## The step

A `Mechanics` model takes a `MechanicsMaterial`, whose step maps a displacement gradient increment to a stress:

::: torchfem.materials.MechanicsMaterial.step
    options:
        show_root_heading: true
        docstring_section_style: list
        show_bases: false

A `Heat` model takes a `HeatMaterial`, whose step maps a temperature gradient increment to a heat flux:

::: torchfem.materials.HeatMaterial.step
    options:
        show_root_heading: true
        docstring_section_style: list
        show_bases: false
        