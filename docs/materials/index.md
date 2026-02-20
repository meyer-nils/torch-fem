---
icon: lucide/cuboid
---

# Materials

Material models describe how the stress in a material depends on its deformation. In a general sense, we can formulate this for an isothermal material abstractly as 

$$
 \pmb{\sigma} = \mathcal{F}(\mathbf{F}, \pmb{\alpha})
$$

with a function $\mathcal{F}$ that maps a deformation gradient $\mathbf{F}$ and material state vector $\pmb{\alpha}$ to a stress. 

In general, this relation is non-linear and often path-dependent. To solve the global equilibrium equations, the material model must be implemented incrementally. Instead of a total mapping, we evaluate the material response over a discrete load step from time $t_n$ to $t_{n+1}$.

The material model is responsible for advancing the material state. Given the state at the beginning of the step ($\mathbf{F}_n,\pmb{\sigma}_n, \pmb{\alpha}_n$) and an increment of deformation $\Delta \mathbf{H}$, the model must determine the new stress and updated internal variables ($\pmb{\sigma}_{n+1}, \pmb{\alpha}_{n+1}$). In addition, it must provide an algorithmic tangent stiffness 
$$
\mathbb{C}_{n+1} = \frac{\partial \Delta \pmb{\sigma}}{\partial \Delta \mathbf{H}}
$$
for convergence speed ot the underlying incremental Newton-Raphson solver. 
In `torch-fem`, this logic is encapsulated in the `step()` method of each material inherited from the abstract `Material` base class:

::: torchfem.materials.Material.step
    options:
        show_root_heading: true
        docstring_section_style: list
        show_bases: false
        