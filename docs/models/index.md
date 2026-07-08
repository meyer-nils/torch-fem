---
icon: lucide/shapes
---

# Models

A FEM model combines a mesh (`nodes` and `elements`) with a material to form a solvable finite-element problem. All models share the same workflow:

1. **Create** the model from nodes, elements, and a material.
2. **Apply loads and boundary conditions** by setting entries of the model attributes: `forces` and `displacements` for mechanics models ([Truss](truss.md), [Planar](planar.md), [Shell](shell.md), [Solid](solid.md)), or `heat_flux` and `temperatures` for thermal models (`PlanarHeat`, `SolidHeat`). Prescribed values are activated by setting the corresponding entries of the boolean mask `constraints` to `True`.
3. **Solve** with `solve()`, which returns the nodal solution, the internal nodal forces, and the flux, gradient, and material state at the elements.
4. **Postprocess** the resulting tensors, e.g. with `plot()`.

See [Getting Started](../getting_started.md) for a worked example.

All models inherit their construction and solution interface from the abstract base class `FEM`:

## FEM

::: torchfem.base.FEM
    options:
        show_root_toc_entry: false
        docstring_section_style: list
        members:
            - __init__
            - solve
