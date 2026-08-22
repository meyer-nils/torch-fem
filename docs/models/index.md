---
icon: lucide/shapes
---

# Models

A FEM model combines a mesh (`nodes` and `elements`) with a material to form a solvable finite-element problem. All models share the same workflow:

1. **Create** the model from nodes, elements, and a material. Several models can be combined as one system in an [Assembly](assembly.md) and coupled by kinematic constraints.
2. **Apply loads and boundary conditions** by setting entries of the model attributes: `forces` and `displacements` for mechanics models ([Truss](truss.md), [Planar](planar.md), [Shell](shell.md), [Solid](solid.md)), or `heat_flux` and `temperatures` for thermal models (`PlanarHeat`, `SolidHeat`). Prescribed values are activated by setting the corresponding entries of the boolean mask `constraints` to `True`. A distributed load, such as gravity or a pressure, becomes nodal values with the [load integrators](#loads) below.
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
            - integrate_field

## Loads

A distributed load is integrated into consistent nodal loads, which are added to the `forces` of a mechanics model or the `heat_flux` of a thermal one:

```py
model.forces += model.integrate_body_load([0.0, -rho * g])
```

Which integrators a model offers follows from the dimension of its elements:

| Model | Body | Surface | Line |
| --- | :-: | :-: | :-: |
| `Truss` | ✓ | | |
| `Planar`, `PlanarHeat` | ✓ | | ✓ |
| `Shell` | ✓ | ✓ | ✓ |
| `Solid`, `SolidHeat` | ✓ | ✓ | |

::: torchfem.base.FEM.integrate_body_load
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3
        docstring_section_style: list

::: torchfem.base.FEM.integrate_surface_load
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3
        docstring_section_style: list

A `Shell` element is its own surface, so [`Shell.integrate_surface_load(...)`](shell.md#torchfem.Shell.integrate_surface_load) loads the elements in the mask rather than the boundary faces of the mesh.

::: torchfem.base.FEM.integrate_line_load
    options:
        show_root_heading: true
        show_root_full_path: false
        heading_level: 3
        docstring_section_style: list
