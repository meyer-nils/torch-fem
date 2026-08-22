# Files

A mesh of arbitrary geometry is read from file with `meshio`, in any format it supports, and results are written back the same way:

```py
from torchfem.io import export_mesh, import_mesh

model = import_mesh("plate.vtu", material)
u, f, sigma, epsilon, state = model.solve()
export_mesh(model, "result.vtu", nodal_data={"u": u})
```

::: torchfem.io.import_mesh
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

The model type follows from the geometry and the element type:

| Mesh | Model |
| --- | --- |
| Any element type, all nodes at `z = 0` | `Planar` |
| Triangles that leave the `z = 0` plane | `Shell` |
| Tetrahedra or hexahedra | `Solid` |

::: torchfem.io.import_planar
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

::: torchfem.io.import_shell
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

::: torchfem.io.import_solid
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

These three check the type that `import_mesh(...)` picked rather than overriding it, so a flat triangle mesh imports as `Planar` and `import_shell(...)` raises a `TypeError` on it. Read such a mesh with `meshio` and pass its points and cells to `Shell` directly.

::: torchfem.io.export_mesh
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

::: torchfem.data.get_data
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

The bundled meshes are the ones the [Examples](../examples.md) import.
