# Generators

Each generator takes the number of grid points per axis followed by the extent of the domain, and places the nodes on a regular grid over that domain. The figures below pair the default unit domain with one of explicit lengths, drawn at the same scale.

::: torchfem.mesh.rect_quad
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

![A quadrilateral mesh on a rectangle](../images/mesh/rect_quad_light.png#only-light)
![A quadrilateral mesh on a rectangle](../images/mesh/rect_quad_dark.png#only-dark)

::: torchfem.mesh.rect_tri
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

![Triangulation variants of rect_tri](../images/mesh/rect_tri_light.png#only-light)
![Triangulation variants of rect_tri](../images/mesh/rect_tri_dark.png#only-dark)

::: torchfem.mesh.cube_hexa
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

![A hexahedral mesh on a cube](../images/mesh/cube_hexa_light.png#only-light)
![A hexahedral mesh on a cube](../images/mesh/cube_hexa_dark.png#only-dark)

::: torchfem.mesh.cube_tetra
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

Two neighbouring hexahedra, showing how the split alternates between them:

![A tetrahedral mesh on a cube](../images/mesh/cube_tetra_light.png#only-light)
![A tetrahedral mesh on a cube](../images/mesh/cube_tetra_dark.png#only-dark)
