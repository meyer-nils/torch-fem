# Conversion

Both converters take a mesh and return a new one, leaving the original untouched.

::: torchfem.mesh.mesh_to_lattice
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

On a quadrilateral mesh, the braced variants add the diagonals of every element:

![Bracing variants on a quadrilateral mesh](../images/mesh/mesh_to_lattice_light.png#only-light)
![Bracing variants on a quadrilateral mesh](../images/mesh/mesh_to_lattice_dark.png#only-dark)

On a hexahedral mesh, they add the diagonals of every element face instead. Neighbouring elements pick matching diagonals, so a face shared by two of them is braced once:

![Bracing variants on two hexahedra](../images/mesh/mesh_to_lattice_hexa_light.png#only-light)
![Bracing variants on two hexahedra](../images/mesh/mesh_to_lattice_hexa_dark.png#only-dark)

::: torchfem.elements.linear_to_quadratic
    options:
        show_root_heading: true
        heading_level: 2
        docstring_section_style: list

The corner nodes are kept and the mid-side nodes appended:

![A linear mesh and its quadratic counterpart](../images/mesh/linear_to_quadratic_light.png#only-light)
![A linear mesh and its quadratic counterpart](../images/mesh/linear_to_quadratic_dark.png#only-dark)
