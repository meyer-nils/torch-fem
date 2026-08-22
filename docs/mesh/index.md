---
icon: lucide/grid-2x2
---

# Mesh

Every model is built from `nodes`, the coordinates of the mesh, and `elements`, the node indices of each element:

```py
from torchfem import Planar
from torchfem.mesh import rect_quad

nodes, elements = rect_quad(11, 6, 10.0, 5.0)
model = Planar(nodes, elements, material)
```

The `torchfem.mesh` module provides this pair in two ways:

- [Generators](generators.md) create a structured mesh on a rectangle or a cube.
- [Conversion](conversion.md) turns an existing mesh into one of a different element type.

A mesh of arbitrary geometry comes from a [file](files.md) instead, which `torchfem.io` reads into a finished model.
