import torch
from meshio import Mesh

from torchfem import Truss, Planar, Solid, Tria1, Quad1, Tetra1, Hexa1


@torch.no_grad()
def export_mesh(mesh, filename, nodal_data={}, elem_data={}):
    if type(mesh) == Truss:
        etype = "line"
    else:
        if type(mesh.etype) == Quad1:
            etype = "quad"
        elif type(mesh.etype) == Tria1:
            etype = "triangle"
        elif type(mesh.etype) == Tetra1:
            etype = "tetra"
        elif type(mesh.etype) == Hexa1:
            etype = "hexahedron"

    mesh = Mesh(
        points=mesh.nodes,
        cells={etype: mesh.elements},
        point_data=nodal_data,
        cell_data=elem_data,
    )
    mesh.write(filename)


def import_planar_mesh(filename, C):
    import meshio
    import numpy as np

    mesh = meshio.read(filename)
    nodes = torch.from_numpy(mesh.points[:, 0:2].astype(np.float64))
    elements = []
    for cell_block in mesh.cells:
        elements += cell_block.data.tolist()
    forces = torch.zeros_like(nodes)
    constraints = torch.zeros_like(nodes, dtype=bool)
    thickness = torch.ones((len(elements)))
    orientation = torch.zeros((len(elements)))

    return Planar(nodes, elements, forces, constraints, thickness, orientation, C)


def import_solid_mesh(filename, C):
    import meshio
    import numpy as np

    mesh = meshio.read(filename)
    nodes = torch.from_numpy(mesh.points.astype(np.float64))
    elements = []
    for cell_block in mesh.cells:
        elements += cell_block.data.tolist()
    forces = torch.zeros_like(nodes)
    constraints = torch.zeros_like(nodes, dtype=bool)

    return Solid(nodes, elements, forces, constraints, C)
