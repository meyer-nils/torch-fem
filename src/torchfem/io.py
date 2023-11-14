import torch
from meshio import Mesh

from torchfem import Planar, Shell, Solid, Truss
from torchfem.elements import Hexa1, Quad1, Tetra1, Tria1


@torch.no_grad()
def export_mesh(mesh, filename, nodal_data={}, elem_data={}):
    if isinstance(mesh, Truss):
        etype = "line"
    else:
        if isinstance(mesh.etype, Quad1):
            etype = "quad"
        elif isinstance(mesh.etype, Tria1):
            etype = "triangle"
        elif isinstance(mesh.etype, Tetra1):
            etype = "tetra"
        elif isinstance(mesh.etype, Hexa1):
            etype = "hexahedron"

    mesh = Mesh(
        points=mesh.nodes,
        cells={etype: mesh.elements},
        point_data=nodal_data,
        cell_data=elem_data,
    )
    mesh.write(filename)


def import_mesh(filename, C):
    import meshio
    import numpy as np

    mesh = meshio.read(filename)
    elements = []
    etypes = []
    for cell_block in mesh.cells:
        if cell_block.type in ["triangle", "quad", "tetra" "hexahedron"]:
            etypes.append(cell_block.type)
            elements += cell_block.data.tolist()
    if len(etypes) > 1:
        raise Exception("Currently, only single element types are supported.")
    etype = etypes[0]

    if not np.allclose(mesh.points[:, 2], np.zeros_like(mesh.points[:, 2])):
        nodes = torch.from_numpy(mesh.points.astype(np.float32))
        forces = torch.zeros_like(nodes)
        displacements = torch.zeros_like(nodes)
        constraints = torch.zeros_like(nodes, dtype=bool)
        if etype in ["triangle", "quad"]:
            return Shell(nodes, elements, forces, displacements, constraints, C)
        elif etype in ["tetra" "hexahedron"]:
            return Solid(nodes, elements, forces, displacements, constraints, C)
    else:
        nodes = torch.from_numpy(mesh.points[:, 0:2].astype(np.float32))
        thickness = torch.ones((len(elements)))
        forces = torch.zeros_like(nodes)
        displacements = torch.zeros_like(nodes)
        constraints = torch.zeros_like(nodes, dtype=bool)
        return Planar(nodes, elements, forces, displacements, constraints, thickness, C)
