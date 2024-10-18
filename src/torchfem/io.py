import torch
from meshio import Mesh

from torchfem import Planar, Shell, Solid, Truss
from torchfem.elements import Hexa1, Hexa2, Quad1, Quad2, Tetra1, Tetra2, Tria1, Tria2


@torch.no_grad()
def export_mesh(mesh, filename, nodal_data={}, elem_data={}):
    if isinstance(mesh, Truss):
        etype = "line"
    else:
        if isinstance(mesh.etype, Quad1):
            etype = "quad"
        elif isinstance(mesh.etype, Quad2):
            etype = "quad8"
        elif isinstance(mesh.etype, Tria1):
            etype = "triangle"
        elif isinstance(mesh.etype, Tria2):
            etype = "triangle6"
        elif isinstance(mesh.etype, Tetra1):
            etype = "tetra"
        elif isinstance(mesh.etype, Tetra2):
            etype = "tetra10"
        elif isinstance(mesh.etype, Hexa1):
            etype = "hexahedron"
        elif isinstance(mesh.etype, Hexa2):
            etype = "hexahedron20"

    mesh = Mesh(
        points=mesh.nodes,
        cells={etype: mesh.elements},
        point_data=nodal_data,
        cell_data=elem_data,
    )
    mesh.write(filename)


def import_mesh(filename, material):
    import meshio
    import numpy as np

    mesh = meshio.read(filename)
    elements = []
    etypes = []
    for cell_block in mesh.cells:
        if cell_block.type in [
            "triangle",
            "triangle6",
            "quad",
            "quad8",
            "tetra",
            "tetra10",
            "hexahedron",
            "hexahedron20",
        ]:
            etypes.append(cell_block.type)
            elements += cell_block.data.tolist()
    if len(etypes) > 1:
        raise Exception("Currently, only single element types are supported.")
    etype = etypes[0]

    elements = torch.tensor(elements)
    dtype = torch.get_default_dtype()

    if not np.allclose(mesh.points[:, 2], np.zeros_like(mesh.points[:, 2])):
        nodes = torch.from_numpy(mesh.points.astype(np.float32)).type(dtype)
        if etype in ["triangle"]:
            return Shell(nodes, elements, material)
        elif etype in ["tetra", "tetra10", "hexahedron", "hexahedron20"]:
            return Solid(nodes, elements, material)
    else:
        nodes = torch.from_numpy(mesh.points.astype(np.float32)[:, 0:2]).type(dtype)
        return Planar(nodes, elements, material)
