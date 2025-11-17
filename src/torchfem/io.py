from os import PathLike
from typing import Dict, List

import torch
from meshio import Mesh
from torch import Tensor

from torchfem import Planar, Shell, Solid

from .base import FEM
from .materials import Material


@torch.no_grad()
def export_mesh(
    mesh: FEM,
    filename: str | PathLike,
    nodal_data: Dict[str, Tensor] = {},
    elem_data: Dict[str, List[Tensor]] = {},
):
    etype = mesh.etype.meshio_type

    msh = Mesh(
        points=mesh.nodes.cpu().detach(),
        cells={etype: mesh.elements.cpu().detach()},
        point_data={key: tensor.cpu().detach() for key, tensor in nodal_data.items()},
        cell_data={
            key: [tensor.cpu().detach() for tensor in tensor_list]
            for key, tensor_list in elem_data.items()
        },
    )
    msh.write(filename)


def import_mesh(filename: PathLike, material: Material):
    import meshio
    import numpy as np

    mesh = meshio.read(filename)
    elems = []
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
            elems += cell_block.data.tolist()
    if len(etypes) > 1:
        raise Exception("Currently, only single element types are supported.")
    etype = etypes[0]

    elements = torch.tensor(elems)
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
