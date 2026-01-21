from os import PathLike
from typing import Dict, List

import numpy as np
import torch
from meshio import Mesh, read
from torch import Tensor

from torchfem import Planar, Shell, Solid

from .base import FEM, Mechanics
from .elements import ELEMENT_REGISTRY
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


def import_mesh(
    filename: PathLike, material: Material, thickness: float = 1.0
) -> Mechanics:

    mesh = read(filename)
    elems = []
    etypes = []
    allowed_meshio_types = set([e.meshio_type for e in ELEMENT_REGISTRY])
    forbidden_meshio_types = {"line"}
    for cell_block in mesh.cells:
        if cell_block.type in allowed_meshio_types.difference(forbidden_meshio_types):
            etypes.append(cell_block.type)
            elems += cell_block.data.tolist()
    if len(etypes) > 1:
        raise Exception("Currently, only single element types are supported.")
    etype = etypes[0]

    elements = torch.tensor(elems)

    device = torch.get_default_device()
    dtype = torch.get_default_dtype()

    if not np.allclose(mesh.points[:, 2], np.zeros_like(mesh.points[:, 2])):
        nodes = torch.from_numpy(mesh.points.astype(np.float32)).type(dtype).to(device)
        if etype in ["triangle"]:
            return Shell(nodes, elements, material, thickness=thickness)
        elif etype in ["tetra", "tetra10", "hexahedron", "hexahedron20"]:
            return Solid(nodes, elements, material)
        else:
            raise Exception(f"Cannot interpret element type {etype}.")
    else:
        nodes = torch.from_numpy(mesh.points.astype(np.float32)[:, 0:2]).type(dtype).to(device)
        return Planar(nodes, elements, material, thickness=thickness)
