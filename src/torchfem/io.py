import torch
from meshio import Mesh

from torchfem import Truss


@torch.no_grad()
def export_mesh(mesh, filename, nodal_data={}, elem_data={}):
    if type(mesh) == Truss:
        etype = "line"
    # else:
    #     if type(mesh.etype) == Quad:
    #         etype = "quad"
    #     elif type(mesh.etype) == Tria:
    #         etype = "triangle"

    mesh = Mesh(
        points=mesh.nodes,
        cells={etype: mesh.elements},
        point_data=nodal_data,
        cell_data=elem_data,
    )
    mesh.write(filename)
