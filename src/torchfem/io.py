from os import PathLike
from pathlib import Path
from typing import Any

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
    nodal_data: dict[str, Tensor] | None = None,
    elem_data: dict[str, list[Tensor]] | None = None,
    compress: bool = True,
):
    """Writes a model and optional result fields to a mesh file.

    The file format is inferred from the suffix of `filename` by meshio. All tensors
    are detached and moved to the CPU before writing.

    Args:
        mesh (FEM): Model providing the nodes and elements to write.
        filename (str | PathLike): Output path. Its suffix selects the format.
        nodal_data (dict[str, Tensor]): Point data to attach, keyed by name.
            *Shape:* `(n_nod, ...)` per entry.
        elem_data (dict[str, list[Tensor]]): Cell data to attach, keyed by name. Each
            value is a *list* with one tensor per cell block, so a single-block model
            takes a one-element list, e.g. `{"rho": [rho]}`.
            *Shape:* `(n_elem, ...)` per tensor.
        compress (bool): Compress the payload. Applies to `.vtu` (zlib) and
            `.xdmf`/`.xmf` (gzip) only, and is ignored for other formats.
    """
    nodal_data = {} if nodal_data is None else nodal_data
    elem_data = {} if elem_data is None else elem_data

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
    suffix = Path(str(filename)).suffix.lower()
    write_kwargs: dict[str, Any] = {}
    if suffix in {".vtu"}:
        write_kwargs["compression"] = "zlib" if compress else None
    elif suffix in {".xdmf", ".xmf"}:
        write_kwargs["compression"] = "gzip" if compress else None
        if compress:
            write_kwargs["compression_opts"] = 4
    msh.write(filename, **write_kwargs)


def import_mesh(
    filename: PathLike, material: Material, thickness: float = 1.0
) -> Mechanics:
    """Imports a mesh file and returns the matching model type.

    The model type follows from the geometry and the element type: a mesh whose nodes
    all lie in the z=0 plane becomes a `Planar` model, a non-planar triangle mesh a
    `Shell`, and a tetrahedral or hexahedral mesh a `Solid`. Use `import_planar(...)`,
    `import_shell(...)`, or `import_solid(...)` to require one specific type.

    Cell blocks that do not correspond to a supported element are skipped, as are line
    blocks, so a mesh storing boundary edges or vertices next to its faces imports as
    expected (`plate_hole.vtk` is one such mesh).

    Args:
        filename (PathLike): Path to a mesh file in any format meshio can read.
        material (Material): Material assigned to the imported model.
        thickness (float): Thickness of the `Planar` or `Shell` model. Unused when a
            `Solid` is returned.

    Returns:
        Mechanics: A `Planar`, `Shell`, or `Solid` model, as described above.

    Raises:
        Exception: If the file holds more than one element type, or if its element
            type has no corresponding model.
    """
    mesh = read(filename)
    elems = []
    etypes = []
    allowed_meshio_types = {e.meshio_type for e in ELEMENT_REGISTRY}
    forbidden_meshio_types = {"line"}
    for cell_block in mesh.cells:
        if cell_block.type in allowed_meshio_types.difference(forbidden_meshio_types):
            etypes.append(cell_block.type)
            elems += cell_block.data.tolist()
    if len(etypes) > 1:
        raise ValueError("Currently, only single element types are supported.")
    etype = etypes[0]

    elements = torch.tensor(elems)

    device = torch.get_default_device()
    dtype = torch.get_default_dtype()

    points = mesh.points.astype(np.float64)

    if not np.allclose(points[:, 2], np.zeros_like(points[:, 2])):
        nodes = torch.tensor(points, dtype=dtype, device=device)
        if etype in ["triangle"]:
            return Shell(nodes, elements, material, thickness=thickness)
        elif etype in ["tetra", "tetra10", "hexahedron", "hexahedron20"]:
            return Solid(nodes, elements, material)
        else:
            raise ValueError(f"Cannot interpret element type {etype}.")
    else:
        nodes = torch.tensor(points[:, 0:2], dtype=dtype, device=device)
        return Planar(nodes, elements, material, thickness=thickness)


def import_shell(
    filename: PathLike, material: Material, thickness: float = 1.0
) -> Shell:
    """Import a mesh as a `Shell`. Raises `TypeError` for other mesh types."""
    mesh = import_mesh(filename, material, thickness)
    if not isinstance(mesh, Shell):
        raise TypeError(f"{filename} is not a shell mesh, but {type(mesh).__name__}.")
    return mesh


def import_planar(
    filename: PathLike, material: Material, thickness: float = 1.0
) -> Planar:
    """Import a mesh as `Planar`. Raises `TypeError` for other mesh types."""
    mesh = import_mesh(filename, material, thickness)
    if not isinstance(mesh, Planar):
        raise TypeError(f"{filename} is not a planar mesh, but {type(mesh).__name__}.")
    return mesh


def import_solid(filename: PathLike, material: Material) -> Solid:
    """Import a mesh as a `Solid`. Raises `TypeError` for other mesh types."""
    mesh = import_mesh(filename, material)
    if not isinstance(mesh, Solid):
        raise TypeError(f"{filename} is not a solid mesh, but {type(mesh).__name__}.")
    return mesh
