from os import PathLike
from pathlib import Path
from typing import Any, overload

import numpy as np
import torch
from meshio import Mesh, read
from torch import Tensor

from torchfem import Planar, PlanarHeat, Shell, Solid, SolidHeat

from .base import FEM, Heat, Mechanics
from .elements import ELEMENT_REGISTRY
from .laminate import Laminate
from .materials import HeatMaterial, Material, MechanicsMaterial


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


def _read_mesh(filename: PathLike) -> tuple[np.ndarray, Tensor, str]:
    """Nodal coordinates, connectivity and element type of a single-block mesh.

    Line blocks and unsupported cell blocks are skipped, so a mesh carrying
    boundary edges beside its faces reads as expected.

    Raises:
        ValueError: If the file holds more than one element type.
    """
    mesh = read(filename)
    elems = []
    etypes = []
    allowed = {e.meshio_type for e in ELEMENT_REGISTRY} - {"line"}
    for cell_block in mesh.cells:
        if cell_block.type in allowed:
            etypes.append(cell_block.type)
            elems += cell_block.data.tolist()
    if len(etypes) > 1:
        raise ValueError("Currently, only single element types are supported.")
    return mesh.points.astype(np.float64), torch.tensor(elems), etypes[0]


@overload
def import_mesh(
    filename: PathLike, material: MechanicsMaterial | Laminate, thickness: float = 1.0
) -> Mechanics: ...


@overload
def import_mesh(
    filename: PathLike, material: HeatMaterial, thickness: float = 1.0
) -> Heat: ...


def import_mesh(
    filename: PathLike, material: Material | Laminate, thickness: float = 1.0
) -> FEM:
    """Imports a mesh file and returns the matching model type.

    The model type follows from the geometry and the element type: a mesh whose nodes
    all lie in the z=0 plane becomes a `Planar` model, a non-planar triangle mesh a
    `Shell`, and a tetrahedral or hexahedral mesh a `Solid`. Use `import_shell(...)`
    to require a shell.

    A `HeatMaterial` gives the heat model of that geometry, and a `Laminate`
    section needs a surface mesh, as only a `Shell` takes one.

    Args:
        filename (PathLike): Path to a mesh file in any format meshio can read.
        material (Material | Laminate): Material or section assigned to the model.
        thickness (float): Thickness of a planar or shell model. Unused for a solid.

    Returns:
        FEM: A model of the type described above.

    Raises:
        Exception: If the file holds more than one element type, or if its element
            type has no corresponding model.
    """
    points, elements, etype = _read_mesh(filename)
    device = torch.get_default_device()
    dtype = torch.get_default_dtype()
    heat = isinstance(material, HeatMaterial)
    planar = np.allclose(points[:, 2], np.zeros_like(points[:, 2]))

    if not planar and etype in ["triangle", "quad"]:
        if heat:
            raise ValueError("A surface mesh has no heat model.")
        nodes = torch.tensor(points, dtype=dtype, device=device)
        return Shell(nodes, elements, material, thickness=thickness)

    if isinstance(material, Laminate):
        raise ValueError(f"A laminate section needs a surface mesh, not {etype}.")

    if planar:
        nodes = torch.tensor(points[:, 0:2], dtype=dtype, device=device)
        model = PlanarHeat if heat else Planar
        return model(nodes, elements, material, thickness=thickness)

    nodes = torch.tensor(points, dtype=dtype, device=device)
    if etype in ["tetra", "tetra10", "hexahedron", "hexahedron20"]:
        model = SolidHeat if heat else Solid
        return model(nodes, elements, material)
    raise ValueError(f"Cannot interpret element type {etype}.")


def import_shell(
    filename: PathLike,
    material: MechanicsMaterial | Laminate,
    thickness: float = 1.0,
    offset: float = 0.0,
) -> Shell:
    """Import a triangle or quadrilateral mesh as a `Shell`, flat or not.

    `import_mesh(...)` reads a flat surface mesh as `Planar` instead. `offset`
    places the reference surface within the section, as a fraction of thickness
    from the mid-plane along the element normal, so `+0.5` puts it on the top
    face and the section hangs below the mesh.

    Raises:
        TypeError: If the mesh is not a surface mesh.
    """
    points, elements, etype = _read_mesh(filename)
    if etype not in ("triangle", "quad"):
        raise TypeError(f"{filename} is not a surface mesh, but {etype}.")
    nodes = torch.tensor(
        points, dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    return Shell(nodes, elements, material, thickness=thickness, offset=offset)
