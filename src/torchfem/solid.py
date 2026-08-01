import typing
from functools import cached_property

import pyvista
import torch
from pyvista import DataSet
from torch import Tensor

from .base import Heat, Mechanics
from .elements import Element, Hexa1, Hexa2, Tetra1, Tetra2
from .materials import Material


class Solid(Mechanics):
    """Solid mechanics model for three-dimensional continua.

    The element type (Tetra1, Tetra2, Hexa1, Hexa2) is inferred from the
    number of nodes per element in the connectivity.

    Attributes:
        nodes: Nodal coordinates with shape [n_nod, 3].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized material model.
        forces: Applied nodal forces with shape [n_nod, 3].
        displacements: Prescribed nodal displacements with shape [n_nod, 3].
        constraints: Boolean mask of constrained DOFs with shape [n_nod, 3].
    """

    def __repr__(self) -> str:
        etype = self.etype.__name__
        return f"<torch-fem solid ({self.n_nod} nodes, {self.n_elem} {etype} elements)>"

    @property
    def n_flux(self) -> list[int]:
        """Shape of the stress tensor."""
        return [3, 3]

    @property
    def etype(self) -> type[Element]:
        """Set element type depending on number of nodes per element."""
        if len(self.elements[0]) == 4:
            return Tetra1
        elif len(self.elements[0]) == 8:
            return Hexa1
        elif len(self.elements[0]) == 10:
            return Tetra2
        elif len(self.elements[0]) == 20:
            return Hexa2
        else:
            raise ValueError("Element type not supported.")

    @cached_property
    def char_lengths(self) -> Tensor:
        """Characteristic lengths of the elements."""
        vols = self.integrate_field()
        return vols ** (1 / 3)

    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        """Element stiffness matrix contribution."""
        return torch.einsum("j,jkl->jkl", detJ, BCB)

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor) -> Tensor:
        """Element internal force vector."""
        return torch.einsum("..., ...iI,...Ai->...IA", detJ, B, S)

    def compute_m(self, detJ: Tensor, rho: Tensor) -> Tensor:
        """Element mass matrix contribution."""
        return rho * detJ

    @torch.no_grad()
    def plot(
        self,
        u: float | Tensor = 0.0,
        node_property: dict[str, Tensor] | None = None,
        element_property: dict[str, Tensor] | None = None,
        orientations: Tensor | None = None,
        show_edges: bool = True,
        show_undeformed: bool = False,
        show_outline: bool = False,
        bcs: bool = False,
        clip: tuple[str, float] | None = None,
        plotter: pyvista.Plotter | None = None,
        **kwargs,
    ):
        """Plot the mesh with optional node and element properties.

        Args:
            u (float or torch.Tensor, optional):
                Displacement field. Defaults to 0.0.
            node_property (dict[str, torch.Tensor], optional):
                Nodal property to plot. Defaults to None.
            element_property (dict[str, torch.Tensor], optional):
                Element property to plot. Defaults to None.
            orientations (torch.Tensor, optional):
                Element orientations. Defaults to None.
            show_edges (bool, optional):
                Show edges. Defaults to True.
            show_undeformed (bool, optional):
                Show undeformed mesh. Defaults to False.
            show_outline (bool, optional):
                Show a box around the full mesh. Defaults to False.
            bcs (bool, optional):
                If True, render boundary conditions (forces as arrows,
                prescribed displacements as arrows and tip markers,
                and constrained DOFs as cones). Defaults to False.
            clip (tuple[str, float], optional):
                Property and value to cut the mesh at. Culls orientations and
                boundary conditions with it. Defaults to None.
            plotter (pyvista.Plotter, optional):
                PyVista plotter. Defaults to None.
            **kwargs:
                Additional keyword arguments passed to pyvista.Plotter.add_mesh.
        """

        pyvista.set_plot_theme("document")
        pl = pyvista.Plotter() if plotter is None else plotter
        pl.enable_anti_aliasing("ssaa")

        # VTK cell types
        if self.etype is Tetra1:
            cell_types = self.n_elem * [pyvista.CellType.TETRA]
        elif self.etype is Tetra2:
            cell_types = self.n_elem * [pyvista.CellType.QUADRATIC_TETRA]
        elif self.etype is Hexa1:
            cell_types = self.n_elem * [pyvista.CellType.HEXAHEDRON]
        elif self.etype is Hexa2:
            cell_types = self.n_elem * [pyvista.CellType.QUADRATIC_HEXAHEDRON]

        # VTK element list
        el = len(self.elements[0]) * torch.ones(self.n_elem, dtype=self.elements.dtype)
        elements = torch.cat([el[:, None], self.elements], dim=1).view(-1).tolist()

        # Deformed node positions
        pos = self.nodes + u

        # Create unstructured mesh
        mesh = pyvista.UnstructuredGrid(elements, cell_types, pos.tolist())

        # Plot node properties
        if node_property:
            for key, val in node_property.items():
                mesh.point_data[key] = val.cpu().numpy()

        # Plot cell properties
        if element_property:
            for key, val in element_property.items():
                mesh.cell_data[key] = val.cpu().numpy()

        if show_outline:
            pl.add_mesh(mesh.outline(), color="black")

        # Averaging onto the nodes makes the field continuous, so the cut runs
        # through the elements instead of around them.
        kept = torch.ones(self.n_elem, dtype=torch.bool)
        if clip:
            scalars, value = clip
            # Surviving elements, to cull orientations and BCs
            if element_property and scalars in element_property:
                kept = element_property[scalars] > value
            elif node_property and scalars in node_property:
                kept = (node_property[scalars][self.elements] > value).any(dim=1)
            mesh = mesh.cell_data_to_point_data()
            mesh = mesh.clip_scalar(scalars=scalars, value=value, invert=False)

        # Plot orientations
        if orientations is not None:
            ecenters = pos[self.elements].mean(dim=1)
            for j, color in enumerate(["red", "green", "blue"]):
                directions = orientations[:, j, :]
                pl.add_arrows(
                    ecenters.cpu().numpy()[kept],
                    directions.cpu().numpy()[kept],
                    mag=1,
                    color=color,
                    show_scalar_bar=False,
                )

        # Plot mesh
        mesh = typing.cast(DataSet, mesh)
        if show_edges:
            if self.etype is Tetra2 or self.etype is Hexa2:
                # Trick to plot edges for quadratic elements
                # See: https://github.com/pyvista/pyvista/discussions/5777
                surface = mesh.separate_cells().extract_surface(nonlinear_subdivision=4)
                edges = surface.extract_feature_edges()
                pl.add_mesh(surface, **kwargs)
                actor = pl.add_mesh(edges, style="wireframe", color="black")
                actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            else:
                pl.add_mesh(mesh, show_edges=True, **kwargs)
        else:
            pl.add_mesh(mesh, **kwargs)

        if show_undeformed:
            undefo = pyvista.UnstructuredGrid(elements, cell_types, self.nodes.tolist())
            edges = (
                undefo.separate_cells()
                .extract_surface(nonlinear_subdivision=4)
                .extract_feature_edges()
            )
            pl.add_mesh(edges, style="wireframe", color="grey")

        if bcs and self.n_dof_per_node != 1:
            deformed = isinstance(u, Tensor)
            prescribed = torch.where(self.constraints, self.displacements, 0.0)
            size = torch.linalg.norm(pos.max(dim=0).values - pos.min(dim=0).values)
            height = 0.5 * float(self.char_lengths.mean())

            # Nodes of culled elements carry no visible boundary conditions
            visible = torch.zeros(self.n_nod, dtype=torch.bool)
            visible[self.elements[kept].reshape(-1)] = True

            def glyph(points: Tensor, geom, directions: Tensor | None = None):
                """Put a gray geom on every point, along and scaled by directions."""
                if points.numel() == 0:
                    return
                data = pyvista.PolyData(points.cpu().numpy())
                if directions is not None:
                    length = torch.linalg.norm(directions, dim=1, keepdim=True)
                    data["dir"] = (directions / length).cpu().numpy()
                    data["len"] = length.ravel().cpu().numpy()
                orient = "dir" if directions is not None else False
                glyphs = data.glyph(geom=geom, orient=orient, scale=orient and "len")
                pl.add_mesh(typing.cast(DataSet, glyphs), color="gray")

            # Forces scaled linearly with the largest arrow spanning 10% of the
            # part, prescribed displacements to scale and only when undeformed
            mag = torch.linalg.norm(self.forces, dim=1)
            loaded = (mag > 0.0) & visible
            pulled = (torch.linalg.norm(prescribed, dim=1) > 0.0) & visible
            shown = pulled & (not deformed)
            glyph(
                torch.cat([pos[loaded], pos[shown]]),
                pyvista.Arrow(),
                torch.cat(
                    [0.1 * size / mag.max() * self.forces[loaded], prescribed[shown]]
                ),
            )

            # Spheres marking where the pulled nodes end up
            sphere = pyvista.Sphere(
                radius=0.3 * height, theta_resolution=10, phi_resolution=10
            )
            glyph((pos if deformed else pos + prescribed)[pulled], sphere)

            # Cones per constrained DOF, unless an arrow already shows it
            fixed = self.constraints & visible[:, None]
            if not deformed:
                fixed = fixed & (prescribed == 0.0)
            node, dof = torch.nonzero(fixed).T
            cone = pyvista.Cone(
                center=(-0.5 * height, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0),
                height=height,
                radius=0.5 * height,
                resolution=12,
            )
            glyph(pos[node], cone, torch.eye(3, dtype=pos.dtype)[dof])

        if plotter is None:
            pl.show(jupyter_backend="html")


class SolidHeat(Heat, Solid):
    """Solid heat conduction model.

    Uses the same elements and plotting as `Solid`, but with a single
    temperature degree of freedom per node.

    Attributes:
        nodes: Nodal coordinates with shape [n_nod, 3].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized thermal material model.
        heat_flux: Applied nodal heat sources with shape [n_nod, 1].
        temperatures: Prescribed nodal temperatures with shape [n_nod, 1].
        constraints: Boolean mask of constrained DOFs with shape [n_nod, 1].
    """

    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize the solid heat conduction problem.

        Args:
            nodes: Nodal coordinates with shape [n_nod, 3].
            elements: Connectivity with shape [n_elem, nodes_per_element].
            material: Thermal material model, e.g. `IsotropicConductivity3D`.
        """
        super().__init__(nodes, elements, material)
        self._external_gradient = torch.zeros(self.n_elem, *self.n_flux)
