from functools import cached_property

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.tri import Triangulation
from torch import Tensor

from .base import Heat, Mechanics
from .elements import Element, Quad1, Quad2, Tria1, Tria2
from .materials import Material


class Planar(Mechanics):
    """Planar mechanics model for plane-stress or plane-strain problems.

    The element type (Tria1, Tria2, Quad1, Quad2) is inferred from the number
    of nodes per element in the connectivity.

    Attributes:
        nodes: Nodal coordinates with shape [n_nod, 2].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized material model.
        thickness: Element thicknesses with shape [n_elem].
        forces: Applied nodal forces with shape [n_nod, 2].
        displacements: Prescribed nodal displacements with shape [n_nod, 2].
        constraints: Boolean mask of constrained DOFs with shape [n_nod, 2].
    """

    def __init__(
        self,
        nodes: Tensor,
        elements: Tensor,
        material: Material,
        thickness: Tensor | float = 1.0,
    ):
        """Initialize the planar FEM problem.

        Args:
            nodes: Nodal coordinates with shape [n_nod, 2].
            elements: Connectivity with shape [n_elem, nodes_per_element].
            material: Plane-stress or plane-strain material model.
            thickness: Element thickness. A float is expanded to all elements,
                a tensor assigns one thickness per element.
        """

        super().__init__(nodes, elements, material)

        # Set up thickness
        if isinstance(thickness, float):
            self.thickness = torch.full((self.n_elem,), thickness)
        else:
            self.thickness = torch.as_tensor(thickness)

    def __repr__(self) -> str:
        etype = self.etype.__class__.__name__
        return (
            f"<torch-fem planar ({self.n_nod} nodes, {self.n_elem} {etype} elements)>"
        )

    @property
    def n_flux(self) -> list[int]:
        """Shape of the stress tensor."""
        return [2, 2]

    @property
    def etype(self) -> type[Element]:
        """Set element type depending on number of nodes per element."""
        if len(self.elements[0]) == 3:
            return Tria1
        elif len(self.elements[0]) == 4:
            return Quad1
        elif len(self.elements[0]) == 6:
            return Tria2
        elif len(self.elements[0]) == 8:
            return Quad2
        else:
            raise ValueError("Element type not supported.")

    @cached_property
    def char_lengths(self) -> Tensor:
        """Characteristic lengths of the elements."""
        areas = self.integrate_field()
        return areas ** (1 / 2)

    def compute_k(self, detJ: Tensor, BCB: Tensor):
        """Element stiffness matrix contribution."""
        return torch.einsum("...,...,...kl->...kl", self.thickness, detJ, BCB)

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        """Element internal force vector."""
        return torch.einsum("...,...,...iI,...Ai->...IA", self.thickness, detJ, B, S)

    def compute_m(self, detJ: Tensor, rho: Tensor) -> Tensor:
        """Element mass matrix contribution."""
        return rho * self.thickness * detJ

    @torch.no_grad()
    def plot(
        self,
        u: float | Tensor = 0.0,
        node_property: Tensor | None = None,
        element_property: Tensor | None = None,
        orientation: Tensor | None = None,
        node_labels: bool = False,
        node_markers: bool = False,
        axes: bool = False,
        bcs: bool = True,
        color: str = "black",
        alpha: float = 1.0,
        cmap: str = "viridis",
        linewidth: float = 1.0,
        figsize: tuple[float, float] = (8.0, 6.0),
        colorbar: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        ax: Axes | None = None,
        **kwargs,
    ):
        """Plot the mesh with matplotlib, optionally with results.

        Args:
            u: Nodal displacements added to the positions, e.g. to plot the
                deformed configuration. Defaults to 0.0 (undeformed).
            node_property: Scalar nodal field with shape [n_nod] rendered as
                interpolated contours.
            element_property: Element field rendered as flat colors (shape
                [n_elem]) or as vector arrows (shape [n_elem, 2]).
            orientation: Element-wise material angles in radians rendered as
                line markers.
            node_labels: If True, annotates nodes with their indices.
            node_markers: If True, draws markers at nodal positions.
            axes: If True, shows the coordinate axes.
            bcs: If True, indicates applied forces and constraints.
            color: Line and marker color.
            alpha: Opacity of nodal contour plots.
            cmap: Matplotlib colormap name.
            linewidth: Element edge line width. Set to 0.0 to hide edges.
            figsize: Figure size when a new figure is created.
            colorbar: If True, adds a colorbar.
            vmin: Lower color limit.
            vmax: Upper color limit.
            title: Plot title.
            ax: Existing matplotlib axes to plot into.
        """
        # Compute deformed positions
        pos = self.nodes + u

        # Copy all tensors to CPU
        pos = pos.cpu()
        elements = self.elements.cpu()
        forces = self.forces.cpu()
        constraints = self.constraints

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min())

        # Set figure size
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Color surface with interpolated nodal properties (if provided)
        if node_property is not None:
            node_property = node_property.squeeze().cpu()
            if self.etype is Quad1 or self.etype is Quad2:
                triangles = []
                for e in elements:
                    triangles.append([e[0], e[1], e[2]])
                    triangles.append([e[2], e[3], e[0]])
            else:
                triangles = elements[:, :3].tolist()
            triangulation = Triangulation(pos[:, 0], pos[:, 1], triangles)
            # Adjust levels for some edge cases
            levels = torch.linspace(
                node_property.min(), 1.001 * node_property.max() + 1e-8, 100
            ).cpu()
            tri = ax.tricontourf(
                triangulation,
                node_property,
                cmap=cmap,
                levels=levels,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
            )
            if colorbar:
                plt.colorbar(tri, ax=ax)

        # Color surface with element properties (if provided)
        if element_property is not None:
            element_property = element_property.squeeze().cpu()
            if element_property.numel() == self.n_elem:
                # Plot scalar field
                if self.etype is Tria2:
                    verts = pos[elements[:, :3]]
                elif self.etype is Quad2:
                    verts = pos[elements[:, :4]]
                else:
                    verts = pos[elements]
                pc = PolyCollection([v for v in verts.numpy()], cmap=cmap)
                pc.set_array(element_property)
                ax.add_collection(pc)
                if colorbar:
                    plt.colorbar(pc, ax=ax)
                    pc.set_clim(vmin=vmin, vmax=vmax)
            elif element_property.numel() == 2 * self.n_elem:
                # Plot vector field
                centers = pos[elements, :].mean(dim=1)
                vectors = element_property / torch.linalg.norm(
                    element_property, dim=1, keepdim=True
                )
                ax.quiver(
                    centers[:, 0],
                    centers[:, 1],
                    vectors[:, 0],
                    vectors[:, 1],
                    torch.linalg.norm(element_property, dim=1),
                    pivot="middle",
                    cmap=cmap,
                )

        # Nodes
        if node_markers:
            ax.scatter(pos[:, 0], pos[:, 1], color=color, marker="o")
            if node_labels:
                for i, node in enumerate(pos):
                    ax.annotate(
                        str(i),
                        (node[0].item() + 0.01, node[1].item() + 0.01),
                        color=color,
                    )

        # Elements
        if linewidth > 0.0:
            coords = pos[elements]
            if self.etype is Tria2:
                coords = coords[:, :3]
            if self.etype is Quad2:
                coords = coords[:, :4]
            closed_segments = torch.cat([coords, coords[:, :1, :]], dim=1)
            segments = closed_segments[:, :, None, :]
            segments = torch.cat([segments[:, :-1], segments[:, 1:]], dim=2)
            segments = segments.reshape(-1, 2, 2)
            lc = LineCollection(segments.tolist(), colors=color, linewidths=linewidth)
            ax.add_collection(lc)

        # Forces
        if bcs:
            for i, force in enumerate(forces):
                if torch.norm(force) > 0.0:
                    x = float(pos[i][0])
                    y = float(pos[i][1])
                    ax.arrow(
                        x,
                        y,
                        size * 0.05 * force[0] / torch.norm(force),
                        size * 0.05 * force[1] / torch.norm(force),
                        width=0.01 * size,
                        facecolor="gray",
                        linewidth=0.0,
                        zorder=10,
                    )

        # Constraints
        if bcs:
            for i, constraint in enumerate(constraints):
                x = float(pos[i][0])
                y = float(pos[i][1])
                if len(constraint) == 2:
                    if constraint[0]:
                        ax.plot(x - 0.01 * size, y, ">", color="gray")
                    if constraint[1]:
                        ax.plot(x, y - 0.01 * size, "^", color="gray")
                elif len(constraint) == 1:
                    if constraint[0]:
                        ax.plot(x, y, "s", color="gray")

        # Material orientations
        if orientation is not None:
            orientation = orientation.cpu()
            centers = pos[elements, :].mean(dim=1)
            dir = torch.stack(
                [
                    torch.cos(orientation),
                    -torch.sin(orientation),
                    torch.zeros_like(orientation),
                ]
            ).T
            ax.quiver(
                centers[:, 0],
                centers[:, 1],
                dir[:, 0],
                dir[:, 1],
                pivot="middle",
                headlength=0,
                headaxislength=0,
                headwidth=0,
                width=0.005,
            )

        # Plot limits (collections do not autoscale)
        margin = 0.1 * size
        ax.set_xlim(pos[:, 0].min() - margin, pos[:, 0].max() + margin)
        ax.set_ylim(pos[:, 1].min() - margin, pos[:, 1].max() + margin)
        ax.set_aspect("equal", adjustable="box")

        if title:
            ax.set_title(title)

        if not axes:
            ax.set_axis_off()


class PlanarHeat(Heat, Planar):
    """Planar heat conduction model.

    Uses the same elements and plotting as `Planar`, but with a single
    temperature degree of freedom per node.

    Attributes:
        nodes: Nodal coordinates with shape [n_nod, 2].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized thermal material model.
        thickness: Element thicknesses with shape [n_elem].
        heat_flux: Applied nodal heat sources with shape [n_nod, 1].
        temperatures: Prescribed nodal temperatures with shape [n_nod, 1].
        constraints: Boolean mask of constrained DOFs with shape [n_nod, 1].
    """

    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize the planar heat conduction problem.

        Args:
            nodes: Nodal coordinates with shape [n_nod, 2].
            elements: Connectivity with shape [n_elem, nodes_per_element].
            material: Thermal material model, e.g. `IsotropicConductivity2D`.
        """
        super().__init__(nodes, elements, material)
        self._external_gradient = torch.zeros(self.n_elem, *self.n_flux)
