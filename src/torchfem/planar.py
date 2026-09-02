from functools import cached_property

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap
from matplotlib.tri import Triangulation
from torch import Tensor

from .base import FEM, Heat, Mechanics
from .elements import Element, Quad1, Quad2, Tria1, Tria2
from .materials import Material
from .plot_utils import LABEL_OFFSET, arrows2d, dots2d, markers2d, signs2d


class PlanarGeometry(FEM):
    """The elements, integration and plotting shared by the planar models.

    It carries the discretization of a surface in the z=0 plane, which `Planar`
    and `PlanarHeat` combine with the physics they solve.

    Attributes:
        nodes: Nodal coordinates with shape [n_nod, 2].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized material model.
        thickness: Element thicknesses with shape [n_elem].
        constraints: Boolean mask of constrained DOFs with shape [n_nod, n_dof].
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
        etype = self.etype.__name__
        return (
            f"<torch-fem planar ({self.n_nod} nodes, {self.n_elem} {etype} elements)>"
        )

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

    @property
    def volume_scale(self) -> Tensor:
        return self.thickness

    def compute_k(self, detJ: Tensor, BCB: Tensor):
        """Element stiffness matrix contribution."""
        return BCB.mul_((self.thickness * detJ)[..., None, None])

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
        color: str = "lightblue",
        alpha: float = 1.0,
        cmap: str | Colormap = "viridis",
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
            orientation: Element-wise material angles in radians, measured
                counter-clockwise, rendered as line markers.
            node_labels: If True, annotates nodes with their indices.
            node_markers: If True, draws markers at nodal positions.
            axes: If True, shows the coordinate axes.
            bcs: If True, indicates applied forces as arrows scaled relative to
                each other and constrained DOFs as markers. In the undeformed
                configuration, prescribed non-zero displacements are drawn to
                scale as arrows with a dot at the tip instead of a marker. In
                the deformed configuration, only the dot is drawn, marking the
                position the node was pulled to. A heat flux is drawn as a plus
                or a minus, and a prescribed temperature keeps its marker.
            color: Element fill color. Edges, markers and labels follow the foreground
                of the style.
            alpha: Opacity of nodal contour plots.
            cmap: Matplotlib colormap or its name.
            linewidth: Element edge line width. Set to 0.0 to hide edges.
            figsize: Figure size when a new figure is created.
            colorbar: If True, adds a colorbar.
            vmin: Lower color limit.
            vmax: Upper color limit.
            title: Plot title.
            ax: Existing matplotlib axes to plot into.
            **kwargs: Forwarded to the `PolyCollection` of the elements, e.g.
                `edgecolor` to override the foreground or `hatch` to fill them
                with a pattern.
        """
        # Compute deformed positions
        pos = self.nodes + u

        # Copy all tensors to CPU
        pos = pos.cpu()
        elements = self.elements.cpu()
        neumann = self._neumann.cpu()
        constraints = self.constraints.cpu()
        prescribed = torch.where(constraints, self._dirichlet.cpu(), 0.0)

        # In a deformed configuration the prescribed displacements are already
        # visible in the plotted positions, so only their tips are drawn there.
        deformed = isinstance(u, Tensor)

        # Bounding box
        size = float(torch.linalg.norm(pos.max() - pos.min()))

        # Set figure size
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Edges, markers and labels follow the style, so a dark theme flips them
        foreground = plt.rcParams["text.color"]

        # Quadratic elements are drawn through their corner nodes
        corners = elements[:, :3] if self.etype in (Tria1, Tria2) else elements[:, :4]
        verts = list(pos[corners].numpy())

        # A property colored onto the surface replaces the plain fill
        colored = node_property is not None

        # Color surface with interpolated nodal properties (if provided)
        if node_property is not None:
            node_property = node_property.squeeze().cpu()
            fan = [corners[:, [0, i, i + 1]] for i in range(1, corners.shape[1] - 1)]
            triangulation = Triangulation(pos[:, 0], pos[:, 1], torch.cat(fan))
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
                colored = True
                pc = PolyCollection(verts, cmap=cmap)
                pc.set_array(element_property)
                pc.set_clim(vmin=vmin, vmax=vmax)
                ax.add_collection(pc)
                if colorbar:
                    plt.colorbar(pc, ax=ax)
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
                    zorder=2,
                )

        # Elements, styled further by any extra keyword
        ax.add_collection(
            PolyCollection(
                verts,
                facecolors="none" if colored else color,
                edgecolors=foreground,
                linewidths=linewidth,
                **kwargs,
            )
        )

        # Nodes
        if node_markers:
            ax.scatter(pos[:, 0], pos[:, 1], color=foreground, marker="o", zorder=3)
            if node_labels:
                for i, (x, y) in enumerate(pos.tolist()):
                    ax.annotate(str(i), (x, y), color=foreground, **LABEL_OFFSET)

        # Boundary conditions
        tips = [pos]
        if bcs:
            # A temperature carries no load arrow, and a prescribed one keeps
            # its marker rather than being drawn to scale
            if neumann.shape[1] == 2:
                fixed = constraints & (deformed | (prescribed == 0.0))
                width = 0.01 * size
                tips.append(arrows2d(ax, pos, neumann, width, span=0.1 * size))
                if not deformed:
                    tips.append(arrows2d(ax, pos, prescribed, width))
                pulled = torch.linalg.norm(prescribed, dim=1) > 0.0
                ends = (pos if deformed else pos + prescribed)[pulled]
                dots2d(ax, ends)
            else:
                fixed = constraints
                signs2d(ax, pos, neumann[:, 0])
            markers2d(ax, pos, fixed)

        # Material orientations
        if orientation is not None:
            orientation = orientation.cpu()
            centers = pos[elements, :].mean(dim=1)
            dir = torch.stack([torch.cos(orientation), torch.sin(orientation)]).T
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

        # Plot limits (collections do not autoscale), including arrow tips
        lo, hi = torch.cat(tips).aminmax(dim=0)
        margin = 0.1 * size
        ax.set_xlim(float(lo[0]) - margin, float(hi[0]) + margin)
        ax.set_ylim(float(lo[1]) - margin, float(hi[1]) + margin)
        ax.set_aspect("equal", adjustable="box")

        if title:
            ax.set_title(title)

        if not axes:
            ax.set_axis_off()


class Planar(PlanarGeometry, Mechanics):
    """Planar mechanics model for plane-stress or plane-strain problems.

    Attributes:
        nodes: Nodal coordinates with shape [n_nod, 2].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized material model.
        thickness: Element thicknesses with shape [n_elem].
        forces: Applied nodal forces with shape [n_nod, 2].
        displacements: Prescribed nodal displacements with shape [n_nod, 2].
        constraints: Boolean mask of constrained DOFs with shape [n_nod, 2].
    """

    @property
    def n_flux(self) -> list[int]:
        """Shape of the stress tensor."""
        return [2, 2]


class PlanarHeat(PlanarGeometry, Heat):
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
