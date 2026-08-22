from functools import cached_property
from typing import cast

import matplotlib.pyplot as plt
import pyvista
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from pyvista import PolyData
from pyvista.plotting import CameraPositionOptions
from torch import Tensor

from .base import Mechanics
from .elements import Bar1, Bar2, Element
from .materials import Material
from .plot_utils import LABEL_OFFSET, arrows, arrows2d, cones, dots, dots2d, markers2d


class Truss(Mechanics):
    """Truss model built from bar elements in 2D or 3D space.

    The element type (Bar1, Bar2) is inferred from the number of nodes per
    element in the connectivity, and the spatial dimension from the nodes.

    Attributes:
        nodes: Nodal coordinates with shape [n_nod, n_dim].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized 1D material model.
        areas: Cross-sectional areas with shape [n_elem]. Defaults to ones.
        forces: Applied nodal forces with shape [n_nod, n_dim].
        displacements: Prescribed nodal displacements with shape
            [n_nod, n_dim].
        constraints: Boolean mask of constrained DOFs with shape
            [n_nod, n_dim].
    """

    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize a truss FEM problem.

        Args:
            nodes: Nodal coordinates with shape [n_nod, n_dim].
            elements: Connectivity with shape [n_elem, nodes_per_element].
            material: 1D material model, e.g. `IsotropicElasticity1D`.
        """
        super().__init__(nodes, elements, material)

        # Set up areas
        self.areas = torch.ones(len(elements))

    def __repr__(self) -> str:
        etype = self.etype.__name__
        return f"<torch-fem truss ({self.n_nod} nodes, {self.n_elem} {etype} elements)>"

    @property
    def n_flux(self) -> list[int]:
        """Shape of the stress tensor."""
        return [1, 1]

    @property
    def etype(self) -> type[Element]:
        """Set element type depending on number of nodes per element."""
        if len(self.elements[0]) == 2:
            return Bar1
        elif len(self.elements[0]) == 3:
            return Bar2
        else:
            raise ValueError("Element type not supported.")

    @cached_property
    def char_lengths(self) -> Tensor:
        """Characteristic lengths of the elements."""
        start_nodes = self.nodes[self.elements[:, 0]]
        end_nodes = self.nodes[self.elements[:, 1]]
        return torch.linalg.norm(end_nodes - start_nodes, dim=-1)

    @property
    def volume_scale(self) -> Tensor:
        return self.areas

    def eval_shape_functions(self, xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Gradient operator at integration points xi."""

        # Compute transformation matrix x = T X with element coords x and
        # global coords X
        nodes = self.nodes[self.elements, :]
        dx = nodes[:, 1] - nodes[:, 0]
        l0 = torch.linalg.norm(dx, dim=-1)
        T = dx[:, None, :] / l0[:, None, None]

        # Compute Jacobian and its determinant
        J = 0.5 * l0[:, None, None]
        detJ = 0.5 * l0[None, :]
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")

        b = self.etype.B(xi)
        B = torch.einsum("...jkl,...lm->...jkm", torch.linalg.inv(J), b)
        B = torch.einsum("...ijk,ijl->...ijkl", B, T).reshape(
            xi.shape[0], self.n_elem, 1, -1
        )
        return self.etype.N(xi), B, detJ

    def compute_k(self, detJ: Tensor, BCB: Tensor):
        """Element stiffness matrix."""
        return torch.einsum("...,...,...kl->...kl", self.areas, detJ, BCB)

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        """Element internal force vector."""
        return torch.einsum("...,...,...ik,...ij->...kj", self.areas, detJ, B, S)

    def compute_m(self, detJ: Tensor, rho: Tensor) -> Tensor:
        """Element mass matrix contribution."""
        return rho * self.areas * detJ

    def plot(self, u: float | Tensor = 0.0, **kwargs):
        """Plot the truss in 2D (matplotlib) or 3D (PyVista).

        Dispatches to `plot2d` or `plot3d` based on the spatial dimension.

        Args:
            u: Nodal displacements added to the positions, e.g. to plot the
                deformed configuration. Defaults to 0.0 (undeformed).
            **kwargs: Forwarded to `plot2d` or `plot3d`, e.g.
                `element_property` (per-element values coloring the bars),
                `show_thickness` (line widths from cross-sectional areas),
                or `node_labels` (annotate node indices, 2D only).
        """
        if self.n_dim == 2:
            self.plot2d(u=u, **kwargs)
        elif self.n_dim == 3:
            self.plot3d(u=u, **kwargs)

    @torch.no_grad()
    def plot2d(
        self,
        u: float | Tensor = 0.0,
        element_property: Tensor | None = None,
        node_labels: bool = True,
        show_thickness: bool = False,
        thickness_threshold: float = 0.0,
        bcs: bool = True,
        default_color: str = "black",
        cmap: str | Colormap = "viridis",
        title: str | None = None,
        axes: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        ax: Axes | None = None,
    ):
        """Plot the truss with matplotlib, optionally with results.

        Args:
            u: Nodal displacements added to the positions, e.g. to plot the
                deformed configuration. Defaults to 0.0 (undeformed).
            element_property: Element field with shape [n_elem] coloring the
                bars.
            node_labels: If True, annotates nodes with their indices.
            show_thickness: If True, scales line widths with the cross-sectional
                areas.
            thickness_threshold: Hides bars with an area below this value. Only
                used when `show_thickness` is False.
            bcs: If True, indicates applied forces as arrows scaled relative to
                each other and constrained DOFs as markers. In the undeformed
                configuration, prescribed non-zero displacements are drawn to
                scale as arrows with a dot at the tip instead of a marker. In
                the deformed configuration, only the dot is drawn, marking the
                position the node was pulled to.
            default_color: Bar, node, and label color.
            cmap: Matplotlib colormap or its name.
            title: Plot title.
            axes: If True, shows the coordinate axes.
            vmin: Lower color limit.
            vmax: Upper color limit.
            ax: Existing matplotlib axes to plot into.
        """
        # Set figure size
        if ax is None:
            _, ax = plt.subplots()

        # Line widths from areas
        if show_thickness:
            a_max = torch.max(self.areas)
            linewidth = 8.0 * self.areas / a_max
        else:
            linewidth = 2.0 * torch.ones(self.n_elem)
            linewidth[self.areas < thickness_threshold] = 0.0

        # Line color from stress (if present)
        if element_property is not None:
            cm = plt.get_cmap(cmap)
            if vmin is None:
                vmin = min(float(element_property.min()), 0.0)
            if vmax is None:
                vmax = max(float(element_property.max()), 0.0)
            color = cm((element_property - vmin) / (vmax - vmin))
            sm = plt.cm.ScalarMappable(cmap=cm, norm=Normalize(vmin=vmin, vmax=vmax))
            plt.colorbar(sm, ax=ax, shrink=0.5)
        else:
            color = self.n_elem * [default_color]

        # Nodes
        pos = self.nodes + u
        ax.scatter(pos[:, 0], pos[:, 1], color=default_color, marker="o", zorder=10)
        if node_labels:
            for i, (x, y) in enumerate(pos.tolist()):
                ax.annotate(str(i), (x, y), color=default_color, **LABEL_OFFSET)

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min())

        # Bars
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            x = [pos[n1][0], pos[n2][0]]
            y = [pos[n1][1], pos[n2][1]]
            ax.plot(x, y, linewidth=linewidth[j], c=color[j], solid_capstyle="round")

        # Boundary conditions
        tips = [pos]
        if bcs:
            deformed = isinstance(u, Tensor)
            prescribed = torch.where(self.constraints, self._dirichlet, 0.0)
            fixed = self.constraints & (deformed | (prescribed == 0.0))
            tips.append(arrows2d(ax, pos, self._neumann, 0.01 * size, span=0.1 * size))
            if not deformed:
                tips.append(arrows2d(ax, pos, prescribed, 0.01 * size))
            pulled = torch.linalg.norm(prescribed, dim=1) > 0.0
            ends = (pos if deformed else pos + prescribed)[pulled]
            dots2d(ax, ends)
            markers2d(ax, pos, fixed)

        # Adjustments (limits include the arrow tips)
        extent = torch.cat(tips)
        nmin = extent.min(dim=0).values
        nmax = extent.max(dim=0).values
        ax.set(
            xlim=(float(nmin[0]) - 0.5, float(nmax[0]) + 0.5),
            ylim=(float(nmin[1]) - 0.5, float(nmax[1]) + 0.5),
        )

        if title:
            ax.set_title(title)

        ax.set_aspect("equal", adjustable="box")
        if not axes:
            ax.set_axis_off()

    @torch.no_grad()
    def plot3d(
        self,
        u: float | Tensor = 0.0,
        element_property: dict[str, Tensor] | None = None,
        axes: bool = False,
        bcs: bool = True,
        cmap: str | Colormap = "viridis",
        plotter: pyvista.Plotter | None = None,
        camera: CameraPositionOptions | None = None,
    ):
        """Plot the truss with PyVista, optionally with results.

        Args:
            u: Nodal displacements added to the positions, e.g. to plot the
                deformed configuration. Defaults to 0.0 (undeformed).
            element_property: Named element fields coloring the bars.
            axes: If True, shows labeled coordinate axes around the truss.
            bcs: If True, renders boundary conditions: arrows for forces and
                prescribed displacements, spheres at displacement tips, and a
                cone per constrained DOF.
            cmap: Colormap name for `element_property`.
            plotter: PyVista plotter. Defaults to None.
            camera: Camera position, either a plane ("xy", "xz", "yz"), "iso",
                or an explicit position, focal point and view up. Defaults to
                None.
        """
        pyvista.set_plot_theme("document")
        pl = pyvista.Plotter() if plotter is None else plotter
        pl.enable_anti_aliasing("ssaa")
        pl.renderer.add_axes()

        # Nodes
        pos = self.nodes + u

        # Bounding box
        size = torch.linalg.norm(pos.max(dim=0).values - pos.min(dim=0).values).item()

        # Radii
        radii = torch.sqrt(self.areas / torch.pi)

        joint = self.elements.ravel()

        def at_joints(values: Tensor) -> Tensor:
            """Largest of a per-bar quantity over the bars meeting at each node."""
            largest = torch.zeros(len(pos))
            source = values.repeat_interleave(2)
            return largest.scatter_reduce_(0, joint, source, "amax", include_self=False)

        # Elements as line segments and joints as points, carrying their radius
        ends = pos[self.elements].reshape(-1, 3).numpy()
        bars = pyvista.line_segments_from_points(ends)
        bars.point_data["radius"] = radii.repeat_interleave(2).numpy()
        joints = pyvista.PolyData(pos.numpy())
        joints.point_data["radius"] = at_joints(radii).numpy()

        scalars = None
        if element_property is not None:
            for scalars, value in element_property.items():
                value = value.squeeze()
                bars.point_data[scalars] = value.repeat_interleave(2).numpy()
                joints.point_data[scalars] = at_joints(value).numpy()

        # Tubes along the bars, with spheres smoothing the joints where they meet
        sphere = pyvista.Sphere(radius=1.0)
        tubes = cast(PolyData, bars.tube(scalars="radius", absolute=True))
        spheres = cast(
            PolyData, joints.glyph(geom=sphere, scale="radius", orient=False)
        )
        pl.add_mesh(tubes + spheres, scalars=scalars, cmap=cmap)

        # Boundary conditions
        if bcs:
            deformed = isinstance(u, Tensor)
            prescribed = torch.where(self.constraints, self._dirichlet, 0.0)
            radius = 0.1 * float(self.char_lengths.mean())
            fixed = self.constraints & (deformed | (prescribed == 0.0))
            arrows(pl, pos, self._neumann, span=0.2 * size)
            if not deformed:
                arrows(pl, pos, prescribed)
            # Large enough to stay visible where the dot sits on a node
            pulled = torch.linalg.norm(prescribed, dim=1) > 0.0
            ends = (pos if deformed else pos + prescribed)[pulled]
            dots(pl, ends, max(0.5 * radius, 1.25 * float(radii.max())))
            cones(pl, pos, fixed, 2.0 * radius)

        if axes:
            pl.renderer.show_grid()

        if camera is not None:
            pl.camera_position = camera

        if plotter is None:
            from .plot_utils import show_html

            show_html(pl)
