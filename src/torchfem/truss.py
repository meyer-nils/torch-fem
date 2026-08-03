import typing
from functools import cached_property

import matplotlib.pyplot as plt
import pyvista
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from pyvista import DataSet
from torch import Tensor

from .base import Mechanics
from .elements import Bar1, Bar2, Element
from .materials import Material


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
        cmap: str = "viridis",
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
            cmap: Matplotlib colormap name.
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
            for i, node in enumerate(pos):
                ax.annotate(
                    str(i),
                    (node[0].item() + 0.01, node[1].item() + 0.1),
                    color=default_color,
                )

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min())

        # Bars
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            x = [pos[n1][0], pos[n2][0]]
            y = [pos[n1][1], pos[n2][1]]
            ax.plot(x, y, linewidth=linewidth[j], c=color[j])

        # Boundary conditions
        tips = [pos]
        if bcs:
            prescribed = torch.where(self.constraints, self.displacements, 0.0)

            # In a deformed configuration the prescribed displacements are already
            # visible in the plotted positions, so only their tips are drawn there.
            deformed = isinstance(u, Tensor)

            arrow_style = {"width": 0.01 * size, "facecolor": "gray"}

            # Forces scaled linearly, the largest arrow spanning 10% of the plot
            magnitude = torch.linalg.norm(self.forces, dim=1)
            if magnitude.max() > 0.0:
                ends = pos + (0.1 * size / magnitude.max()) * self.forces
                for i in torch.nonzero(magnitude).ravel():
                    ax.arrow(
                        float(pos[i, 0]),
                        float(pos[i, 1]),
                        float(ends[i, 0] - pos[i, 0]),
                        float(ends[i, 1] - pos[i, 1]),
                        **arrow_style,
                    )
                tips.append(ends[magnitude > 0.0])

            # Prescribed displacements to scale, with a dot marking the tip
            magnitude = torch.linalg.norm(prescribed, dim=1)
            if magnitude.max() > 0.0:
                if deformed:
                    # The nodes already sit at the prescribed positions
                    ends = pos[magnitude > 0.0]
                else:
                    ends = (pos + prescribed)[magnitude > 0.0]
                    for i in torch.nonzero(magnitude).ravel():
                        ax.arrow(
                            float(pos[i, 0]),
                            float(pos[i, 1]),
                            float(prescribed[i, 0]),
                            float(prescribed[i, 1]),
                            length_includes_head=True,
                            **arrow_style,
                        )
                ax.scatter(ends[:, 0], ends[:, 1], color="gray", marker="o", zorder=10)
                tips.append(ends)

            # Constrained DOFs as markers, unless an arrow already shows them
            for i, constraint in enumerate(self.constraints):
                if constraint[0] and (deformed or prescribed[i, 0] == 0.0):
                    ax.plot(pos[i][0] - 0.1, pos[i][1], ">", color="gray")
                if constraint[1] and (deformed or prescribed[i, 1] == 0.0):
                    ax.plot(pos[i][0], pos[i][1] - 0.1, "^", color="gray")

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
        bcs: bool = True,
        cmap: str = "viridis",
        plotter: pyvista.Plotter | None = None,
    ):
        """Plot the truss with PyVista, optionally with results.

        Args:
            u: Nodal displacements added to the positions, e.g. to plot the
                deformed configuration. Defaults to 0.0 (undeformed).
            element_property: Named element fields coloring the bars.
            bcs: If True, renders boundary conditions: arrows for forces and
                prescribed displacements, spheres at displacement tips, and a
                cone per constrained DOF.
            cmap: Colormap name for `element_property`.
            plotter: PyVista plotter. Defaults to None.
        """
        pyvista.set_plot_theme("document")
        pl = pyvista.Plotter() if plotter is None else plotter
        pl.enable_anti_aliasing("ssaa")

        # Nodes
        pos = self.nodes + u

        # Bounding box
        size = torch.linalg.norm(pos.max(dim=0).values - pos.min(dim=0).values).item()

        # Radii
        radii = torch.sqrt(self.areas / torch.pi).numpy()

        # Elements
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            tube = pyvista.Tube(
                pointa=pos[n1].numpy(), pointb=pos[n2].numpy(), radius=radii[j]
            )
            if element_property is not None:
                for key, value in element_property.items():
                    value = element_property[key].squeeze()
                    tube.cell_data[key] = value[j].numpy()
                pl.add_mesh(tube, scalars=key, cmap=cmap)
            else:
                pl.add_mesh(tube)

        def glyph(points: Tensor, geom, directions: Tensor | None = None, color=None):
            """Place a copy of geom on every point, oriented along directions."""
            if points.numel() == 0:
                return
            data = pyvista.PolyData(points.numpy())
            if directions is not None:
                data["dir"] = directions.numpy()
            orient = "dir" if directions is not None else False
            glyphs = data.glyph(geom=geom, orient=orient, scale=False)
            pl.add_mesh(typing.cast(DataSet, glyphs), color=color)

        # Spheres smoothing the joints where the tubes meet
        glyph(
            pos,
            pyvista.Sphere(radius=float(radii.mean())),
            color="gray" if element_property is not None else None,
        )

        # Boundary conditions
        if bcs:
            prescribed = torch.where(self.constraints, self.displacements, 0.0)

            # In a deformed configuration the prescribed displacements are already
            # visible in the plotted positions, so only their tips are drawn there.
            deformed = isinstance(u, Tensor)

            # Forces scaled linearly, the largest arrow spanning 20% of the model
            magnitude = torch.linalg.norm(self.forces, dim=1)
            if magnitude.max() > 0.0:
                nonzero = magnitude > 0.0
                pl.add_arrows(
                    pos[nonzero].numpy(),
                    (self.forces[nonzero] / magnitude.max()).numpy(),
                    mag=0.2 * size,
                    color="gray",
                )

            # Cone size
            radius = 0.1 * float(self.char_lengths.mean())

            # Prescribed displacements to scale, with a dot marking the tip
            magnitude = torch.linalg.norm(prescribed, dim=1)
            if magnitude.max() > 0.0:
                nonzero = magnitude > 0.0
                if deformed:
                    ends = pos[nonzero]
                else:
                    ends = (pos + prescribed)[nonzero]
                    pl.add_arrows(
                        pos[nonzero].numpy(), prescribed[nonzero].numpy(), color="gray"
                    )
                # Large enough to stay visible where the dot sits on a node
                dot_radius = max(0.5 * radius, 1.25 * float(radii.max()))
                glyph(ends, pyvista.Sphere(radius=dot_radius), color="gray")

            # Constrained DOFs as cones pointing at the node, unless an arrow
            # already shows them
            fixed = (
                self.constraints if deformed else self.constraints & (prescribed == 0.0)
            )
            node, dof = torch.nonzero(fixed).T
            cone = pyvista.Cone(
                center=(-radius, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0),
                height=2.0 * radius,
                radius=radius,
                resolution=16,
            )
            glyph(pos[node], cone, torch.eye(3)[dof], color="gray")

        if plotter is None:
            pl.show(jupyter_backend="html")
