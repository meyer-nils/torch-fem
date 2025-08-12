import matplotlib.pyplot as plt
import pyvista
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from torch import Tensor

from .base import FEM
from .elements import Bar1, Bar2
from .materials import Material


class Truss(FEM):
    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize a truss FEM problem."""

        super().__init__(nodes, elements, material)

        # Set up areas
        self.areas = torch.ones((len(elements)))

        # Element type
        if len(elements[0]) == 2:
            self.etype = Bar1()
        elif len(elements[0]) == 3:
            self.etype = Bar2()
        else:
            raise ValueError("Element type not supported.")

        # Initialize characteristic lengths
        start_nodes = self.nodes[self.elements[:, 0]]
        end_nodes = self.nodes[self.elements[:, 1]]
        self.char_lengths = torch.linalg.norm(end_nodes - start_nodes, dim=-1)

        # Set element type specific sizes
        self.n_stress = 1
        self.n_int = len(self.etype.iweights())

        # Initialize external strain
        self.ext_strain = torch.zeros(self.n_elem, 1, 1)

    def __repr__(self) -> str:
        etype = self.etype.__class__.__name__
        return f"<torch-fem truss ({self.n_nod} nodes, {self.n_elem} {etype} elements)>"

    def eval_shape_functions(
        self, xi: Tensor, u: Tensor | float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gradient operator at integration points xi."""
        nodes = self.nodes + u
        nodes = nodes[self.elements, :]
        # Direction of the element
        dx = nodes[:, 1] - nodes[:, 0]
        # Length of the element
        l0 = torch.linalg.norm(dx, dim=-1)
        # Cosine and sine of the element
        cs = dx / l0[:, None]

        J = 0.5 * torch.linalg.norm(dx, dim=1)[:, None, None]
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")

        b = self.etype.B(xi)
        B = torch.einsum("jkl,lm->jkm", torch.linalg.inv(J), b)
        B = torch.einsum("ijk,il->ijkl", B, cs).reshape(self.n_elem, -1)[:, None, :]

        return self.etype.N(xi), B, detJ

    def compute_k(self, detJ: Tensor, BCB: Tensor):
        """Element stiffness matrix."""
        return torch.einsum("...,...,...kl->...kl", self.areas, detJ, BCB)

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        """Element internal force vector."""
        return torch.einsum("...,...,...ik,...ij->...kj", self.areas, detJ, B, S)

    def plot(self, u: float | Tensor = 0.0, **kwargs):
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
        default_color: str = "black",
        cmap: str = "viridis",
        title: str | None = None,
        axes: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        ax: Axes | None = None,
    ):
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

        # Forces
        for i, force in enumerate(self.forces):
            if torch.norm(force) > 0.0:
                s = 0.05 * size / torch.linalg.norm(force)  # scale
                plt.arrow(
                    float(pos[i][0]),
                    float(pos[i][1]),
                    s * force[0],
                    s * force[1],
                    width=0.05,
                    facecolor="gray",
                )

        # Constraints
        for i, constraint in enumerate(self.constraints):
            if constraint[0]:
                ax.plot(pos[i][0] - 0.1, pos[i][1], ">", color="gray")
            if constraint[1]:
                ax.plot(pos[i][0], pos[i][1] - 0.1, "^", color="gray")

        # Adjustments
        nmin = pos.min(dim=0).values
        nmax = pos.max(dim=0).values
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
        force_size_factor: float = 0.5,
        constraint_size_factor: float = 0.1,
        cmap: str = "viridis",
    ):

        pyvista.set_plot_theme("document")
        pyvista.set_jupyter_backend("client")
        pl = pyvista.Plotter()
        pl.enable_anti_aliasing("ssaa")

        # Nodes
        pos = self.nodes + u

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min()).item()

        # Radii
        radii = torch.sqrt(self.areas / torch.pi).numpy()

        # Elements
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            tube = pyvista.Tube(pos[n1].numpy(), pos[n2].numpy(), radius=radii[j])
            if element_property is not None:
                for key, value in element_property.items():
                    value = element_property[key].squeeze()
                    tube.cell_data[key] = value[j]
                pl.add_mesh(tube, scalars=key, cmap=cmap)
            else:
                pl.add_mesh(tube, color="gray")

        # Forces
        force_centers = []
        force_directions = []
        for i, force in enumerate(self.forces):
            if torch.norm(force) > 0.0:
                force_centers.append(pos[i])
                force_directions.append(force / torch.linalg.norm(force))
        pl.add_arrows(
            torch.stack(force_centers).numpy(),
            torch.stack(force_directions).numpy(),
            mag=force_size_factor * size,
            color="gray",
        )

        # Constraints
        for i, constraint in enumerate(self.constraints):
            if constraint.any():
                sphere = pyvista.Sphere(
                    radius=constraint_size_factor * size, center=pos[i].numpy()
                )
                pl.add_mesh(sphere, color="gray")

        pl.show(jupyter_backend="html")
