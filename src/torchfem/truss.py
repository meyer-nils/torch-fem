import torch

from .sparse import sparse_index_select, sparse_solve


class Truss:
    def __init__(self, nodes: torch.Tensor, elements: torch.Tensor):
        self.nodes = nodes
        self.dim = nodes.shape[1]
        self.n_dofs = torch.numel(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = torch.zeros_like(nodes)
        self.displacements = torch.zeros_like(nodes)
        self.constraints = torch.zeros_like(nodes, dtype=bool)
        self.areas = torch.ones((len(elements)))
        self.moduli = torch.ones((len(elements)))

        # Precompute mapping from local to global indices
        global_indices = []
        for element in self.elements:
            indices = [self.dim * n + i for n in element for i in range(self.dim)]
            idx = torch.tensor(indices)
            global_indices.append(torch.stack(torch.meshgrid(idx, idx, indexing="xy")))
        self.indices = torch.stack(global_indices, dim=1)

    def k(self):
        dx = self.nodes[self.elements[:, 0]] - self.nodes[self.elements[:, 1]]
        l0 = torch.linalg.norm(dx, dim=-1)
        c = dx / l0[:, None]
        dims = range(self.dim)
        s1 = [1, -1]
        s2 = [-1, 1]
        m = torch.stack(
            [
                torch.stack([s * c[:, i] * c[:, j] for s in s1 for j in dims], dim=-1)
                for i in dims
            ]
            + [
                torch.stack([s * c[:, i] * c[:, j] for s in s2 for j in dims], dim=-1)
                for i in dims
            ],
            dim=-1,
        )
        return (
            self.areas[:, None, None]
            * self.moduli[:, None, None]
            / l0[:, None, None]
            * m
        )

    def stiffness(self):
        # Assemble global stiffness matrix
        indices = self.indices.reshape((2, -1))
        values = self.k().ravel()
        size = (self.n_dofs, self.n_dofs)
        return torch.sparse_coo_tensor(indices, values, size=size).coalesce()

    def solve(self):
        # Compute global stiffness matrix
        K = self.stiffness()

        # Get reduced stiffness matrix
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        uncon = torch.nonzero(~self.constraints.ravel(), as_tuple=False).ravel()
        f_d = sparse_index_select(K, [None, con]) @ self.displacements.ravel()[con]
        K_red = sparse_index_select(K, [uncon, uncon])
        f_red = (self.forces.ravel() - f_d)[uncon]

        # Solve for displacement
        u_red = sparse_solve(K_red, f_red)
        u = self.displacements.clone().ravel()
        u[uncon] = u_red

        # Evaluate force
        f = K @ u

        # Reshape
        u = u.reshape((-1, self.dim))
        f = f.reshape((-1, self.dim))

        return u, f

    def compute_strain(self, u: torch.Tensor):
        # Compute displacement degrees of freedom
        disp = u[self.elements].reshape(-1, 2 * self.dim)

        # Compute orientations
        dx = self.nodes[self.elements[:, 1]] - self.nodes[self.elements[:, 0]]
        l0 = torch.linalg.norm(dx, dim=-1)
        c = dx / l0[:, None]
        m = torch.stack([s * c[:, j] for s in [-1, 1] for j in range(self.dim)], dim=-1)

        # Compute strain
        return torch.einsum("ij,ij->i", m, disp) / l0

    def compute_stress(self, u: torch.Tensor):
        # Compute displacement degrees of freedom
        strain = self.compute_strain(u)

        # Compute stress
        return self.moduli * strain

    def plot(self, **kwargs):
        if self.dim == 2:
            self.plot2d(**kwargs)
        elif self.dim == 3:
            self.plot3d(**kwargs)

    @torch.no_grad()
    def plot2d(
        self,
        u=0.0,
        sigma=None,
        node_labels=True,
        show_thickness=False,
        thickness_threshold=0.0,
        default_color="black",
    ):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except ImportError:
            raise Exception("Plotting 2D requires matplotlib.")

        # Line widths from areas
        if show_thickness:
            a_max = torch.max(self.areas)
            linewidth = 8.0 * self.areas / a_max
        else:
            linewidth = 2.0 * torch.ones(self.n_elem)
            linewidth[self.areas < thickness_threshold] = 0.0

        # Line color from stress (if present)
        if sigma is not None:
            cmap = cm.viridis
            vmin = min(sigma.min(), 0.0)
            vmax = max(sigma.max(), 0.0)
            color = cmap((sigma - vmin) / (vmax - vmin))
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            plt.colorbar(sm, ax=plt.gca(), label="Stress", shrink=0.5)
        else:
            color = self.n_elem * [default_color]

        # Nodes
        pos = self.nodes + u
        plt.scatter(pos[:, 0], pos[:, 1], color=default_color, marker="o")
        if node_labels:
            for i, node in enumerate(pos):
                plt.annotate(i, (node[0] + 0.01, node[1] + 0.1), color=default_color)

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min())

        # Bars
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            x = [pos[n1][0], pos[n2][0]]
            y = [pos[n1][1], pos[n2][1]]
            plt.plot(x, y, linewidth=linewidth[j], c=color[j])

        # Forces
        for i, force in enumerate(self.forces):
            if torch.norm(force) > 0.0:
                s = 0.05 * size / torch.linalg.norm(force)  # scale
                x = pos[i][0]
                y = pos[i][1]
                plt.arrow(
                    x, y, s * force[0], s * force[1], width=0.05, facecolor="gray"
                )

        # Constraints
        for i, constraint in enumerate(self.constraints):
            x = pos[i][0]
            y = pos[i][1]
            if constraint[0]:
                plt.plot(x - 0.1, y, ">", color="gray")
            if constraint[1]:
                plt.plot(x, y - 0.1, "^", color="gray")

        # Adjustments
        nmin = pos.min(dim=0).values
        nmax = pos.max(dim=0).values
        plt.axis([nmin[0] - 0.5, nmax[0] + 0.5, nmin[1] - 0.5, nmax[1] + 0.5])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("off")

    @torch.no_grad()
    def plot3d(
        self, u=0.0, sigma=None, force_size_factor=0.5, constraint_size_factor=0.1
    ):
        try:
            import pyvista
        except ImportError:
            raise Exception("Plotting 3D requires pyvista.")

        pyvista.set_plot_theme("document")
        pyvista.set_jupyter_backend("client")
        pl = pyvista.Plotter()
        pl.enable_anti_aliasing("ssaa")

        # Nodes
        pos = self.nodes + u

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min()).item()

        # Radii
        radii = torch.sqrt(self.areas / torch.pi)

        # Elements
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            tube = pyvista.Tube(pos[n1], pos[n2], radius=radii[j])
            if sigma is not None:
                tube.cell_data["Stress"] = sigma[j]
                pl.add_mesh(tube, scalars="Stress", cmap="viridis")
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

        pl.show(jupyter_backend="client")
