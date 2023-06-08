import torch

torch.set_default_dtype(torch.double)


class Truss:
    def __init__(self, nodes, elements, forces, constraints, areas, moduli):
        self.nodes = nodes
        self.dim = nodes.shape[1]
        self.n_dofs = torch.numel(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = forces
        self.constraints = constraints
        self.areas = areas
        self.moduli = moduli

        # Precompute mapping from local to global indices
        self.global_indices = []
        for element in self.elements:
            indices = torch.tensor(
                [self.dim * n + i for n in element for i in range(self.dim)]
            )
            self.global_indices.append(torch.meshgrid(indices, indices, indexing="xy"))

    def k(self, j):
        element = self.elements[j]
        n1 = element[0]
        n2 = element[1]
        dx = self.nodes[n1] - self.nodes[n2]
        l0 = torch.linalg.norm(dx)
        c = dx / l0
        dims = range(self.dim)
        s1 = [1, -1]
        s2 = [-1, 1]
        m = torch.stack(
            [torch.stack([s * c[i] * c[j] for s in s1 for j in dims]) for i in dims]
            + [torch.stack([s * c[i] * c[j] for s in s2 for j in dims]) for i in dims]
        )

        return self.areas[j] * self.moduli[j] / l0 * m

    def element_lengths(self):
        l0 = torch.zeros((self.n_elem))
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            dx = self.nodes[n1] - self.nodes[n2]
            l0[j] = torch.linalg.norm(dx)
        return l0

    def element_strain_energies(self, u):
        w = torch.zeros((self.n_elem))
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            u_j = torch.stack([u[n1, 0], u[n1, 1], u[n2, 0], u[n2, 1]])
            k0 = self.k(j) / self.areas[j]
            w[j] = 0.5 * u_j @ k0 @ u_j
        return w

    def stiffness(self):
        n_dofs = torch.numel(self.nodes)
        K = torch.zeros((n_dofs, n_dofs))
        for j in range(len(self.elements)):
            K[self.global_indices[j]] += self.k(j)
        return K

    def solve(self):
        # Compute global stiffness matrix
        K = self.stiffness()
        # Get reduced stiffness matrix
        uncon = torch.nonzero(~self.constraints.ravel(), as_tuple=False).ravel()
        K_red = torch.index_select(K, 0, uncon)
        K_red = torch.index_select(K_red, 1, uncon)
        f_red = self.forces.ravel()[uncon]
        # Solve for displacement
        u_red = torch.linalg.solve(K_red, f_red)
        u = torch.zeros_like(self.nodes).ravel()
        u[uncon] = u_red

        # Evaluate force
        f = K @ u

        # Reshape
        u = u.reshape((-1, self.dim))
        f = f.reshape((-1, self.dim))

        # Evaluate stress
        sigma = torch.zeros((self.n_elem))
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            dx = self.nodes[n1] - self.nodes[n2]
            l0 = torch.linalg.norm(dx)
            c = dx / l0
            m = torch.tensor([c[0], c[1], -c[0], -c[1]])
            u_j = torch.tensor([u[n1, 0], u[n1, 1], u[n2, 0], u[n2, 1]])
            sigma[j] = self.moduli[j] / l0 * torch.inner(m, u_j)

        return [u, f, sigma]

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
            plt.colorbar(sm, label="Stress", shrink=0.5)
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
        self,
        u=0.0,
        sigma=None,
        show_thickness=False,
        thickness_threshold=0.0,
        default_color="black",
    ):
        try:
            import pyvista
        except ImportError:
            raise Exception("Plotting 3D requires pyvista.")

        pyvista.set_plot_theme("document")
        pl = pyvista.Plotter()
        pl.enable_anti_aliasing("ssaa")

        # Nodes
        pos = self.nodes + u

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min()).item()

        # Elements
        tubes = pyvista.MultiBlock()
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            radius = torch.sqrt(self.areas[j] / torch.pi)
            line = pyvista.Line(pos[n1], pos[n2])
            tube = line.tube(radius=radius)
            tube.cell_data["Stress"] = sigma[j]
            tubes.append(tube)
        pl.add_mesh(tubes, scalars="Stress", show_scalar_bar=False)

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
            mag=0.5 * size,
            color="gray",
        )

        # Constraints
        for i, constraint in enumerate(self.constraints):
            if constraint.any():
                sphere = pyvista.Sphere(radius=0.1 * size, center=pos[i].numpy())
                pl.add_mesh(sphere, color="gray")

        pl.show()
