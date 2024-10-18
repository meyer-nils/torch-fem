import matplotlib.pyplot as plt
import torch
from matplotlib.collections import PolyCollection

from .elements import Quad1, Quad2, Tria1, Tria2
from .sparse import sparse_index_select, sparse_solve


class Planar:
    def __init__(self, nodes: torch.Tensor, elements: torch.Tensor, material):
        self.nodes = nodes
        self.n_dofs = torch.numel(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = torch.zeros_like(nodes)
        self.displacements = torch.zeros_like(nodes)
        self.constraints = torch.zeros_like(nodes, dtype=bool)
        self.thickness = torch.ones(len(elements))

        # Stack stiffness tensor (for general anisotropy and multi-material assignment)
        C = material.C()
        if C.shape == torch.Size([3, 3]):
            self.C = C.unsqueeze(0).repeat(self.n_elem, 1, 1)
        else:
            self.C = C

        # Element type
        if len(elements[0]) == 3:
            self.etype = Tria1()
        elif len(elements[0]) == 4:
            self.etype = Quad1()
        elif len(elements[0]) == 6:
            self.etype = Tria2()
        elif len(elements[0]) == 8:
            self.etype = Quad2()

        # Compute mapping from local to global indices (hard to read, but fast)
        N = self.n_elem
        idx = ((2 * self.elements).unsqueeze(-1) + torch.arange(2)).reshape(N, -1)
        self.indices = torch.stack(
            [
                idx.unsqueeze(-1).expand(N, -1, idx.shape[1]),
                idx.unsqueeze(1).expand(N, idx.shape[1], -1),
            ],
            dim=0,
        )

    def k(self):
        # Perform integrations
        nodes = self.nodes[self.elements, :]
        k = torch.zeros((self.n_elem, 2 * self.etype.nodes, 2 * self.etype.nodes))
        for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
            # Jacobian
            J = self.etype.B(q) @ nodes
            detJ = torch.linalg.det(J)
            if torch.any(detJ <= 0.0):
                raise Exception("Negative Jacobian. Check element numbering.")
            # Element stiffness
            B = torch.linalg.inv(J) @ self.etype.B(q)
            zeros = torch.zeros(self.n_elem, self.etype.nodes)
            D0 = torch.stack([B[:, 0, :], zeros], dim=-1).reshape(self.n_elem, -1)
            D1 = torch.stack([zeros, B[:, 1, :]], dim=-1).reshape(self.n_elem, -1)
            D2 = torch.stack([B[:, 1, :], B[:, 0, :]], dim=-1).reshape(self.n_elem, -1)
            D = torch.stack([D0, D1, D2], dim=1)
            DCD = torch.einsum("...ji,...jk,...kl->...il", D, self.C, D)
            k[:, :, :] += torch.einsum("i,ijk->ijk", w * self.thickness * detJ, DCD)
        return k

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
        u = self.displacements.detach().ravel()
        u[uncon] = u_red

        # Evaluate force
        f = K @ u

        u = u.reshape((-1, 2))
        f = f.reshape((-1, 2))
        return u, f

    def compute_strain(self, u, xi=[0.0, 0.0]):
        # Extract node positions of element
        nodes = self.nodes[self.elements, :]

        # Extract displacement degrees of freedom
        disp = u[self.elements, :].reshape(self.n_elem, -1)

        # Jacobian
        xi = torch.tensor(xi)
        J = self.etype.B(xi) @ nodes

        # Compute B
        B = torch.linalg.inv(J) @ self.etype.B(xi)

        # Compute D
        zeros = torch.zeros(self.n_elem, self.etype.nodes)
        D0 = torch.stack([B[:, 0, :], zeros], dim=-1).reshape(self.n_elem, -1)
        D1 = torch.stack([zeros, B[:, 1, :]], dim=-1).reshape(self.n_elem, -1)
        D2 = torch.stack([B[:, 1, :], B[:, 0, :]], dim=-1).reshape(self.n_elem, -1)
        D = torch.stack([D0, D1, D2], dim=1)

        # Compute strain
        epsilon = torch.einsum("...jk,...k->...j", D, disp)

        return epsilon

    def compute_stress(self, u, xi=[0.0, 0.0], mises=False):
        # Compute strain
        strain = self.compute_strain(u, xi)

        # Compute stress
        sigma = torch.einsum("...ij,...j->...i", self.C, strain)

        # Return stress
        if mises:
            return torch.sqrt(
                sigma[:, 0] ** 2
                - sigma[:, 0] * sigma[:, 1]
                + sigma[:, 1] ** 2
                + 3 * sigma[:, 2] ** 2
            )
        else:
            return sigma

    @torch.no_grad()
    def plot(
        self,
        u=0.0,
        node_property=None,
        element_property=None,
        node_labels=False,
        node_markers=False,
        orientation=False,
        axes=False,
        bcs=True,
        color="black",
        alpha=1.0,
        cmap="viridis",
        linewidth=1.0,
        figsize=(8.0, 6.0),
        colorbar=False,
        vmin=None,
        vmax=None,
        title=None,
    ):
        # Compute deformed positions
        pos = self.nodes + u

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min())

        # Set figure size
        plt.figure(figsize=figsize)

        # Color surface with interpolated nodal properties (if provided)
        if node_property is not None:
            if isinstance(self.etype, (Quad1, Quad2)):
                triangles = []
                for e in self.elements:
                    triangles.append([e[0], e[1], e[2]])
                    triangles.append([e[2], e[3], e[0]])
            else:
                triangles = self.elements[:, :3]
            plt.tricontourf(
                pos[:, 0],
                pos[:, 1],
                triangles,
                node_property,
                cmap=cmap,
                levels=100,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
            )
            if colorbar:
                plt.colorbar()

        # Color surface with element properties (if provided)
        if element_property is not None:
            ax = plt.gca()
            if isinstance(self.etype, Tria2):
                verts = pos[self.elements[:, :3]]
            elif isinstance(self.etype, Quad2):
                verts = pos[self.elements[:, :4]]
            else:
                verts = pos[self.elements]
            pc = PolyCollection(verts, cmap=cmap)
            pc.set_array(element_property)
            ax.add_collection(pc)
            if colorbar:
                plt.colorbar(pc)
                pc.set_clim(vmin=vmin, vmax=vmax)

        # Nodes
        if node_markers:
            plt.scatter(pos[:, 0], pos[:, 1], color=color, marker="o")
            if node_labels:
                for i, node in enumerate(pos):
                    plt.annotate(i, (node[0] + 0.01, node[1] + 0.01), color=color)

        # Elements
        for element in self.elements:
            if isinstance(self.etype, Tria2):
                element = element[:3]
            if isinstance(self.etype, Quad2):
                element = element[:4]
            x1 = [pos[node, 0] for node in element] + [pos[element[0], 0]]
            x2 = [pos[node, 1] for node in element] + [pos[element[0], 1]]
            plt.plot(x1, x2, color=color, linewidth=linewidth)

        # Forces
        if bcs:
            for i, force in enumerate(self.forces):
                if torch.norm(force) > 0.0:
                    x = pos[i][0]
                    y = pos[i][1]
                    plt.arrow(
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
            for i, constraint in enumerate(self.constraints):
                x = pos[i][0]
                y = pos[i][1]
                if constraint[0]:
                    plt.plot(x - 0.01 * size, y, ">", color="gray")
                if constraint[1]:
                    plt.plot(x, y - 0.01 * size, "^", color="gray")

        # Material orientations
        if orientation:
            centers = pos[self.elements, :].mean(dim=1)
            dir = torch.stack(
                [torch.cos(self.phi), -torch.sin(self.phi), torch.zeros_like(self.phi)]
            ).T
            plt.quiver(
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

        if title:
            plt.title(title)

        plt.gca().set_aspect("equal", adjustable="box")
        if not axes:
            plt.axis("off")
