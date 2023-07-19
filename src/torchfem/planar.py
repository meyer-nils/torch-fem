from math import sqrt

import matplotlib.pyplot as plt
import torch
from matplotlib.collections import PolyCollection


class Tria1:
    def __init__(self):
        self.nodes = 3

    def N(self, xi):
        N_1 = 1.0 - xi[0] - xi[1]
        N_2 = xi[0]
        N_3 = xi[1]
        return torch.tensor([N_1, N_2, N_3])

    def B(self, _):
        return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    def ipoints(self):
        return [[1.0 / 3.0, 1.0 / 3.0]]

    def iweights(self):
        return [0.5]


class Quad1:
    def __init__(self):
        self.nodes = 4

    def N(self, xi):
        N_1 = (1.0 - xi[0]) * (1.0 - xi[1])
        N_2 = (1.0 + xi[0]) * (1.0 - xi[1])
        N_3 = (1.0 + xi[0]) * (1.0 + xi[1])
        N_4 = (1.0 - xi[0]) * (1.0 + xi[1])
        return 0.25 * torch.tensor([N_1, N_2, N_3, N_4])

    def B(self, xi):
        return 0.25 * torch.tensor(
            [
                [-(1.0 - xi[1]), (1.0 - xi[1]), (1.0 + xi[1]), -(1.0 + xi[1])],
                [-(1.0 - xi[0]), -(1.0 + xi[0]), (1.0 + xi[0]), (1.0 - xi[0])],
            ]
        )

    def ipoints(self):
        return [
            [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
            for xi_2 in [-1.0, 1.0]
            for xi_1 in [-1.0, 1.0]
        ]

    def iweights(self):
        return [1.0, 1.0, 1.0, 1.0]


class Planar:
    def __init__(
        self,
        nodes,
        elements,
        forces,
        constraints,
        thickness,
        C,
    ):
        self.nodes = nodes
        self.n_dofs = torch.numel(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = forces
        self.constraints = constraints
        self.thickness = thickness

        # Stack stiffness tensor (for general anisotropy and multi-material assignement)
        if C.shape == torch.Size([3, 3]):
            self.C = C.unsqueeze(0).repeat(self.n_elem, 1, 1)
        else:
            self.C = C

        # Element type
        if len(elements[0]) == 4:
            self.etype = Quad1()
        else:
            self.etype = Tria1()

        # Compute efficient mapping from local to global indices
        gidx_1 = []
        gidx_2 = []
        for element in self.elements:
            indices = torch.tensor([2 * n + i for n in element for i in range(2)])
            idx_1, idx_2 = torch.meshgrid(indices, indices, indexing="xy")
            gidx_1.append(idx_1)
            gidx_2.append(idx_2)
        self.gidx_1 = torch.stack(gidx_1)
        self.gidx_2 = torch.stack(gidx_2)

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
        K = torch.zeros((self.n_dofs, self.n_dofs))
        K.index_put_((self.gidx_1, self.gidx_2), self.k(), accumulate=True)
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

        u = u.reshape((-1, 2))
        f = f.reshape((-1, 2))
        return u, f

    @torch.no_grad()
    def plot(
        self,
        u=0.0,
        node_property=None,
        element_property=None,
        node_labels=False,
        orientation=False,
        color="black",
        cmap="viridis",
        linewidth=1.0,
    ):
        # Compute deformed positions
        pos = self.nodes + u

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min())

        # Color surface with interpolated nodal properties (if provided)
        if node_property is not None:
            if type(self.etype) == Quad1:
                triangles = []
                for e in self.elements:
                    triangles.append([e[0], e[1], e[2]])
                    triangles.append([e[2], e[3], e[0]])
            else:
                triangles = self.elements
            plt.tricontourf(pos[:, 0], pos[:, 1], triangles, node_property, cmap=cmap)

        # Color surface with element properties (if provided)
        if element_property is not None:
            ax = plt.gca()
            verts = pos[self.elements]
            pc = PolyCollection(verts, cmap=cmap)
            pc.set_array(element_property)
            ax.add_collection(pc)

        # Nodes
        if len(pos) < 200:
            plt.scatter(pos[:, 0], pos[:, 1], color=color, marker="o")
            if node_labels:
                for i, node in enumerate(pos):
                    plt.annotate(i, (node[0] + 0.01, node[1] + 0.1), color=color)

        # Elements
        for element in self.elements:
            x1 = [pos[node, 0] for node in element] + [pos[element[0], 0]]
            x2 = [pos[node, 1] for node in element] + [pos[element[0], 1]]
            plt.plot(x1, x2, color=color, linewidth=linewidth)

        # Forces
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

        # Contraints
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

        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("off")
