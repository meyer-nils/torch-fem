from math import sqrt

import torch


class Tetra1:
    def __init__(self):
        self.nodes = 4

    def N(self, xi):
        N_1 = 1.0 - xi[0] - xi[1] - xi[2]
        N_2 = xi[0]
        N_3 = xi[1]
        N_4 = xi[2]
        return torch.stack([N_1, N_2, N_3, N_4], dim=2)

    def B(self, _):
        return torch.tensor(
            [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]]
        )

    def ipoints(self):
        return [[0.25, 0.25, 0.25]]

    def iweights(self):
        return [1.0 / 6.0]


class Hexa1:
    def __init__(self):
        self.nodes = 8

    def N(self, xi):
        N_1 = (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
        N_2 = (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2])
        N_3 = (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
        N_4 = (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2])
        N_5 = (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
        N_6 = (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2])
        N_7 = (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])
        N_8 = (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2])
        return 0.125 * torch.tensor([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8])

    def B(self, xi):
        return 0.125 * torch.tensor(
            [
                [
                    -(1.0 - xi[1]) * (1.0 - xi[2]),
                    (1.0 - xi[1]) * (1.0 - xi[2]),
                    (1.0 + xi[1]) * (1.0 - xi[2]),
                    -(1.0 + xi[1]) * (1.0 - xi[2]),
                    -(1.0 - xi[1]) * (1.0 + xi[2]),
                    (1.0 - xi[1]) * (1.0 + xi[2]),
                    (1.0 + xi[1]) * (1.0 + xi[2]),
                    -(1.0 + xi[1]) * (1.0 + xi[2]),
                ],
                [
                    -(1.0 - xi[0]) * (1.0 - xi[2]),
                    -(1.0 + xi[0]) * (1.0 - xi[2]),
                    (1.0 + xi[0]) * (1.0 - xi[2]),
                    (1.0 - xi[0]) * (1.0 - xi[2]),
                    -(1.0 - xi[0]) * (1.0 + xi[2]),
                    -(1.0 + xi[0]) * (1.0 + xi[2]),
                    (1.0 + xi[0]) * (1.0 + xi[2]),
                    (1.0 - xi[0]) * (1.0 + xi[2]),
                ],
                [
                    -(1.0 - xi[0]) * (1.0 - xi[1]),
                    -(1.0 + xi[0]) * (1.0 - xi[1]),
                    -(1.0 + xi[0]) * (1.0 + xi[1]),
                    -(1.0 - xi[0]) * (1.0 + xi[1]),
                    (1.0 - xi[0]) * (1.0 - xi[1]),
                    (1.0 + xi[0]) * (1.0 - xi[1]),
                    (1.0 + xi[0]) * (1.0 + xi[1]),
                    (1.0 - xi[0]) * (1.0 + xi[1]),
                ],
            ]
        )

    def ipoints(self):
        return [
            [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0), xi_3 / sqrt(3.0)]
            for xi_3 in [-1.0, 1.0]
            for xi_2 in [-1.0, 1.0]
            for xi_1 in [-1.0, 1.0]
        ]

    def iweights(self):
        return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


class Solid:
    def __init__(
        self,
        nodes,
        elements,
        forces,
        constraints,
        C,
    ):
        self.nodes = nodes
        self.n_dofs = torch.numel(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = forces
        self.constraints = constraints
        if len(elements[0]) == 8:
            self.etype = Hexa1()
        elif len(elements[0]) == 4:
            self.etype = Tetra1()
        self.C = C

        # Compute efficient mapping from local to global indices
        self.global_indices = []
        for element in self.elements:
            indices = torch.tensor([3 * n + i for n in element for i in range(3)])
            self.global_indices.append(torch.meshgrid(indices, indices, indexing="xy"))

    def volumes(self):
        volumes = torch.zeros((self.n_elem))
        for j, element in enumerate(self.elements):
            # Perform integrations
            nodes = self.nodes[element, :]
            volume = 0.0
            for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
                # Jacobian
                J = self.etype.B(q) @ nodes
                detJ = torch.linalg.det(J)
                # Volume integration
                volume += w * detJ
            volumes[j] = volume
        return volumes

    def k(self, j):
        # Perform integrations
        nodes = self.nodes[self.elements[j], :]
        k = torch.zeros((3 * self.etype.nodes, 3 * self.etype.nodes))
        for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
            # Jacobian
            J = self.etype.B(q) @ nodes
            detJ = torch.linalg.det(J)
            if detJ <= 0.0:
                raise Exception("Negative Jacobian. Check element numbering.")
            # Element stiffness
            B = torch.linalg.inv(J) @ self.etype.B(q)
            zeros = torch.zeros(self.etype.nodes)
            D0 = torch.stack([B[0, :], zeros, zeros], dim=-1).ravel()
            D1 = torch.stack([zeros, B[1, :], zeros], dim=-1).ravel()
            D2 = torch.stack([zeros, zeros, B[2, :]], dim=-1).ravel()
            D3 = torch.stack([zeros, B[2, :], B[1, :]], dim=-1).ravel()
            D4 = torch.stack([B[2, :], zeros, B[0, :]], dim=-1).ravel()
            D5 = torch.stack([B[1, :], B[0, :], zeros], dim=-1).ravel()
            D = torch.stack([D0, D1, D2, D3, D4, D5])
            k[:, :] += w * D.T @ self.C @ D * detJ
        return k

    def stiffness(self):
        # Assemble global stiffness matrix
        K = torch.zeros((self.n_dofs, self.n_dofs))
        for j in range(len(self.elements)):
            K[self.global_indices[j]] += self.k(j)
        return K

    def solve(self):
        # Compute global stiffness matrix
        K = self.stiffness()

        # Get reduced stiffness matrix
        uncon = torch.nonzero(~self.constraints.ravel(), as_tuple=False).ravel()
        K_red = K[uncon][:, uncon]
        f_red = self.forces.ravel()[uncon]

        # Solve for displacement
        u_red = torch.linalg.solve(K_red, f_red)
        u = torch.zeros_like(self.nodes).ravel()
        u[uncon] = u_red

        # Evaluate force
        f = K @ u

        u = u.reshape((-1, 3))
        f = f.reshape((-1, 3))
        return u, f

    @torch.no_grad()
    def plot(self, u=0.0, node_property=None, element_property=None):
        try:
            import pyvista
        except ImportError:
            raise Exception("Plotting 3D requires pyvista.")

        pyvista.set_plot_theme("document")
        pl = pyvista.Plotter()
        pl.enable_anti_aliasing("ssaa")

        # VTK cell types
        if isinstance(self.etype, Tetra1):
            cell_types = self.n_elem * [pyvista.CellType.TETRA]
        elif isinstance(self.etype, Hexa1):
            cell_types = self.n_elem * [pyvista.CellType.HEXAHEDRON]

        # VTK element list
        elements = []
        for element in self.elements:
            elements += [len(element), *element]

        # Deformed node positions
        pos = self.nodes + u

        # Create unstructured mesh
        mesh = pyvista.UnstructuredGrid(elements, cell_types, pos.tolist())

        # Plot node properties
        if node_property:
            for key, val in node_property.items():
                mesh.point_data[key] = val

        # Plot cell properties
        if element_property:
            for key, val in element_property.items():
                mesh.cell_data[key] = val

        mesh.plot(show_edges=True)
