import torch

from .elements import Hexa1, Hexa2, Tetra1, Tetra2
from .sparse import sparse_index_select, sparse_solve


class Solid:
    def __init__(self, nodes: torch.Tensor, elements: torch.Tensor, material):
        self.nodes = nodes
        self.n_dofs = torch.numel(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = torch.zeros_like(nodes)
        self.displacements = torch.zeros_like(nodes)
        self.constraints = torch.zeros_like(nodes, dtype=bool)
        if len(elements[0]) == 4:
            self.etype = Tetra1()
        elif len(elements[0]) == 8:
            self.etype = Hexa1()
        elif len(elements[0]) == 10:
            self.etype = Tetra2()
        elif len(elements[0]) == 20:
            self.etype = Hexa2()

        # Additional strains
        self.strains = None

        # Stack stiffness tensor (for general anisotropy and multi-material assignment)
        C = material.C()
        if C.shape == torch.Size([6, 6]):
            self.C = C.unsqueeze(0).repeat(self.n_elem, 1, 1)
        else:
            self.C = C

        # Compute mapping from local to global indices (hard to read, but fast)
        N = self.n_elem
        idx = ((3 * self.elements).unsqueeze(-1) + torch.arange(3)).reshape(N, -1)
        self.indices = torch.stack(
            [
                idx.unsqueeze(1).expand(N, idx.shape[1], -1),
                idx.unsqueeze(-1).expand(N, -1, idx.shape[1]),
            ],
            dim=0,
        )

    def J(self, q, nodes):
        # Jacobian and Jacobian determinant
        J = self.etype.B(q) @ nodes
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")
        return J, detJ

    def D(self, B):
        # Element strain matrix
        zeros = torch.zeros(self.n_elem, self.etype.nodes)
        shape = [self.n_elem, -1]
        D0 = torch.stack([B[:, 0, :], zeros, zeros], dim=-1).reshape(shape)
        D1 = torch.stack([zeros, B[:, 1, :], zeros], dim=-1).reshape(shape)
        D2 = torch.stack([zeros, zeros, B[:, 2, :]], dim=-1).reshape(shape)
        D3 = torch.stack([zeros, B[:, 2, :], B[:, 1, :]], dim=-1).reshape(shape)
        D4 = torch.stack([B[:, 2, :], zeros, B[:, 0, :]], dim=-1).reshape(shape)
        D5 = torch.stack([B[:, 1, :], B[:, 0, :], zeros], dim=-1).reshape(shape)
        return torch.stack([D0, D1, D2, D3, D4, D5], dim=1)

    def k(self):
        # Perform numerical integrations for element stiffness matrix
        nodes = self.nodes[self.elements, :]
        k = torch.zeros((self.n_elem, 3 * self.etype.nodes, 3 * self.etype.nodes))
        for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
            J, detJ = self.J(q, nodes)
            B = torch.matmul(torch.linalg.inv(J), self.etype.B(q))
            D = self.D(B)
            CD = torch.matmul(self.C, D)
            DCD = torch.matmul(CD.transpose(1, 2), D)
            k[:, :, :] += (w * detJ)[:, None, None] * DCD
        return k

    def f(self, epsilon):
        # Compute inelastic forces (e.g. from thermal strain fields)
        nodes = self.nodes[self.elements, :]
        f = torch.zeros(self.n_elem, 3 * self.etype.nodes)
        for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
            J, detJ = self.J(q, nodes)
            B = torch.matmul(torch.linalg.inv(J), self.etype.B(q))
            D = self.D(B)
            CD = torch.matmul(self.C, D)
            DCE = torch.matmul(CD.transpose(1, 2), epsilon.unsqueeze(-1)).squeeze(-1)
            f[:, :] += (w * detJ)[:, None] * DCE
        return f

    def stiffness(self):
        # Assemble global stiffness matrix
        indices = self.indices.reshape((2, -1))
        values = self.k().ravel()
        size = (self.n_dofs, self.n_dofs)
        return torch.sparse_coo_tensor(indices, values, size=size).coalesce()

    def solve(self):
        # Compute global stiffness matrix
        K = self.stiffness()

        # Compute inelastic strains (if provided)
        F = torch.zeros(self.n_dofs)
        if self.strains is not None:
            f = self.f(self.strains)
            for j in range(len(self.strains)):
                F[self.indices[0, j]] += f[j]

        # Get reduced stiffness matrix
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        uncon = torch.nonzero(~self.constraints.ravel(), as_tuple=False).ravel()
        f_d = sparse_index_select(K, [None, con]) @ self.displacements.ravel()[con]
        K_red = sparse_index_select(K, [uncon, uncon])
        f_red = (self.forces.ravel() - f_d + F)[uncon]

        # Solve for displacement
        u_red = sparse_solve(K_red, f_red)
        u = self.displacements.clone().ravel()
        u[uncon] = u_red

        # Evaluate force
        f = K @ u

        u = u.reshape((-1, 3))
        f = f.reshape((-1, 3))
        return u, f

    def compute_strain(self, u, xi=[0.0, 0.0, 0.0]):
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
        D = self.D(B)

        # Compute strain
        epsilon = torch.einsum("...ij,...j->...i", D, disp)
        if self.strains is not None:
            epsilon -= self.strains

        return epsilon

    def compute_stress(self, u, xi=[0.0, 0.0, 0.0]):
        # Compute strain
        epsilon = self.compute_strain(u, xi)

        # Compute stress
        sigma = torch.einsum("...ij,...j->...i", self.C, epsilon)

        # Return stress
        return sigma

    @torch.no_grad()
    def plot(
        self,
        u=0.0,
        node_property=None,
        element_property=None,
        show_edges=True,
        show_undeformed=False,
        contour=None,
    ):
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
        elif isinstance(self.etype, Tetra2):
            cell_types = self.n_elem * [pyvista.CellType.QUADRATIC_TETRA]
        elif isinstance(self.etype, Hexa1):
            cell_types = self.n_elem * [pyvista.CellType.HEXAHEDRON]
        elif isinstance(self.etype, Hexa2):
            cell_types = self.n_elem * [pyvista.CellType.QUADRATIC_HEXAHEDRON]

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

        if contour:
            mesh = mesh.cell_data_to_point_data()
            mesh = mesh.contour(contour)

        # Trick to plot edges for quadratic elements
        # See: https://github.com/pyvista/pyvista/discussions/5777
        if show_edges:
            surface = mesh.separate_cells().extract_surface(nonlinear_subdivision=4)
            edges = surface.extract_feature_edges()
            pl.add_mesh(surface)
            actor = pl.add_mesh(edges, style="wireframe", color="black", line_width=3)
            actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        else:
            pl.add_mesh(mesh)

        if show_undeformed:
            undefo = pyvista.UnstructuredGrid(elements, cell_types, self.nodes.tolist())
            edges = (
                undefo.separate_cells()
                .extract_surface(nonlinear_subdivision=4)
                .extract_feature_edges()
            )
            pl.add_mesh(edges, style="wireframe", color="grey", line_width=1)

        pl.show(jupyter_backend="client")
