import torch

from .base import sparse_solve
from .elements import Hexa1, Tetra1


class Solid:
    def __init__(
        self,
        nodes: torch.Tensor,
        elements: torch.Tensor,
        forces: torch.Tensor,
        displacements: torch.Tensor,
        constraints: torch.Tensor,
        C: torch.Tensor,
        strains=None,
    ):
        self.nodes = nodes
        self.n_dofs = torch.numel(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = forces
        self.displacements = displacements
        self.constraints = constraints
        if len(elements[0]) == 8:
            self.etype = Hexa1()
        elif len(elements[0]) == 4:
            self.etype = Tetra1()
        self.strains = strains

        # Stack stiffness tensor (for general anisotropy and multi-material assignement)
        if C.shape == torch.Size([6, 6]):
            self.C = C.unsqueeze(0).repeat(self.n_elem, 1, 1)
        else:
            self.C = C

        # Compute efficient mapping from local to global indices
        global_indices = []
        for element in self.elements:
            idx = torch.tensor([3 * n + i for n in element for i in range(3)])
            global_indices.append(torch.stack(torch.meshgrid(idx, idx, indexing="xy")))
        self.indices = torch.stack(global_indices, dim=1)

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
        D0 = torch.stack([B[:, 0, :], zeros, zeros], dim=-1).reshape(self.n_elem, -1)
        D1 = torch.stack([zeros, B[:, 1, :], zeros], dim=-1).reshape(self.n_elem, -1)
        D2 = torch.stack([zeros, zeros, B[:, 2, :]], dim=-1).reshape(self.n_elem, -1)
        D3 = torch.stack([zeros, B[:, 2, :], B[:, 1, :]], dim=-1).reshape(
            self.n_elem, -1
        )
        D4 = torch.stack([B[:, 2, :], zeros, B[:, 0, :]], dim=-1).reshape(
            self.n_elem, -1
        )
        D5 = torch.stack([B[:, 1, :], B[:, 0, :], zeros], dim=-1).reshape(
            self.n_elem, -1
        )
        return torch.stack([D0, D1, D2, D3, D4, D5], dim=1)

    def k(self):
        # Perform numerical integrations for element stiffness matrix
        nodes = self.nodes[self.elements, :]
        k = torch.zeros((self.n_elem, 3 * self.etype.nodes, 3 * self.etype.nodes))
        for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
            J, detJ = self.J(q, nodes)
            B = torch.linalg.inv(J) @ self.etype.B(q)
            D = self.D(B)
            DCD = torch.einsum("...ji,...jk,...kl->...il", D, self.C, D)
            k[:, :, :] += torch.einsum("i,ijk->ijk", w * detJ, DCD)
        return k

    def f(self, epsilon):
        # Compute inelastic forces (e.g. from thermal strain fields)
        nodes = self.nodes[self.elements, :]
        f = torch.zeros(self.n_elem, 3 * self.etype.nodes)
        for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
            J, detJ = self.J(q, nodes)
            B = torch.linalg.inv(J) @ self.etype.B(q)
            D = self.D(B)
            DC = torch.einsum("...ji,...jk->...ik", D, self.C)
            DCE = torch.einsum("...ij,...j->...i", DC, epsilon)
            f[:, :] += torch.einsum("i,ij->ij", w * detJ, DCE)
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
        f_d = torch.index_select(K, dim=1, index=con) @ self.displacements.ravel()[con]
        K_red = torch.index_select(
            torch.index_select(K, dim=0, index=uncon), dim=1, index=uncon
        )
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

    def compute_stress(self, u, xi=[0.0, 0.0, 0.0]):
        # Extract node positions of element
        nodes = self.nodes[self.elements, :]

        # Extract displacement degrees of freedom
        disp = u[self.elements, :].reshape(self.n_elem, -1)

        # Jacobian
        J = self.etype.B(xi) @ nodes

        # Compute B
        B = torch.linalg.inv(J) @ self.etype.B(xi)

        # Compute D
        D = self.D(B)

        # Compute strain
        epsilon = torch.einsum("...ij,...j->...i", D, disp)
        if self.strains is not None:
            epsilon -= self.strains

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

        pl.add_mesh(mesh, show_edges=show_edges)

        if show_undeformed:
            undefo = pyvista.UnstructuredGrid(elements, cell_types, self.nodes.tolist())
            pl.add_mesh(undefo, color="grey", style="wireframe")

        pl.show(jupyter_backend="client")
