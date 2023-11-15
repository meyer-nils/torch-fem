import torch

from .elements import Tria1


class Shell:
    def __init__(
        self,
        nodes,
        elements,
        forces,
        displacements,
        constraints,
        thickness,
        C,
    ):
        self.nodes = nodes
        self.n_dofs = 3 * len(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = forces
        self.displacements = displacements
        self.constraints = constraints
        self.thickness = thickness

        # Stack stiffness tensor (for general anisotropy and multi-material assignement)
        if C.shape == torch.Size([3, 3]):
            self.C = C.unsqueeze(0).repeat(self.n_elem, 1, 1)
        else:
            self.C = C

        # Element type and local coordinates
        local_coords = []
        if len(elements[0]) == 4:
            pass
            # self.etype = Quad1()
        else:
            self.etype = Tria1()
            for element in self.elements:
                edge1 = self.nodes[element[1]] - self.nodes[element[0]]
                edge2 = self.nodes[element[2]] - self.nodes[element[1]]
                normal = torch.cross(edge1, edge2)
                normal /= torch.linalg.norm(normal)
                dir1 = edge1 / torch.linalg.norm(edge1)
                dir2 = -torch.linalg.cross(edge1, normal)
                dir2 /= torch.linalg.norm(dir2)
                local_coords.append(torch.vstack([dir1, dir2, normal]))
            # tranformation matrix x = t X with element coords x and global coords X
            self.t = torch.stack(local_coords)
            t = self.t[:, 0:2, :]
            nodes = self.nodes[self.elements, :]
            rel_pos = (nodes - nodes[:, 0, None]).transpose(2, 1)
            self.loc_nodes = (t @ rel_pos).transpose(2, 1)
            self.T = torch.zeros((self.n_elem, 6, 9))
            self.T[:, 0:2, 0:3] = t
            self.T[:, 2:4, 3:6] = t
            self.T[:, 4:6, 6:9] = t

        # Compute efficient mapping from local to global indices
        gidx_1 = []
        gidx_2 = []
        for element in self.elements:
            indices = torch.tensor([3 * n + i for n in element for i in range(3)])
            idx_1, idx_2 = torch.meshgrid(indices, indices, indexing="xy")
            gidx_1.append(idx_1)
            gidx_2.append(idx_2)
        self.gidx_1 = torch.stack(gidx_1)
        self.gidx_2 = torch.stack(gidx_2)

    def k(self):
        # Perform integrations
        k = torch.zeros((self.n_elem, 3 * self.etype.nodes, 3 * self.etype.nodes))
        for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
            # Jacobian
            J = self.etype.B(q) @ self.loc_nodes
            detJ = torch.linalg.det(J)
            if torch.any(detJ <= 0.0):
                raise Exception("Negative Jacobian. Check element numbering.")
            # Element membrane stiffness
            B = torch.linalg.inv(J) @ self.etype.B(q)
            zeros = torch.zeros(self.n_elem, self.etype.nodes)
            D0 = torch.stack([B[:, 0, :], zeros], dim=-1).reshape(self.n_elem, -1)
            D1 = torch.stack([zeros, B[:, 1, :]], dim=-1).reshape(self.n_elem, -1)
            D2 = torch.stack([B[:, 1, :], B[:, 0, :]], dim=-1).reshape(self.n_elem, -1)
            D = torch.stack([D0, D1, D2], dim=1)
            DCD = torch.einsum("...ji,...jk,...kl->...il", D, self.C, D)
            k_mem = torch.einsum("i,ijk->ijk", w * self.thickness * detJ, DCD)
            # Complete stiffness
            k[:, :, :] += self.T.transpose(1, 2) @ k_mem @ self.T
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
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        uncon = torch.nonzero(~self.constraints.ravel(), as_tuple=False).ravel()
        f_d = K[:, con] @ self.displacements.ravel()[con]
        K_red = K[uncon][:, uncon]
        f_red = (self.forces.ravel() - f_d)[uncon]

        # Solve for displacement
        u_red = torch.linalg.solve(K_red, f_red)
        u = self.displacements.clone().ravel()
        u[uncon] = u_red

        # Evaluate force
        f = K @ u

        u = u.reshape((-1, 3))
        f = f.reshape((-1, 3))
        return u, f

    def compute_stress(self, u, xi=[0.0, 0.0]):
        # Extract displacement degrees of freedom
        disp = u[self.elements, :].reshape(self.n_elem, -1)

        # Jacobian
        J = self.etype.B(xi) @ self.loc_nodes

        # Compute B
        B = torch.linalg.inv(J) @ self.etype.B(xi)

        # Compute D
        zeros = torch.zeros(self.n_elem, self.etype.nodes)
        D0 = torch.stack([B[:, 0, :], zeros], dim=-1).reshape(self.n_elem, -1)
        D1 = torch.stack([zeros, B[:, 1, :]], dim=-1).reshape(self.n_elem, -1)
        D2 = torch.stack([B[:, 1, :], B[:, 0, :]], dim=-1).reshape(self.n_elem, -1)
        D = torch.stack([D0, D1, D2], dim=1)

        # Compute stress in local coordinate system
        loc_disp = torch.einsum("...ij,...j->...i", self.T, disp)
        sigma = torch.einsum("...ij,...jk,...k->...i", self.C, D, loc_disp)

        # Compute stress in global coordinate system
        stress_tensor = torch.zeros((self.n_elem, 3, 3))
        stress_tensor[:, 0, 0] = sigma[:, 0]
        stress_tensor[:, 1, 1] = sigma[:, 1]
        stress_tensor[:, 0, 1] = sigma[:, 2]
        stress_tensor[:, 1, 0] = sigma[:, 2]
        return self.t.transpose(1, 2) @ stress_tensor @ self.t

    @torch.no_grad()
    def plot(self, u=0.0, node_property=None, element_property=None):
        try:
            import pyvista
        except ImportError:
            raise Exception("Plotting 3D requires pyvista.")

        pyvista.set_plot_theme("document")
        pyvista.set_jupyter_backend("client")
        pl = pyvista.Plotter()
        pl.enable_anti_aliasing("ssaa")

        # VTK cell types
        cell_types = self.n_elem * [pyvista.CellType.TRIANGLE]

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
