"""Shell formulation

The Shell element formulation is based on:
Krysl, Petr, Robust flat-facet triangular shell finite element, International Journal
for Numerical Methods in Engineering, vol. 123, issue 10, pp. 2399-2423, 2022
https://doi.org/10.1002/nme.6944
"""
from math import sqrt

import torch

from .elements import Tria1

NDOF = 6


class Shell:
    def __init__(
        self,
        nodes,
        elements,
        forces,
        displacements,
        constraints,
        thickness,
        E,
        nu,
    ):
        self.nodes = nodes
        self.n_dofs = NDOF * len(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = forces
        self.displacements = displacements
        self.constraints = constraints
        self.thickness = thickness

        # Set nodes
        self.update_local_nodes()

        # Build isotropic stiffness tensor
        self.C = (E / (1.0 - nu**2)) * torch.tensor(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
        )

        # Shear stiffness properties
        self.kappa = 5.0 / 6.0
        self.alpha = self.kappa / (2 * (1 + nu))
        self.G = E / (2 * (1 + nu))

        # Drill stiffness properties
        self.drill_penalty = 1.0

        # Element type
        self.etype = Tria1()

        # Compute efficient mapping from local to global indices
        gidx_1 = []
        gidx_2 = []
        for element in self.elements:
            indices = torch.tensor([NDOF * n + i for n in element for i in range(NDOF)])
            idx_1, idx_2 = torch.meshgrid(indices, indices, indexing="xy")
            gidx_1.append(idx_1)
            gidx_2.append(idx_2)
        self.gidx_1 = torch.stack(gidx_1)
        self.gidx_2 = torch.stack(gidx_2)

    def _Dm(self, B):
        """Aggregate strain-displacement matrices

        Args:
            B (torch tensor): Derivative of element shape functions (shape: [N x 2 x 3])

        Returns:
            torch tensor: Strain-displacement matrices shaped [N x 3 x 18]
        """
        N = self.n_elem
        z = torch.zeros(N, self.etype.nodes)
        D0 = torch.stack([B[:, 0, :], z, z, z, z, z], dim=-1).reshape(N, -1)
        D1 = torch.stack([z, B[:, 1, :], z, z, z, z], dim=-1).reshape(N, -1)
        D2 = torch.stack([B[:, 1, :], B[:, 0, :], z, z, z, z], dim=-1).reshape(N, -1)
        return torch.stack([D0, D1, D2], dim=1)

    def _Db(self, B):
        """Aggregate curvature-displacement matrices

        Args:
            B (torch tensor): Derivative of element shape functions (shape: [N x 2 x 3])

        Returns:
            torch tensor: Curvature-displacement matrices shaped [N x 3 x 18]
        """
        N = self.n_elem
        z = torch.zeros(N, self.etype.nodes)
        D0 = torch.stack([z, z, z, z, B[:, 0, :], z], dim=-1).reshape(N, -1)
        D1 = torch.stack([z, z, z, -B[:, 1, :], z, z], dim=-1).reshape(N, -1)
        D2 = torch.stack([z, z, z, -B[:, 0, :], B[:, 1, :], z], dim=-1).reshape(N, -1)
        return torch.stack([D0, D1, D2], dim=1)

    def _Ds(self, A):
        """Aggregate shear-displacement matrices.

        Args:
            A (torch tensor): Element surface areas (shape: [N])

        Returns:
            torch tensor: Shear-displacement matrices shaped [N x 3 x 18]
        """
        N = self.n_elem
        z = torch.zeros(N)

        def compute(nodes):
            a = nodes[:, 1, 0] - nodes[:, 0, 0]
            b = nodes[:, 1, 1] - nodes[:, 0, 1]
            c = nodes[:, 2, 0] - nodes[:, 0, 0]
            d = nodes[:, 2, 1] - nodes[:, 0, 1]
            D0 = torch.stack(
                [
                    torch.stack([z, z, b - d, z, A, z], dim=-1),
                    torch.stack([z, z, c - a, -A, z, z], dim=-1),
                ],
                dim=1,
            ) / (2.0 * A[:, None, None])
            D1 = torch.stack(
                [
                    torch.stack([z, z, d, -b * d / 2.0, a * d / 2.0, z], dim=-1),
                    torch.stack([z, z, -c, b * c / 2.0, -a * c / 2.0, z], dim=-1),
                ],
                dim=1,
            ) / (2.0 * A[:, None, None])
            D2 = torch.stack(
                [
                    torch.stack([z, z, -b, b * d / 2.0, -b * c / 2.0, z], dim=-1),
                    torch.stack([z, z, a, -a * d / 2.0, a * c / 2.0, z], dim=-1),
                ],
                dim=1,
            ) / (2.0 * A[:, None, None])
            return D0, D1, D2

        D0_012, D1_012, D2_012 = compute(self.loc_nodes[:, [0, 1, 2], :])
        D1_120, D2_120, D0_120 = compute(self.loc_nodes[:, [1, 2, 0], :])
        D2_201, D0_201, D1_201 = compute(self.loc_nodes[:, [2, 0, 1], :])
        D0 = (D0_012 + D0_120 + D0_201) / 3.0
        D1 = (D1_012 + D1_120 + D1_201) / 3.0
        D2 = (D2_012 + D2_120 + D2_201) / 3.0
        return torch.cat([D0, D1, D2], dim=-1)

    def update_local_nodes(self):
        # Element type and local coordinates
        local_coords = []
        for element in self.elements:
            edge1 = self.nodes[element[1]] - self.nodes[element[0]]
            edge2 = self.nodes[element[2]] - self.nodes[element[1]]
            normal = torch.cross(edge1, edge2)
            normal = normal / torch.linalg.norm(normal)
            dir1 = edge1 / torch.linalg.norm(edge1)
            dir2 = -torch.linalg.cross(edge1, normal)
            dir2 = dir2 / torch.linalg.norm(dir2)
            local_coords.append(torch.vstack([dir1, dir2, normal]))

        # Tranformation matrix x = t X with element coords x and global coords X
        self.t = torch.stack(local_coords)
        self.T = torch.func.vmap(torch.block_diag)(*(NDOF * [self.t]))

        # Compute local node coordinates
        nodes = self.nodes[self.elements, :]
        rel_pos = (nodes - nodes[:, 0, None]).transpose(2, 1)
        self.loc_nodes = (self.t @ rel_pos).transpose(2, 1)[:, :, 0:2]

    def k(self):
        # Perform integrations
        k = torch.zeros((self.n_elem, NDOF * self.etype.nodes, NDOF * self.etype.nodes))
        for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
            # Jacobian
            J = self.etype.B(q) @ self.loc_nodes
            detJ = torch.linalg.det(J)
            A = detJ / 2.0
            if torch.any(detJ <= 0.0):
                raise Exception("Negative Jacobian. Check element numbering.")

            # Derivative of shape functions
            B = torch.linalg.inv(J) @ self.etype.B(q)

            # Element membrane stiffness
            Dm = self._Dm(B)
            DmCDm = torch.einsum("...ji,...jk,...kl->...il", Dm, self.C, Dm)
            km = torch.einsum("i,ijk->ijk", w * self.thickness * detJ, DmCDm)

            # Element bending stiffness
            Db = self._Db(B)
            DbCDb = torch.einsum("...ji,...jk,...kl->...il", Db, self.C, Db)
            kb = torch.einsum(
                "i,ijk->ijk", w * self.thickness**3 * detJ / 12.0, DbCDb
            )

            # Element transverse stiffness
            Ds = self._Ds(A)
            h = sqrt(2) * A
            psi = (
                self.kappa
                * self.thickness**2
                / (self.thickness**2 + self.alpha * h**2)
            )
            Cs = self.G * torch.eye(2)
            DsCsDs = torch.einsum("...ji,...jk,...kl->...il", Ds, Cs, Ds)
            ks = torch.einsum("i,ijk->ijk", w * A * psi * self.thickness * detJ, DsCsDs)

            # Element drilling stiffness
            kd = torch.zeros_like(km)
            for i in range(self.etype.nodes):
                kd[:, i * NDOF - 1, i * NDOF - 1] = self.drill_penalty

            # Total elemnt stiffness in local coordinates
            kt = km + kb + ks + kd

            # Total element stiffness in global coordinates
            k[:, :, :] += self.T.transpose(1, 2) @ kt @ self.T
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
        u = self.displacements.detach().ravel()
        u[uncon] = u_red

        # Evaluate force
        f = K @ u

        u = u.reshape((-1, NDOF))
        f = f.reshape((-1, NDOF))
        return u, f

    def compute_stress(self, u, xi=[0.0, 0.0], z=0, mises=False):
        # Extract displacement degrees of freedom
        disp = u[self.elements, :].reshape(self.n_elem, -1)

        # Jacobian
        J = self.etype.B(xi) @ self.loc_nodes
        A = torch.linalg.det(J) / 2.0

        # Compute B
        B = torch.linalg.inv(J) @ self.etype.B(xi)

        # Compute in-plane stresses in local coordinate system
        loc_disp = torch.einsum("...ij,...j->...i", self.T, disp)
        sigma_m = torch.einsum("...ij,...jk,...k->...i", self.C, self._Dm(B), loc_disp)
        sigma_b = torch.einsum("...ij,...jk,...k->...i", self.C, self._Db(B), loc_disp)
        sigma = sigma_m + z * sigma_b

        # Compute transverse shear stresses in local coordinate system
        psi = (
            self.kappa
            * self.thickness**2
            / (self.thickness**2 + self.alpha * (sqrt(2) * A) ** 2)
        )
        Cs = torch.einsum("i,jk->ijk", psi, self.G * torch.eye(2))
        sigma_s = torch.einsum("...ij,...jk,...k->...i", Cs, self._Ds(A), loc_disp)

        # Assemble stress tensor
        stress_tensor = torch.zeros((self.n_elem, 3, 3))
        stress_tensor[:, 0, 0] = sigma[:, 0]
        stress_tensor[:, 1, 1] = sigma[:, 1]
        stress_tensor[:, 0, 1] = sigma[:, 2]
        stress_tensor[:, 1, 0] = sigma[:, 2]
        stress_tensor[:, 0, 2] = sigma_s[:, 0]
        stress_tensor[:, 1, 2] = sigma_s[:, 1]

        # Compute stress in global coordinate system
        S = self.t.transpose(1, 2) @ stress_tensor @ self.t

        if mises:
            return torch.sqrt(
                0.5
                * (
                    (S[:, 0, 0] - S[:, 1, 1]) ** 2
                    + (S[:, 0, 0] - S[:, 2, 2]) ** 2
                    + (S[:, 1, 1] - S[:, 2, 2]) ** 2
                )
                + 3.0 * (S[:, 0, 1] ** 2 + S[:, 1, 2] ** 2 + S[:, 0, 2] ** 2)
            )
        else:
            return S

    @torch.no_grad()
    def plot(self, u=0.0, node_property=None, element_property=None, thickness=False):
        try:
            import numpy as np
            import pyvista
        except ImportError:
            raise Exception("Plotting 3D requires pyvista.")

        pyvista.set_plot_theme("document")
        pyvista.set_jupyter_backend("client")
        pl = pyvista.Plotter()
        pl.enable_anti_aliasing("ssaa")

        # VTK element list
        elements = []
        for element in self.elements:
            elements += [len(element), *element]

        # Deformed node positions
        pos = self.nodes + u

        # Create unstructured mesh
        mesh = pyvista.PolyData(pos.tolist(), elements)

        # Plot node properties
        if node_property:
            for key, val in node_property.items():
                mesh.point_data[key] = val

        # Plot cell properties
        if element_property:
            for key, val in element_property.items():
                mesh.cell_data[key] = val

        # Plot as seperate top and bottom surface
        if thickness:
            nodal_thickness = np.zeros((len(self.nodes)))
            count = np.zeros((len(self.nodes)))
            for i, face in enumerate(mesh.faces.reshape(-1, 4)):
                idx = face[1::]
                nodal_thickness[idx] += self.thickness[i].item()
                count[idx] += 1
            nodal_thickness /= count

            top = mesh.copy()
            top.points += 0.5 * nodal_thickness[:, None] * mesh.point_normals
            bottom = mesh.copy()
            bottom.points -= 0.5 * nodal_thickness[:, None] * mesh.point_normals

            pl.add_mesh(top, show_edges=True)
            pl.add_mesh(bottom, show_edges=True)
            pl.show()
        else:
            mesh.plot(show_edges=True)
