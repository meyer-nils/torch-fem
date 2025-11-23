"""Shell formulation

The Shell element formulation is based on:
Krysl, Petr, Robust flat-facet triangular shell finite element, International Journal
for Numerical Methods in Engineering, vol. 123, issue 10, pp. 2399-2423, 2022
https://doi.org/10.1002/nme.6944
"""

from math import sqrt
from typing import Tuple

import torch
from torch import Tensor

from .base import Mechanics
from .elements import Element, Tria1
from .materials import Material
from .utils import stiffness2voigt


class Shell(Mechanics):
    def __init__(
        self,
        nodes: Tensor,
        elements: Tensor,
        material: Material,
        transverse_nu: float = 0.5,
        transverse_kappa: float = 5.0 / 6.0,
        transverse_G: list[float] | list[Tensor] | None = None,
        drill_penalty: float = 1.0,
        n_simpson: int = 3,
    ):
        """Initialize the planar FEM problem."""

        super().__init__(nodes, elements, material)

        # Set up thickness
        self.thickness = torch.ones(self.n_elem)

        # Drill penalty
        self.drill_penalty = drill_penalty

        # Thickness integration points
        if n_simpson % 2 == 0:
            raise ValueError("n_simpson must be an odd integer.")
        else:
            self.n_simpson = n_simpson

        # Update number of integration points to account for thickness integration
        self.n_int = self.n_int * n_simpson

        # Compute Simpson points in thickness direction
        self.z_simpson = torch.linspace(-0.5, 0.5, n_simpson)

        # Simpson weights for thickness integration
        self.w_simpson = torch.ones(n_simpson)
        self.w_simpson[1:-1:2] = 4.0
        self.w_simpson[2:-2:2] = 2.0
        self.w_simpson /= 3.0

        # Transverse shear properties
        self.transverse_nu = transverse_nu
        self.transverse_kappa = transverse_kappa
        z = torch.zeros(self.n_elem)
        if transverse_G is None:
            if hasattr(self.material, "G"):
                self.G = [self.material.G, self.material.G]  # type: ignore
            else:
                raise ValueError(
                    "Material must have shear modulus 'G' defined or "
                    "transverse_G must be provided."
                )
        else:
            self.G = [
                torch.as_tensor(transverse_G[0]).repeat(self.n_elem),
                torch.as_tensor(transverse_G[1]).repeat(self.n_elem),
            ]
        self.Cs = torch.stack(
            [
                torch.stack([self.G[0], z], dim=-1),
                torch.stack([z, self.G[1]], dim=-1),
            ],
            dim=-1,
        )

    def __repr__(self) -> str:
        etype = self.etype.__class__.__name__
        return f"<torch-fem shell ({self.n_nod} nodes, {self.n_elem} {etype} elements)>"

    @property
    def n_dof_per_node(self) -> int:
        """Number of DOFs per node"""
        return 6

    @property
    def n_flux(self) -> list[int]:
        """Shape of the stress tensor."""
        return [2, 2]

    @property
    def etype(self) -> type[Element]:
        """Set element type."""
        return Tria1

    @property
    def char_lengths(self) -> Tensor:
        """Characteristic lengths of the elements."""
        areas = self.integrate_field()
        return areas ** (1 / 2)

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

    def eval_shape_functions(
        self, xi: Tensor, u: Tensor | float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gradient operator at integration points xi."""
        # Compute transformation matrix x = T X with element coords x and
        # global coords X
        nodes = self.nodes + u
        nodes = nodes[self.elements, :]
        edge1 = nodes[:, 1] - nodes[:, 0]
        edge2 = nodes[:, 2] - nodes[:, 1]
        dir1 = torch.nn.functional.normalize(edge1, dim=-1)
        normal = torch.nn.functional.normalize(torch.linalg.cross(edge1, edge2), dim=-1)
        dir2 = torch.nn.functional.normalize(-torch.linalg.cross(edge1, normal), dim=-1)
        self.t = torch.stack([dir1, dir2, normal], dim=1)
        self.T = torch.func.vmap(torch.block_diag)(*(self.n_dof_per_node * [self.t]))

        # Compute Jacobian and its determinant
        b = self.etype.B(xi)
        dx = (nodes - nodes[:, 0, None]).transpose(2, 1)
        self.loc_nodes = (self.t[:, 0:2, :] @ dx).transpose(2, 1)
        J = b @ self.loc_nodes
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")

        # Compute B
        B = torch.linalg.inv(J) @ b

        return self.etype.N(xi), B, detJ

    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        raise NotImplementedError

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        raise NotImplementedError

    def integrate_material(
        self,
        u: Tensor,
        grad: Tensor,
        flux: Tensor,
        state: Tensor,
        n: int,
        iter: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix."""

        # Mechanical interpretation of variables
        F = grad
        stress = flux

        # Compute updated configuration
        u_trial = u[n - 1] + du.view((-1, self.n_dof_per_node))

        # Reshape displacement increment and rotation increment
        du = du.view(-1, self.n_dof_per_node)[self.elements]
        d_u = du[..., :3]
        d_w = du[..., 3:]

        # Initialize nodal force and stiffness
        N_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dof_per_node * N_nod)
        k = torch.zeros(
            (self.n_elem, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod)
        )

        if nlgeom:
            raise NotImplementedError(
                "Geometric nonlinearity is not yet implemented for shells."
            )

        for i, (wi, xi) in enumerate(zip(self.etype.iweights, self.etype.ipoints)):
            # Compute gradient operators
            _, B, detJ = self.eval_shape_functions(xi)

            # Transform displacement increment to local element coordinates
            du_local = torch.einsum("...ij,...kj->...ki", self.t, d_u)
            dw_local = torch.einsum("...ij,...kj->...ki", self.t, d_w)

            f_loc = torch.zeros(self.n_elem, *self.n_flux)
            m_loc = torch.zeros(self.n_elem, *self.n_flux)

            for j, (wz, z) in enumerate(zip(self.w_simpson, self.z_simpson)):
                # Compute integration point index
                ip = i * self.n_simpson + j
                z = z * self.thickness[:, None, None]

                # Compute total strain increment
                dudxi = B @ du_local
                dwdxi = B @ dw_local
                # Build Koiter bending strain operator
                kappa = torch.stack(
                    [
                        torch.stack([dwdxi[..., 0, 1], -dwdxi[..., 0, 0]], dim=-1),
                        torch.stack([dwdxi[..., 1, 1], -dwdxi[..., 1, 0]], dim=-1),
                    ],
                    dim=-1,
                )

                H_inc = dudxi[..., 0:2] + z * kappa

                # Evaluate material response
                stress[n, ip], state[n, ip], ddsdde = self.material.step(
                    H_inc,
                    F[n - 1, ip],
                    stress[n - 1, ip],
                    state[n - 1, ip],
                    de0,
                    self.char_lengths,
                    iter,
                )

                # Compute local internal forces
                f_loc += wz / 2 * stress[n, ip].clone()
                m_loc += wz / 2 * z * stress[n, ip].clone()

            # Material stiffness
            C = stiffness2voigt(ddsdde)
            Cs = self.Cs

            # Element membrane stiffness
            Dm = self._Dm(B)
            DmCDm = torch.einsum("...ji,...jk,...kl->...il", Dm, C, Dm)
            km = torch.einsum("i,ijk->ijk", wi * self.thickness * detJ, DmCDm)

            # Element bending stiffness
            Db = self._Db(B)
            DbCDb = torch.einsum("...ji,...jk,...kl->...il", Db, C, Db)
            kb = torch.einsum("i,ijk->ijk", wi * self.thickness**3 * detJ / 12.0, DbCDb)

            # Element transverse stiffness
            A = detJ / 2.0
            Ds = self._Ds(A)
            h = sqrt(2) * A
            alpha = self.transverse_kappa / (2 * (1 + self.transverse_nu))
            psi = (
                self.transverse_kappa
                * self.thickness**2
                / (self.thickness**2 + alpha * h**2)
            )
            DsCsDs = torch.einsum("...ji,...jk,...kl->...il", Ds, Cs, Ds)
            ks = torch.einsum(
                "i,ijk->ijk", wi * A * psi * self.thickness * detJ, DsCsDs
            )

            # Element drilling stiffness
            kd = torch.zeros_like(km)
            for i in range(self.etype.nodes):
                kd[:, i * self.n_dof_per_node - 1, i * self.n_dof_per_node - 1] = (
                    self.drill_penalty
                )

            # Total element stiffness in local coordinates
            kt = km + kb + ks + kd

            # Total element stiffness in global coordinates
            k[:, :, :] += self.T.transpose(1, 2) @ kt @ self.T

            # Total force contribution
            disp = u_trial[self.elements, :].reshape(self.n_elem, -1)
            loc_disp = torch.einsum("...ij,...j->...i", self.T, disp)
            f[:, :] += torch.einsum(
                "...ki, ...ij,...j->...k", self.T.transpose(1, 2), kt, loc_disp
            )
            # temp_f = torch.einsum("...ij,...j->...i", km + kb, loc_disp)
            # print(temp_f[0])

            # fv = torch.stack(
            #     [f_loc[:, 0, 0], f_loc[:, 1, 1], f_loc[:, 0, 1]],
            #     dim=-1,
            # )
            # mv = torch.stack(
            #     [m_loc[:, 0, 0], m_loc[:, 1, 1], m_loc[:, 0, 1]],
            #     dim=-1,
            # )
            # fm = torch.einsum("...ji,...j->...i", Dm, fv)
            # fb = torch.einsum("...ji,...j->...i", Db, mv)
            # force = wi * self.thickness[:, None] * detJ[:, None] * (fm + fb)
            # print(force[0])

        return k, f

    def compute_stress(
        self,
        u: Tensor,
        xi: Tensor = torch.tensor([0.0, 0.0]),
        z: float = 0,
        mises: bool = False,
    ):
        # Extract displacement degrees of freedom
        disp = u[self.elements, :].reshape(self.n_elem, -1)

        # Jacobian
        J = self.etype.B(xi) @ self.loc_nodes
        A = torch.linalg.det(J) / 2.0

        # Compute B
        B = torch.linalg.inv(J) @ self.etype.B(xi)

        # Evaluate material stiffness
        _, _, ddsdde = self.material.step(
            torch.zeros(self.n_elem, *self.n_flux),
            torch.zeros(self.n_elem, *self.n_flux),
            torch.zeros(self.n_elem, *self.n_flux),
            torch.zeros(self.n_elem, self.material.n_state),
            torch.zeros(self.n_elem, *self.n_flux),
            self.char_lengths,
            0,
        )
        C = stiffness2voigt(ddsdde)
        Cs = self.Cs

        # Compute in-plane stresses in local coordinate system
        loc_disp = torch.einsum("...ij,...j->...i", self.T, disp)
        sigma_m = torch.einsum("...ij,...jk,...k->...i", C, self._Dm(B), loc_disp)
        sigma_b = torch.einsum("...ij,...jk,...k->...i", C, self._Db(B), loc_disp)
        sigma = sigma_m + z * sigma_b

        # Compute transverse shear stresses in local coordinate system
        h = sqrt(2) * A
        alpha = self.transverse_kappa / (2 * (1 + self.transverse_nu))
        psi = (
            self.transverse_kappa
            * self.thickness**2
            / (self.thickness**2 + alpha * h**2)
        )
        Cs = torch.einsum("...,...jk->...jk", psi, Cs)
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
        # S = self.t.transpose(1, 2) @ stress_tensor @ self.t
        S = stress_tensor

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
    def plot(
        self,
        u: float | Tensor = 0.0,
        node_property: dict[str, Tensor] | None = None,
        element_property: dict[str, Tensor] | None = None,
        thickness: bool = False,
        mirror: list[bool] = [False, False, False],
        **kwargs,
    ):
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
                mesh.point_data[key] = val.cpu().numpy()

        # Plot cell properties
        if element_property:
            for key, val in element_property.items():
                mesh.cell_data[key] = val.cpu().numpy()

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
        else:
            pl.add_mesh(mesh, show_edges=True)

        if mirror[0]:
            for msh in pl.meshes:
                pl.add_mesh(msh.reflect((1, 0, 0)), show_edges=True, opacity=0.5)

        if mirror[1]:
            for msh in pl.meshes:
                pl.add_mesh(msh.reflect((0, 1, 0)), show_edges=True, opacity=0.5)

        if mirror[2]:
            for msh in pl.meshes:
                pl.add_mesh(msh.reflect((0, 0, 1)), show_edges=True, opacity=0.5)
        pl.show(jupyter_backend="html")
