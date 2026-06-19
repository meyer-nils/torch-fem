"""Shell formulation

The Shell element formulation is based on:
Krysl, Petr, Robust flat-facet triangular shell finite element, International Journal
for Numerical Methods in Engineering, vol. 123, issue 10, pp. 2399-2423, 2022
https://doi.org/10.1002/nme.6944
"""

from functools import cached_property
from math import sqrt
from typing import Tuple

import torch
from torch import Tensor

from .base import Mechanics
from .elements import Element, Tria1
from .laminate import Laminate
from .materials import Material
from .utils import stiffness2voigt, stress2voigt


class Shell(Mechanics):
    def __init__(
        self,
        nodes: Tensor,
        elements: Tensor,
        material: Material | Laminate,
        thickness: Tensor | float = 1.0,
        transverse_nu: float = 0.5,
        transverse_kappa: float = 5.0 / 6.0,
        transverse_G: list[float] | list[Tensor] | None = None,
        drill_penalty: float = 1.0,
        n_simpson: int = 3,
    ):
        """Initialize the shell FEM problem.

        Args:
            material: Either a single plane-stress `Material` (homogeneous
                shell) or a `Laminate` describing a layered stacking sequence.
                When a `Laminate` is passed, `thickness` and `n_simpson` are
                taken from the laminate and the corresponding arguments here are
                ignored.
        """

        super().__init__(nodes, elements, material)

        # Drill penalty
        self.drill_penalty = drill_penalty

        # Transverse shear properties
        self.transverse_nu = transverse_nu
        self.transverse_kappa = transverse_kappa

        if isinstance(self.material, Laminate):
            # Layered shell: take thickness, stations, and shear from laminate.
            self.n_simpson = self.material.n_simpson
            self.n_z = self.material.n_z
            self.thickness = self.material.thickness
            if transverse_G is None:
                self.As = self.material.As
            else:
                self.As = self._build_As(transverse_G)
        else:
            # Homogeneous shell (unchanged behavior).
            if isinstance(thickness, float):
                self.thickness = torch.full((self.n_elem,), thickness)
            else:
                self.thickness = torch.as_tensor(thickness)

            # Thickness integration points
            if n_simpson % 2 == 0:
                raise ValueError("n_simpson must be an odd integer.")
            self.n_simpson = n_simpson
            self.n_z = n_simpson

            # Simpson points (normalized) and weights (summing to 1)
            self.z_simpson = torch.linspace(-0.5, 0.5, n_simpson)
            self.w_simpson = torch.ones(n_simpson)
            self.w_simpson[1:-1:2] = 4.0
            self.w_simpson[2:-2:2] = 2.0
            self.w_simpson *= 1.0 / (n_simpson - 1) / 3.0

            # Effective through-thickness transverse shear stiffness
            if transverse_G is None:
                if hasattr(self.material, "G"):
                    G = [self.material.G, self.material.G]  # type: ignore
                else:
                    raise ValueError(
                        "Material must have shear modulus 'G' defined or "
                        "transverse_G must be provided."
                    )
                z = torch.zeros(self.n_elem)
                Cs = torch.stack(
                    [
                        torch.stack([G[0], z], dim=-1),
                        torch.stack([z, G[1]], dim=-1),
                    ],
                    dim=-1,
                )
                self.As = self.thickness[:, None, None] * Cs
            else:
                self.As = self._build_As(transverse_G)

        # Update number of integration points to account for thickness
        # integration over the through-thickness stations.
        self.n_int = self.n_int * self.n_z

    def _build_As(self, transverse_G: list[float] | list[Tensor]) -> Tensor:
        """Build the integrated transverse shear stiffness from a user override.

        Args:
            transverse_G: Pair ``[G_xz, G_yz]`` of effective transverse shear
                moduli. The values are integrated over the total thickness.

        Returns:
            Tensor of shape `(n_elem, 2, 2)`.
        """
        g0 = torch.as_tensor(transverse_G[0]).repeat(self.n_elem)
        g1 = torch.as_tensor(transverse_G[1]).repeat(self.n_elem)
        z = torch.zeros(self.n_elem)
        Cs = torch.stack(
            [
                torch.stack([g0, z], dim=-1),
                torch.stack([z, g1], dim=-1),
            ],
            dim=-1,
        )
        return self.thickness[:, None, None] * Cs

    def _thickness_stations(self) -> tuple[list[Material], Tensor, Tensor]:
        """Through-thickness integration stations.

        Returns:
            Tuple ``(materials, z, w)`` where ``materials`` is a list of length
            ``n_z`` giving the material active at each station, ``z`` has shape
            `(n_z, n_elem)` with absolute through-thickness coordinates, and
            ``w`` has shape `(n_z, n_elem)` with absolute integration weights
            (such that ``sum_j w_j f_j`` approximates ``integral f dz``).
        """
        if isinstance(self.material, Laminate):
            return self.material.materials_per_station, self.material.z, self.material.w
        else:
            z = self.z_simpson[:, None] * self.thickness[None, :]
            w = self.w_simpson[:, None] * self.thickness[None, :]
            return [self.material] * self.n_simpson, z, w

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

    @cached_property
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
            torch tensor: Shear-displacement matrices shaped [N x 2 x 18]
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

    def eval_shape_functions(self, xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Gradient operator at integration points xi."""
        # Compute transformation matrix x = T X with element coords x and
        # global coords X
        nodes = self.nodes[self.elements, :]
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
        J = torch.einsum("...iN, ANj -> ...Aij", b, self.loc_nodes)
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")

        # Compute B
        B = torch.linalg.inv(J) @ b

        return self.etype.N(xi), B, detJ

    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        return torch.einsum("i,ijk->ijk", detJ, BCB)

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        raise NotImplementedError

    def integrate_mass(self) -> Tensor:
        """Mass matrix (translational: ∫ρ dz, rotational: ∫ρz² dz)."""
        n = self.n_dof_per_node * self.etype.nodes
        m = torch.zeros((self.n_elem, n, n))
        if isinstance(self.material, Laminate):
            rho_trans = self.material.rho_h
            rho_rot = self.material.rho_zz
        else:
            rho_trans = self.material.rho * self.thickness
            rho_rot = self.material.rho * self.thickness**3 / 12
        D = torch.diag_embed(torch.stack([rho_trans] * 3 + [rho_rot] * 3, dim=-1))
        N, _, detJ = self.eval_shape_functions(self.etype.ipoints)
        for i, w in enumerate(self.etype.iweights):
            m_loc = w * torch.einsum(
                "ENM,Eij->ENiMj", torch.einsum("N,M,E->ENM", N[i], N[i], detJ[i]), D
            )
            m_loc = m_loc.reshape(self.n_elem, n, n)
            m += self.T.transpose(1, 2) @ m_loc @ self.T
        return m

    def integrate_material(
        self,
        u_prev: Tensor,
        grad_prev: Tensor,
        flux_prev: Tensor,
        state_prev: Tensor,
        du: Tensor,
        de0: Tensor,
        iter: int,
        nlgeom: bool,
        compute_stiffness: bool = True,
    ) -> Tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix.

        Args:
            grad_prev: Deformation gradient at previous step [n_int, n_elem, *n_flux]
            flux_prev: Stress at previous step [n_int, n_elem, *n_flux]
            state_prev: Material state at previous step [n_int, n_elem, n_state]
            compute_stiffness: If True, assemble and return element stiffness.

        Returns:
            k, f, grad_new, flux_new, state_new
        """

        # Initialize output for new state
        grad_new = torch.zeros_like(grad_prev)
        flux_new = torch.zeros_like(flux_prev)
        state_new = torch.zeros_like(state_prev)

        # Compute updated configuration
        u_trial = u_prev + du.view((-1, self.n_dof_per_node))

        # Reshape displacement increment and rotation increment
        du = du.view(-1, self.n_dof_per_node)[self.elements]
        d_u = du[..., :3]
        d_w = du[..., 3:]

        # Initialize nodal force and stiffness
        N_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dof_per_node * N_nod)
        need_k = compute_stiffness and (
            self.K.numel() == 0 or self.material.n_state != 0 or nlgeom
        )
        k = (
            torch.zeros(
                (
                    self.n_elem,
                    self.n_dof_per_node * N_nod,
                    self.n_dof_per_node * N_nod,
                )
            )
            if need_k
            else None
        )

        if nlgeom:
            raise NotImplementedError(
                "Geometric nonlinearity is not yet implemented for shells."
            )

        # Compute gradient operators
        _, B, detJ = self.eval_shape_functions(self.etype.ipoints)

        # Through-thickness integration stations (layer-aware for laminates)
        materials, z_stations, w_stations = self._thickness_stations()

        for i, wi in enumerate(self.etype.iweights):

            # Transform displacement increment to local element coordinates
            du_local = torch.einsum("...ij,...kj->...ki", self.t, d_u)
            dw_local = torch.einsum("...ij,...kj->...ki", self.t, d_w)

            # Initialize local force contributions
            f_loc = torch.zeros(self.n_elem, *self.n_flux)
            m_loc = torch.zeros(self.n_elem, *self.n_flux)

            # Initialize ABD matrices
            A_matrix = torch.zeros((self.n_elem, 3, 3))
            B_matrix = torch.zeros((self.n_elem, 3, 3))
            D_matrix = torch.zeros((self.n_elem, 3, 3))

            # Thickness integration of membrane and bending stresses
            for j, material in enumerate(materials):
                # Compute integration point index
                ip = i * self.n_z + j

                # Absolute through-thickness position and integration weight
                z = z_stations[j][:, None, None]
                wz = w_stations[j][:, None, None]

                # Compute gradient of displacement increment and rotation increment
                dudxi = B[i] @ du_local
                dwdxi = B[i] @ dw_local

                # Compute curvature
                dkappa = torch.stack(
                    [
                        torch.stack([dwdxi[..., 0, 1], -dwdxi[..., 0, 0]], dim=-1),
                        torch.stack([dwdxi[..., 1, 1], -dwdxi[..., 1, 0]], dim=-1),
                    ],
                    dim=-1,
                )

                # Compute in-plane displacement gradient increment
                H_inc = dudxi[..., 0:2] + z * dkappa

                # Evaluate material response
                flux_new[ip], state_new[ip], ddsdde = material.step(
                    H_inc,
                    grad_prev[ip],
                    flux_prev[ip],
                    state_prev[ip],
                    de0,
                    self.char_lengths,
                    iter,
                )

                # Thickness integration of membrane forces and bending moments.
                f_loc += wz * flux_new[ip].clone()
                m_loc += wz * z * flux_new[ip].clone()

                # Compute ABD matrix contributions
                C = stiffness2voigt(ddsdde)
                A_matrix += C * wz
                B_matrix += C * wz * z
                D_matrix += C * wz * z**2

            # Copy grad from grad_prev (shells don't update deformation gradient)
            grad_new[:] = grad_prev

            # Element membrane stiffness
            Dm = self._Dm(B[i])
            DmCDm = torch.einsum("...ji,...jk,...kl->...il", Dm, A_matrix, Dm)
            km = wi * self.compute_k(detJ[i], DmCDm)

            # Element bending stiffness
            Db = self._Db(B[i])
            DbCDb = torch.einsum("...ji,...jk,...kl->...il", Db, D_matrix, Db)
            kb = wi * self.compute_k(detJ[i], DbCDb)

            # Element transverse stiffness
            A = detJ[i] / 2.0
            h = sqrt(2) * A
            alpha = self.transverse_kappa / (2 * (1 + self.transverse_nu))
            psi = (
                self.transverse_kappa
                * self.thickness**2
                / (self.thickness**2 + alpha * h**2)
            )
            Ds = self._Ds(A)
            int_Cs = (A * psi)[:, None, None] * self.As
            DsCsDs = torch.einsum("...ji,...jk,...kl->...il", Ds, int_Cs, Ds)
            ks = wi * self.compute_k(detJ[i], DsCsDs)

            # Element drilling stiffness
            kd = torch.zeros_like(km)
            for a in range(self.etype.nodes):
                kd[:, a * self.n_dof_per_node - 1, a * self.n_dof_per_node - 1] = (
                    self.drill_penalty
                )

            if k is not None:
                # Total element stiffness in local coordinates
                kt = km + kb + ks + kd

                # Total element stiffness in global coordinates
                k[:, :, :] += self.T.transpose(1, 2) @ kt @ self.T

            # Total force contribution
            disp = u_trial[self.elements, :].reshape(self.n_elem, -1)
            loc_disp = torch.einsum("...ij,...j->...i", self.T, disp)
            n_loc = stress2voigt(f_loc)
            m_loc_voigt = stress2voigt(m_loc)
            f_membrane = wi * torch.einsum("...,...ji,...j->...i", detJ[i], Dm, n_loc)
            f_bending = wi * torch.einsum(
                "...,...ji,...j->...i", detJ[i], Db, m_loc_voigt
            )
            f_shear_drill = torch.einsum("...ij,...j->...i", ks + kd, loc_disp)
            f_loc_total = f_membrane + f_bending + f_shear_drill
            f[:, :] += torch.einsum(
                "...ij,...j->...i", self.T.transpose(1, 2), f_loc_total
            )

        return k, f, grad_new, flux_new, state_new

    @torch.no_grad()
    def plot(
        self,
        u: float | Tensor = 0.0,
        node_property: dict[str, Tensor] | None = None,
        element_property: dict[str, Tensor] | None = None,
        thickness: bool = False,
        mirror: tuple[bool, bool, bool] = (False, False, False),
        screenshot: str | None = None,
        **kwargs,
    ):
        try:
            import numpy as np
            import pyvista
        except ImportError:
            raise Exception("Plotting 3D requires pyvista.")

        pyvista.set_plot_theme("document")
        off_screen = screenshot is not None
        if not off_screen:
            pyvista.set_jupyter_backend("client")
        pl = pyvista.Plotter(off_screen=off_screen)
        pl.enable_anti_aliasing("ssaa")

        # VTK element list
        elements = []
        for element in self.elements.cpu().numpy():
            elements += [len(element), *element]

        # Deformed node positions
        pos = (self.nodes + u).cpu().numpy()

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

        # Plot as separate top and bottom surface
        base_meshes = []
        if thickness:
            nodal_thickness = np.zeros((len(self.nodes)))
            count = np.zeros((len(self.nodes)))
            for i, face in enumerate(mesh.faces.reshape(-1, 4)):
                idx = face[1::]
                nodal_thickness[idx] += self.thickness[i].cpu().item()
                count[idx] += 1
            nodal_thickness /= count

            top = mesh.copy()
            top.points += 0.5 * nodal_thickness[:, None] * mesh.point_normals
            bottom = mesh.copy()
            bottom.points -= 0.5 * nodal_thickness[:, None] * mesh.point_normals

            pl.add_mesh(top, show_edges=True)
            pl.add_mesh(bottom, show_edges=True)
            base_meshes.extend([top, bottom])
        else:
            pl.add_mesh(mesh, show_edges=True)
            base_meshes.append(mesh)

        # Mirror meshes across specified planes
        sx_values = [1.0, -1.0] if mirror[0] else [1.0]
        sy_values = [1.0, -1.0] if mirror[1] else [1.0]
        sz_values = [1.0, -1.0] if mirror[2] else [1.0]
        for sx in sx_values:
            for sy in sy_values:
                for sz in sz_values:
                    if sx == 1.0 and sy == 1.0 and sz == 1.0:
                        continue
                    for msh in base_meshes:
                        mirrored = msh.copy()
                        mirrored.points[:, 0] *= sx
                        mirrored.points[:, 1] *= sy
                        mirrored.points[:, 2] *= sz
                        pl.add_mesh(mirrored, show_edges=True, opacity=0.5)
        if screenshot:
            pl.screenshot(screenshot)
        else:
            pl.show(jupyter_backend="html")
