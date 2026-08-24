"""Shell formulation

The triangular element is based on:
Krysl, Petr, Robust flat-facet triangular shell finite element, International Journal
for Numerical Methods in Engineering, vol. 123, issue 10, pp. 2399-2423, 2022
https://doi.org/10.1002/nme.6944

The quadrilateral element uses the transverse shear interpolation of:
Dvorkin, Eduardo N. and Bathe, Klaus-Juergen, A continuum mechanics based four-node
shell element for general non-linear analysis, Engineering Computations, vol. 1,
issue 1, pp. 77-88, 1984
https://doi.org/10.1108/eb023562
"""

from functools import cached_property
from math import sqrt
from typing import cast

import numpy as np
import pyvista
import torch
from pyvista.plotting import CameraPositionOptions
from torch import Tensor

from .base import Mechanics
from .elements import Element, Quad1, Tria1
from .laminate import Laminate
from .materials import Material
from .plot_utils import arrows, cones, dots
from .utils import stiffness2voigt, stress2voigt


class Shell(Mechanics):
    """Flat-facet shell model for thin-walled structures.

    Triangles follow Krysl, quadrilaterals the MITC4 shear interpolation of Dvorkin
    and Bathe on the mean plane of their four nodes, so warp is neglected. Each node
    carries six degrees of freedom (three translations, three rotations). The section
    is either a homogeneous plane-stress material with a thickness or a layered
    `Laminate`.

    Attributes:
        nodes: Nodal coordinates with shape [n_nod, 3].
        elements: Triangle or quadrilateral connectivity with shape [n_elem, 3]
            or [n_elem, 4].
        material: Vectorized plane-stress material (None for laminate shells).
        section: Laminate section (None for homogeneous shells).
        thickness: Element thicknesses with shape [n_elem].
        orientation: Per-element material reference direction with shape
            [n_elem, 3].
        forces: Applied nodal forces and moments with shape [n_nod, 6].
        displacements: Prescribed nodal displacements and rotations with
            shape [n_nod, 6].
        constraints: Boolean mask of constrained DOFs with shape [n_nod, 6].
    """

    supports_nlgeom = False

    def __init__(
        self,
        nodes: Tensor,
        elements: Tensor,
        material: Material | Laminate,
        thickness: Tensor | float = 1.0,
        transverse_nu: float = 0.5,
        transverse_kappa: float = 5.0 / 6.0,
        transverse_G: list[float] | list[Tensor] | None = None,
        drill_penalty: float = 1e-3,
        n_simpson: int = 3,
        orientation: Tensor | None = None,
    ):
        """Initialize the shell FEM problem.

        Args:
            nodes: Nodal coordinates with shape [n_nod, 3].
            elements: Triangle or quadrilateral connectivity with shape
                [n_elem, 3] or [n_elem, 4].
            material: Either a single plane-stress `Material` (homogeneous
                shell) or a `Laminate` describing a layered stacking sequence.
                When a `Laminate` is passed, `thickness` and `n_simpson` are
                taken from the laminate and the corresponding arguments here are
                ignored.
            thickness: Shell thickness. A float is expanded to all elements, a
                tensor assigns one thickness per element.
            transverse_nu: Poisson's ratio used for the shear relaxation of a
                homogeneous triangle. A quadrilateral needs none.
            transverse_kappa: Shear correction factor, 5/6 for a homogeneous
                section.
            transverse_G: Pair `[G_xz, G_yz]` of effective transverse shear
                moduli, integrated over the thickness. Taken from the material
                or the laminate when omitted.
            drill_penalty: Stiffness of the drilling degree of freedom, which the
                flat-facet element does not carry itself, as a fraction of the
                shear stiffness of the section.
            n_simpson: Number of Simpson integration points through the
                thickness. Must be an odd integer.
            orientation: Global reference direction from which material/ply
                angles are measured. It is projected onto each element's surface
                to define the element's local material 0°-axis. Accepts
                a single `(3,)` vector (shared by all elements) or a per-element
                `(n_elem, 3)` tensor. Defaults to the global x-direction.
        """

        # A Laminate is the shell's section, not a pointwise material: keep it
        # in self.section and give the base no material.
        if isinstance(material, Laminate):
            super().__init__(nodes, elements, None)
            self.section: Laminate | None = (
                material if material.is_vectorized else material.vectorize(self.n_elem)
            )
        else:
            super().__init__(nodes, elements, material)
            self.section = None

        # Material reference orientation
        if orientation is None:
            orientation = torch.tensor([1.0, 0.0, 0.0])
        orientation = torch.as_tensor(orientation, dtype=self.nodes.dtype)
        if orientation.dim() == 1:
            orientation = orientation.unsqueeze(0).expand(self.n_elem, 3)
        self.orientation = orientation

        # Drill penalty
        self.drill_penalty = drill_penalty

        # Transverse shear properties
        self.transverse_nu = transverse_nu
        self.transverse_kappa = transverse_kappa

        if self.section is not None:
            # Layered shell: take thickness, stations, and shear from the section.
            self.n_simpson = self.section.n_simpson
            self.n_z = self.section.n_z
            self.thickness = self.section.thickness
            if transverse_G is None:
                self.As = self.section.As
            else:
                self.As = self._build_As(transverse_G)
        else:
            # Homogeneous shell (unchanged behavior).
            assert self.material is not None
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
                # getattr keeps these dynamic: the shear moduli exist only on
                # some Material subclasses, which is what the hasattr guards test.
                if hasattr(self.material, "G"):
                    G = 2 * [getattr(self.material, "G")]  # noqa: B009
                elif hasattr(self.material, "G_13") and hasattr(self.material, "G_23"):
                    G = [
                        getattr(self.material, "G_13"),  # noqa: B009
                        getattr(self.material, "G_23"),  # noqa: B009
                    ]
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
        if self.section is not None:
            return self.section.materials_per_station, self.section.z, self.section.w
        else:
            assert self.material is not None
            z = self.z_simpson[:, None] * self.thickness[None, :]
            w = self.w_simpson[:, None] * self.thickness[None, :]
            return [self.material] * self.n_simpson, z, w

    def __repr__(self) -> str:
        etype = self.etype.__name__
        return f"<torch-fem shell ({self.n_nod} nodes, {self.n_elem} {etype} elements)>"

    @property
    def n_state(self) -> int:
        """Number of internal state variables per through-thickness station."""
        if self.section is not None:
            return self.section.n_state
        assert self.material is not None
        return self.material.n_state

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
        """Set element type depending on number of nodes per element."""
        if len(self.elements[0]) == 3:
            return Tria1
        elif len(self.elements[0]) == 4:
            return Quad1
        else:
            raise ValueError("Element type not supported.")

    @property
    def volume_scale(self) -> Tensor:
        return self.thickness

    def integrate_surface_load(self, mask: Tensor, load: float | Tensor) -> Tensor:
        """Consistent nodal loads from a load per unit area, e.g. a pressure.

        A shell element is its own surface, so the loaded surface is made up of the
        elements whose nodes all lie in `mask`.

        Args:
            mask: Boolean nodal mask with shape [n_nod] selecting the surface.
            load: Load per unit area. A float is a pressure acting along the element
                normal, while shape [3] or [n_elem, 3] is a traction in global
                coordinates.

        Returns:
            Nodal loads with shape [n_nod, 3], to be added to `forces[:, 0:3]`.
        """
        conn = self.elements[mask[self.elements].all(dim=1)]
        return self._integrate_facet_load(
            conn, self.etype, torch.as_tensor(load, dtype=self.nodes.dtype)
        )

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

    def _Ds(self, detJ: Tensor) -> Tensor:
        """Aggregate shear-displacement matrices.

        A triangle averages the operator of Krysl over the three node orderings,
        while a quadrilateral ties its covariant shear strains to the midpoints of
        the edges (MITC4).

        Args:
            detJ (torch tensor): Jacobian determinants (shape: [n_ip x N])

        Returns:
            torch tensor: Shear-displacement matrices shaped
                [n_ip x N x 2 x 6*nodes]
        """
        if self.etype is Quad1:
            xi = self.etype.ipoints.to(self.loc_nodes)
            # Tying points at the midpoints of the edges 0-1, 1-2, 2-3 and 3-0
            mid = [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
            tie = torch.tensor(mid).to(xi)
            b = self.etype.B(tie)
            J = torch.einsum("...iN, ANj -> ...Aij", b, self.loc_nodes)

            # Covariant shear g_a = w_,a + N (x_,a th_y - y_,a th_x), where J B = b
            w = b[:, None].expand(-1, self.n_elem, -1, -1)
            Nt = self.etype.N(tie)[:, None, None, :]
            zq = torch.zeros_like(w)
            cols = [zq, zq, w, -Nt * J[..., 1:2], Nt * J[..., 0:1], zq]
            g = torch.stack(cols, dim=-1).reshape(len(tie), self.n_elem, 2, -1)

            # Interpolate between the tying points and pull back to the element frame
            r, s = xi[:, 0, None, None], xi[:, 1, None, None]
            g_xi = 0.5 * ((1.0 - s) * g[0, :, 0] + (1.0 + s) * g[2, :, 0])
            g_eta = 0.5 * ((1.0 - r) * g[3, :, 1] + (1.0 + r) * g[1, :, 1])
            b_ip = self.etype.B(xi)
            J_ip = torch.einsum("...iN, ANj -> ...Aij", b_ip, self.loc_nodes)
            return torch.linalg.inv(J_ip) @ torch.stack([g_xi, g_eta], dim=2)
        else:
            # A triangle has a constant Jacobian, so one operator serves every point
            N = self.n_elem
            A = detJ[0] / 2.0
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
            return torch.cat([D0, D1, D2], dim=-1).expand(len(detJ), -1, -1, -1)

    def _shear_correction(self, detJ: Tensor) -> Tensor:
        """Shear correction factor scaling the shear stiffness of the section.

        A triangle reduces it by the relaxation of Krysl to avoid locking, with `h`
        the element edge length, which the MITC4 tying of a quadrilateral does not
        need.

        Args:
            detJ (torch tensor): Jacobian determinants (shape: [n_ip x N])

        Returns:
            torch tensor: Shear correction factors shaped [n_ip x N]
        """
        if self.etype is Quad1:
            return torch.full_like(detJ, self.transverse_kappa)
        else:
            h = sqrt(2) * torch.sqrt(detJ / 2.0)
            alpha = self.transverse_kappa / (2 * (1 + self.transverse_nu))
            t2 = self.thickness**2
            return self.transverse_kappa * t2 / (t2 + alpha * h**2)

    def eval_shape_functions(self, xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Gradient operator at integration points xi."""
        # Compute transformation matrix x = T X with element coords x and
        # global coords X
        nodes = self.nodes[self.elements, :]
        edge1 = nodes[:, 1] - nodes[:, 0]
        # Mean plane normal (Newell's formula), which averages the warp of a quad
        rel = nodes - nodes.mean(dim=1, keepdim=True)
        area = torch.linalg.cross(rel, rel.roll(-1, dims=1), dim=-1).sum(dim=1)
        normal = torch.nn.functional.normalize(area, dim=-1)
        # Material x-axis: the global reference orientation projected onto the
        # element surface. Fall back to the first edge where the orientation is
        # (nearly) normal to the element and the projection vanishes.
        o = self.orientation
        proj = o - (o * normal).sum(dim=-1, keepdim=True) * normal
        degen = (proj.norm(dim=-1) < 1e-8).unsqueeze(-1)
        dir1 = torch.nn.functional.normalize(torch.where(degen, edge1, proj), dim=-1)
        dir2 = torch.nn.functional.normalize(torch.linalg.cross(normal, dir1), dim=-1)
        self.t = torch.stack([dir1, dir2, normal], dim=1)
        self.T = torch.func.vmap(torch.block_diag)(*(2 * self.etype.nodes * [self.t]))

        # Compute Jacobian and its determinant
        b = self.etype.B(xi)
        dx = (nodes - nodes[:, 0, None]).transpose(2, 1)
        self.loc_nodes = (self.t[:, 0:2, :] @ dx).transpose(2, 1)
        J = torch.einsum("...iN, ANj -> ...Aij", b, self.loc_nodes)
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise ValueError("Negative Jacobian. Check element numbering.")

        # Compute B
        B = torch.einsum("...Eij,...jN->...EiN", torch.linalg.inv(J), b)

        return self.etype.N(xi), B, detJ

    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        return torch.einsum("i,ijk->ijk", detJ, BCB)

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        raise NotImplementedError

    def integrate_mass(self) -> Tensor:
        """Mass matrix (translational: ∫ρ dz, rotational: ∫ρz² dz)."""
        n = self.n_dof_per_node * self.etype.nodes
        m = torch.zeros((self.n_elem, n, n))
        if self.section is not None:
            rho_trans = self.section.rho_h
            rho_rot = self.section.rho_zz
        else:
            assert self.material is not None
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
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
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
        n_dof = self.n_dof_per_node * N_nod
        f = torch.zeros(self.n_elem, n_dof)
        need_k = compute_stiffness and (
            self.K.numel() == 0 or self.n_state != 0 or nlgeom
        )
        k = torch.zeros(self.n_elem, n_dof, n_dof) if need_k else None

        # Compute gradient operators
        _, B, detJ = self.eval_shape_functions(self.etype.ipoints)

        # Through-thickness integration stations (layer-aware for laminates)
        materials, z_stations, w_stations = self._thickness_stations()

        # Transverse shear operators and stiffnesses at the integration points
        Ds = self._Ds(detJ)
        int_Cs = self._shear_correction(detJ)[..., None, None] * self.As

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

            # Element membrane-bending coupling stiffness.
            DmCDb = torch.einsum("...ji,...jk,...kl->...il", Dm, B_matrix, Db)
            DbCDm = torch.einsum("...ji,...jk,...kl->...il", Db, B_matrix, Dm)
            kc = wi * self.compute_k(detJ[i], DmCDb + DbCDm)

            # Element transverse stiffness
            DsCsDs = torch.einsum("...ji,...jk,...kl->...il", Ds[i], int_Cs[i], Ds[i])
            ks = wi * self.compute_k(detJ[i], DsCsDs)

            # Element drilling stiffness, a fraction of the section shear stiffness
            kd = torch.zeros_like(km)
            drill = torch.arange(N_nod) * self.n_dof_per_node + 5
            shear = self.As.diagonal(dim1=-2, dim2=-1).mean(-1)
            kd[:, drill, drill] = (self.drill_penalty * wi * detJ[i] * shear)[:, None]

            if k is not None:
                # Total element stiffness in local coordinates
                kt = km + kb + kc + ks + kd

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
        orientations: Tensor | None = None,
        thickness: bool = False,
        mirror: tuple[bool, bool, bool] = (False, False, False),
        show_undeformed: bool = False,
        axes: bool = False,
        bcs: bool = False,
        plotter: pyvista.Plotter | None = None,
        camera: CameraPositionOptions | None = None,
        **kwargs,
    ):
        """Plot the shell mesh with PyVista, optionally with results.

        Args:
            u: Nodal displacements added to the positions, e.g. to plot the
                deformed configuration. Defaults to 0.0 (undeformed).
            node_property: Named nodal fields, e.g. `{"u": u[:, :3]}`.
            element_property: Named element fields.
            orientations: Per-element direction vectors with shape
                [n_elem, k, 3] with k <= 3, e.g. the local frames `self.t`,
                drawn on the unmirrored mesh as red, green, and blue arrows of
                the mean element size.
            thickness: If True, extrudes elements by their thickness.
            mirror: Mirrors the mesh about the (x, y, z) planes, e.g. to
                visualize symmetric halves. Warns if the nodes on a mirrored
                plane are not constrained to enforce that symmetry.
            show_undeformed: If True, draws the undeformed mesh as a grey
                wireframe.
            axes: If True, shows labeled coordinate axes around the mesh.
            bcs: If True, renders boundary conditions on the unmirrored mesh:
                arrows for forces and prescribed displacements, spheres at
                displacement tips, and a cone per constrained DOF. Rotational
                DOFs use doubled heads, the usual convention for moments.
                Constraints enforcing the symmetry of a mirrored plane are
                skipped, since the mirrored copy shows that symmetry already.
            plotter: PyVista plotter. Defaults to None.
            camera: Camera position, either a plane ("xy", "xz", "yz"), "iso",
                or an explicit position, focal point and view up. Defaults to
                None.
            **kwargs: Forwarded to `pyvista.Plotter.add_mesh`.
        """
        pyvista.set_plot_theme("document")
        pl = pyvista.Plotter() if plotter is None else plotter
        pl.enable_anti_aliasing("ssaa")
        pl.renderer.add_axes()

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
        kwargs.setdefault("show_edges", True)
        base_meshes = []
        if thickness:
            nodal_thickness = np.zeros(len(self.nodes))
            count = np.zeros(len(self.nodes))
            for i, face in enumerate(mesh.faces.reshape(-1, self.etype.nodes + 1)):
                idx = face[1::]
                nodal_thickness[idx] += self.thickness[i].cpu().item()
                count[idx] += 1
            nodal_thickness /= count

            normals = np.asarray(mesh.point_normals)
            top = mesh.copy()
            top.points += 0.5 * nodal_thickness[:, None] * normals
            bottom = mesh.copy()
            bottom.points -= 0.5 * nodal_thickness[:, None] * normals

            pl.add_mesh(top, **kwargs)
            pl.add_mesh(bottom, **kwargs)
            base_meshes.extend([top, bottom])
        else:
            pl.add_mesh(mesh, **kwargs)
            base_meshes.append(mesh)

        # Plot orientations, lifted onto the top surface of a thick shell so
        # that they do not disappear inside it
        if orientations is not None:
            centers = (self.nodes + u)[self.elements].mean(dim=1).cpu().numpy()
            if thickness:
                offset = 0.5 * self.thickness[:, None].cpu().numpy()
                centers = centers + offset * np.asarray(mesh.cell_normals)
            mag = float(self.char_lengths.mean())
            for j, color in zip(range(orientations.shape[1]), ["red", "green", "blue"]):
                directions = torch.nn.functional.normalize(orientations[:, j], dim=-1)
                pl.add_arrows(
                    centers,
                    directions.cpu().numpy(),
                    mag=mag,
                    color=color,
                    show_scalar_bar=False,
                )

        # Symmetry constraints expected on each mirrored plane: the normal
        # translation and the two rotations about the in-plane axes
        symmetry = torch.zeros_like(self.constraints)
        tol = 1e-6 * float(self.char_lengths.mean())
        for axis, mirrored_axis in enumerate(mirror):
            if not mirrored_axis:
                continue
            on_plane = self.nodes[:, axis].abs() < tol
            dofs = sorted([axis, 3 + (axis + 1) % 3, 3 + (axis + 2) % 3])
            for dof in dofs:
                symmetry[on_plane, dof] = True
            if not (on_plane.any() and self.constraints[on_plane][:, dofs].all()):
                print(
                    f"Mirroring about {'xyz'[axis]} = 0, but its nodes are not "
                    f"constrained in DOFs {dofs} to enforce that symmetry."
                )

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
                        pl.add_mesh(mirrored, **{"opacity": 0.5, **kwargs})

        if show_undeformed:
            undefo = pyvista.PolyData(self.nodes.cpu().numpy(), elements)
            edges = cast(pyvista.DataSet, undefo.extract_all_edges())
            pl.add_mesh(edges, style="wireframe", color="grey")

        if bcs:
            points = self.nodes + u
            deformed = isinstance(u, Tensor)
            prescribed = torch.where(self.constraints, self._dirichlet, 0.0)
            size = torch.linalg.norm(
                points.max(dim=0).values - points.min(dim=0).values
            )
            height = 0.5 * float(self.char_lengths.mean())

            # Forces and moments scaled linearly, each normalized on its own
            # because they carry different units
            span = 0.1 * float(size)
            arrows(pl, points, self._neumann[:, :3], span=span)
            arrows(pl, points, self._neumann[:, 3:], span=span, doubled=True)

            # Prescribed translations to scale, with a sphere marking the tip. A
            # prescribed rotation cannot be drawn to scale, so it keeps its cone.
            fixed = self.constraints & ~symmetry
            if not deformed:
                fixed[:, :3] = fixed[:, :3] & (prescribed[:, :3] == 0.0)
                arrows(pl, points, prescribed[:, :3])
            pulled = torch.linalg.norm(prescribed[:, :3], dim=1) > 0.0
            ends = points if deformed else points + prescribed[:, :3]
            dots(pl, ends[pulled], 0.3 * height)
            cones(pl, points, fixed[:, :3], height)
            cones(pl, points, fixed[:, 3:], height, doubled=True)

        if axes:
            pl.renderer.show_grid()

        if camera is not None:
            pl.camera_position = camera

        if plotter is None:
            from .plot_utils import show_html

            show_html(pl)
