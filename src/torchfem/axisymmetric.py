from functools import cached_property

import torch
from torch import Tensor

from .base import Heat, Mechanics
from .materials import Material
from .planar import PlanarGeometry


class AxisymmetricGeometry(PlanarGeometry):
    """The elements, integration and plotting shared by the axisymmetric models.

    It carries a cross section in the half plane r >= 0 with nodes (r, z). Every
    measure carries the revolution 2 pi r, so `integrate_field(...)` returns volumes
    where the planar model returns areas.

    Attributes:
        nodes: Nodal coordinates (r, z) with shape [n_nod, 2].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized material model.
        constraints: Boolean mask of constrained DOFs with shape [n_nod, n_dof].
    """

    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize the axisymmetric FEM problem, which carries no thickness."""
        super().__init__(nodes, elements, material)

    def __repr__(self) -> str:
        etype, n = self.etype.__name__, self.n_nod
        return f"<torch-fem axisymmetric ({n} nodes, {self.n_elem} {etype} elements)>"

    def radii(self, N: Tensor, conn: Tensor) -> Tensor:
        """Radius at the quadrature points `N` is evaluated at."""
        return torch.einsum("in,en->ie", N, self.nodes[conn, 0])

    def eval_shape_functions(self, xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """As the planar model, with the revolved measure folded into detJ."""
        N, B, detJ = super().eval_shape_functions(xi)
        return N, B, 2 * torch.pi * self.radii(N, self.elements) * detJ

    def facet_measure(self, conn: Tensor, N: Tensor, detJ: Tensor) -> Tensor:
        """A revolved edge is a surface of revolution."""
        return 2 * torch.pi * self.radii(N, conn) * detJ

    @cached_property
    def char_lengths(self) -> Tensor:
        """Characteristic lengths of the cross sections, not of the revolved solid."""
        _, _, detJ = super().eval_shape_functions(self.etype.ipoints)
        return (self.etype.iweights @ detJ) ** (1 / 2)


class Axisymmetric(AxisymmetricGeometry, Mechanics):
    """Axisymmetric mechanics model for a torsionless solid of revolution.

    It takes a three-dimensional material and orders its tensors (r, z, hoop), the
    order an anisotropic material reads its axes in. The hoop strain u_r / r is
    carried by the radial degree of freedom, so a node holds two of them.

    Attributes:
        nodes: Nodal coordinates (r, z) with shape [n_nod, 2].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized material model.
        forces: Applied nodal forces with shape [n_nod, 2], acting on the full
            circumference rather than per radian.
        displacements: Prescribed nodal displacements with shape [n_nod, 2].
        constraints: Boolean mask of constrained DOFs with shape [n_nod, 2].
    """

    @property
    def n_flux(self) -> list[int]:
        """Shape of the stress tensor."""
        return [3, 3]

    def eval_shape_functions(self, xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """As the geometry, with the hoop row N / r appended to B."""
        N, B, detJ = super().eval_shape_functions(xi)
        hoop = N[:, None, :] / self.radii(N, self.elements)[..., None]
        return N, torch.cat([B, hoop.unsqueeze(-2)], dim=-2), detJ

    def compute_h(self, du: Tensor, B: Tensor) -> Tensor:
        """Gradient increment, whose hoop component is the radial DOF over r."""
        H = torch.zeros(self.n_elem, 3, 3, device=du.device)
        H[..., :2, :2] = du.transpose(-1, -2) @ B[..., :2, :].transpose(-1, -2)
        H[..., 2, 2] = (du[..., 0] * B[..., 2, :]).sum(dim=-1)
        return H

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor) -> Tensor:
        """Element internal force, with the hoop stress working on the radial DOF."""
        w = self.thickness * detJ
        f = torch.einsum("...,...iI,...Ai->...IA", w, B[..., :2, :], S[..., :2, :2])
        f[..., 0] += w[:, None] * B[..., 2, :] * S[..., 2, 2][:, None]
        return f

    def compute_bcb(self, B: Tensor, ddsdde: Tensor) -> Tensor:
        """Tangent between the gradient operators, the hoop row acting on u_r."""
        Brz, h = B[..., :2, :], B[..., 2, :]
        Cpp, Chp = ddsdde[..., :2, :2, :2, :2], ddsdde[..., 2, 2, :2, :2]
        Cph, Chh = ddsdde[..., :2, :2, 2, 2], ddsdde[..., 2, 2, 2, 2]
        k = torch.einsum("...Jp,...iJkL,...Lq->...piqk", Brz, Cpp, Brz)
        k[..., 0, :, :] += torch.einsum("...p,...kL,...Lq->...pqk", h, Chp, Brz)
        k[..., 0] += torch.einsum("...Jp,...iJ,...q->...piq", Brz, Cph, h)
        k[..., 0, :, 0] += torch.einsum("...p,...,...q->...pq", h, Chh, h)
        return k

    def near_null_space(self) -> Tensor:
        """Axial translation alone, since a radial one strains the hoop direction."""
        return torch.tensor([0.0, 1.0]).repeat(self.n_nod)[:, None]


class AxisymmetricHeat(AxisymmetricGeometry, Heat):
    """Axisymmetric heat conduction model.

    The temperature gradient has no hoop component, so this is the planar model on
    the revolved measure, with the same two-dimensional material.

    Attributes:
        nodes: Nodal coordinates (r, z) with shape [n_nod, 2].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized thermal material model.
        heat_flux: Applied nodal heat sources with shape [n_nod, 1].
        temperatures: Prescribed nodal temperatures with shape [n_nod, 1].
        constraints: Boolean mask of constrained DOFs with shape [n_nod, 1].
    """
