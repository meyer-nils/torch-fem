"""Ply failure criteria for composite laminates.

These operate on plane-stress ply stresses expressed in the *material* (fiber)
axes: ``s11`` (fiber direction), ``s22`` (transverse direction), and ``t12``
(in-plane shear). All inputs may be batched tensors of identical shape.

A failure index ``FI`` is returned for each criterion: ``FI < 1`` is safe and
``FI >= 1`` indicates failure. For the quadratic criteria, the strength ratio
(load factor to first-ply failure under proportional loading) is
``R = 1 / sqrt(FI)``.

Strengths are positive magnitudes:
    - ``Xt`` / ``Xc``: longitudinal (fiber) tensile / compressive strength.
    - ``Yt`` / ``Yc``: transverse (matrix) tensile / compressive strength.
    - ``S12``: in-plane (longitudinal) shear strength.
    - ``S23``: transverse shear strength (Hashin matrix compression only;
      defaults to ``S12`` if not provided).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .rotations import planar_rotation


@dataclass(frozen=True)
class PlyStrength:
    """Plane-stress strength allowables of a single ply (positive magnitudes)."""

    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S12: float
    S23: float | None = None

    @property
    def St(self) -> float:
        """Transverse shear strength (falls back to in-plane shear strength)."""
        return self.S23 if self.S23 is not None else self.S12


def rotate_stress_to_ply(sigma_elem: Tensor, angle: float | Tensor) -> Tensor:
    """Rotate an element-axes plane-stress tensor into ply material axes.

    Args:
        sigma_elem: Stress tensor in element axes, shape `(..., 2, 2)`.
        angle: Ply orientation (radians), measured as for `planar_rotation`.

    Returns:
        Stress tensor in material axes, shape `(..., 2, 2)`
        (``sigma_mat = R^T sigma R``).
    """
    R = planar_rotation(torch.as_tensor(angle))
    return torch.einsum("ji,...jk,kl->...il", R, sigma_elem, R)


def tsai_hill(s11: Tensor, s22: Tensor, t12: Tensor, strength: PlyStrength) -> Tensor:
    """Tsai-Hill failure index.

    $$
        FI = \\frac{\\sigma_{11}^2}{X^2} - \\frac{\\sigma_{11}\\sigma_{22}}{X^2}
             + \\frac{\\sigma_{22}^2}{Y^2} + \\frac{\\tau_{12}^2}{S^2}
    $$

    where ``X`` and ``Y`` are the tensile or compressive strengths selected by
    the sign of ``s11`` and ``s22`` respectively, and ``S = S12``.
    """
    X = torch.where(s11 >= 0.0, torch.as_tensor(strength.Xt), torch.as_tensor(strength.Xc))
    Y = torch.where(s22 >= 0.0, torch.as_tensor(strength.Yt), torch.as_tensor(strength.Yc))
    return s11**2 / X**2 - s11 * s22 / X**2 + s22**2 / Y**2 + t12**2 / strength.S12**2


def hashin(
    s11: Tensor, s22: Tensor, t12: Tensor, strength: PlyStrength
) -> dict[str, Tensor]:
    """Hashin (2D, plane-stress) failure indices.

    Returns a dict with four mode indices plus the per-point envelope:
        - ``fiber_tension``      (s11 >= 0)
        - ``fiber_compression``  (s11 < 0)
        - ``matrix_tension``     (s22 >= 0)
        - ``matrix_compression`` (s22 < 0)
        - ``fiber``  = active fiber mode (selected by sign of s11)
        - ``matrix`` = active matrix mode (selected by sign of s22)
        - ``max``    = max(fiber, matrix)
    """
    Xt, Xc, Yt, Yc = strength.Xt, strength.Xc, strength.Yt, strength.Yc
    Sl, St = strength.S12, strength.St

    fiber_tension = (s11 / Xt) ** 2 + (t12 / Sl) ** 2
    fiber_compression = (s11 / Xc) ** 2
    matrix_tension = (s22 / Yt) ** 2 + (t12 / Sl) ** 2
    matrix_compression = (
        (s22 / (2.0 * St)) ** 2
        + ((Yc / (2.0 * St)) ** 2 - 1.0) * (s22 / Yc)
        + (t12 / Sl) ** 2
    )

    fiber = torch.where(s11 >= 0.0, fiber_tension, fiber_compression)
    matrix = torch.where(s22 >= 0.0, matrix_tension, matrix_compression)
    return {
        "fiber_tension": fiber_tension,
        "fiber_compression": fiber_compression,
        "matrix_tension": matrix_tension,
        "matrix_compression": matrix_compression,
        "fiber": fiber,
        "matrix": matrix,
        "max": torch.maximum(fiber, matrix),
    }
