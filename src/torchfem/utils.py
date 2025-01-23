import torch
from torch import Tensor


def planar_stress_rotation(phi: Tensor) -> Tensor:
    """Rotate a planar stress with an angle phi assuming Voigt notation."""
    cos = torch.cos(phi)
    cos2 = cos**2
    sin = torch.sin(phi)
    sin2 = sin**2
    sincos = sin * cos
    return torch.stack(
        [
            torch.stack([cos2, sin2, 2.0 * sincos], dim=-1),
            torch.stack([sin2, cos2, -2.0 * sincos], dim=-1),
            torch.stack([-sincos, sincos, cos2 - sin2], dim=-1),
        ],
        dim=-1,
    )


def planar_strain_rotation(phi: Tensor) -> Tensor:
    """Rotate a planar strain with an angle phi assuming Voigt notation."""
    cos = torch.cos(phi)
    cos2 = cos**2
    sin = torch.sin(phi)
    sin2 = sin**2
    sincos = sin * cos
    return torch.stack(
        [
            torch.stack([cos2, sin2, sincos], dim=-1),
            torch.stack([sin2, cos2, -sincos], dim=-1),
            torch.stack([-sincos, sincos, cos2 - sin2], dim=-1),
        ],
        dim=-1,
    )


def stress_rotation(R: Tensor) -> Tensor:
    return torch.stack(
        [
            torch.stack(
                [
                    R[..., 0, 0] ** 2,
                    R[..., 0, 1] ** 2,
                    R[..., 0, 2] ** 2,
                    2 * R[..., 0, 1] * R[..., 0, 2],
                    2 * R[..., 0, 2] * R[..., 0, 0],
                    2 * R[..., 0, 0] * R[..., 0, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    R[..., 1, 0] ** 2,
                    R[..., 1, 1] ** 2,
                    R[..., 1, 2] ** 2,
                    2 * R[..., 1, 1] * R[..., 1, 2],
                    2 * R[..., 1, 2] * R[..., 1, 0],
                    2 * R[..., 1, 0] * R[..., 1, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    R[..., 2, 0] ** 2,
                    R[..., 2, 1] ** 2,
                    R[..., 2, 2] ** 2,
                    2 * R[..., 2, 1] * R[..., 2, 2],
                    2 * R[..., 2, 2] * R[..., 2, 0],
                    2 * R[..., 2, 0] * R[..., 2, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    R[..., 1, 0] * R[..., 2, 0],
                    R[..., 1, 1] * R[..., 2, 1],
                    R[..., 1, 2] * R[..., 2, 2],
                    R[..., 1, 1] * R[..., 2, 2] + R[..., 1, 2] * R[..., 2, 1],
                    R[..., 1, 2] * R[..., 2, 0] + R[..., 1, 0] * R[..., 2, 2],
                    R[..., 1, 0] * R[..., 2, 1] + R[..., 1, 1] * R[..., 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    R[..., 2, 0] * R[..., 0, 0],
                    R[..., 2, 1] * R[..., 0, 1],
                    R[..., 2, 2] * R[..., 0, 2],
                    R[..., 2, 1] * R[..., 0, 2] + R[..., 2, 2] * R[..., 0, 1],
                    R[..., 2, 2] * R[..., 0, 0] + R[..., 2, 0] * R[..., 0, 2],
                    R[..., 2, 0] * R[..., 0, 1] + R[..., 2, 1] * R[..., 0, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    R[..., 0, 0] * R[..., 1, 0],
                    R[..., 0, 1] * R[..., 1, 1],
                    R[..., 0, 2] * R[..., 1, 2],
                    R[..., 0, 1] * R[..., 1, 2] + R[..., 0, 2] * R[..., 1, 1],
                    R[..., 0, 2] * R[..., 1, 0] + R[..., 0, 0] * R[..., 1, 2],
                    R[..., 0, 0] * R[..., 1, 1] + R[..., 0, 1] * R[..., 1, 0],
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )


def strain_rotation(R: Tensor) -> Tensor:
    return torch.stack(
        [
            torch.stack(
                [
                    R[..., 0, 0] ** 2,
                    R[..., 0, 1] ** 2,
                    R[..., 0, 2] ** 2,
                    R[..., 0, 1] * R[..., 0, 2],
                    R[..., 0, 2] * R[..., 0, 0],
                    R[..., 0, 0] * R[..., 0, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    R[..., 1, 0] ** 2,
                    R[..., 1, 1] ** 2,
                    R[..., 1, 2] ** 2,
                    R[..., 1, 1] * R[..., 1, 2],
                    R[..., 1, 2] * R[..., 1, 0],
                    R[..., 1, 0] * R[..., 1, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    R[..., 2, 0] ** 2,
                    R[..., 2, 1] ** 2,
                    R[..., 2, 2] ** 2,
                    R[..., 2, 1] * R[..., 2, 2],
                    R[..., 2, 2] * R[..., 2, 0],
                    R[..., 2, 0] * R[..., 2, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * R[..., 1, 0] * R[..., 2, 0],
                    2 * R[..., 1, 1] * R[..., 2, 1],
                    2 * R[..., 1, 2] * R[..., 2, 2],
                    R[..., 1, 1] * R[..., 2, 2] + R[..., 1, 2] * R[..., 2, 1],
                    R[..., 1, 2] * R[..., 2, 0] + R[..., 1, 0] * R[..., 2, 2],
                    R[..., 1, 0] * R[..., 2, 1] + R[..., 1, 1] * R[..., 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * R[..., 2, 0] * R[..., 0, 0],
                    2 * R[..., 2, 1] * R[..., 0, 1],
                    2 * R[..., 2, 2] * R[..., 0, 2],
                    R[..., 2, 1] * R[..., 0, 2] + R[..., 2, 2] * R[..., 0, 1],
                    R[..., 2, 2] * R[..., 0, 0] + R[..., 2, 0] * R[..., 0, 2],
                    R[..., 2, 0] * R[..., 0, 1] + R[..., 2, 1] * R[..., 0, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * R[..., 0, 0] * R[..., 1, 0],
                    2 * R[..., 0, 1] * R[..., 1, 1],
                    2 * R[..., 0, 2] * R[..., 1, 2],
                    R[..., 0, 1] * R[..., 1, 2] + R[..., 0, 2] * R[..., 1, 1],
                    R[..., 0, 2] * R[..., 1, 0] + R[..., 0, 0] * R[..., 1, 2],
                    R[..., 0, 0] * R[..., 1, 1] + R[..., 0, 1] * R[..., 1, 0],
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
