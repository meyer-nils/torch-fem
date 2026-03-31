import torch
from torch import Tensor


def planar_rotation(phi: float | Tensor) -> Tensor:
    """Create a planar rotation matrix with an angle phi."""
    phi = torch.as_tensor(phi)
    c = torch.cos(phi)
    s = torch.sin(phi)
    return torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-1,
    )


def axis_rotation(axis: Tensor, phi: float | Tensor) -> Tensor:
    """Create a rotation matrix with an angle phi around an axis."""
    phi = torch.as_tensor(phi)
    axis = axis / torch.norm(axis)
    x, y, z = axis
    c = torch.cos(phi)
    s = torch.sin(phi)
    t = 1 - c
    return torch.stack(
        [
            torch.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], dim=-1),
            torch.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], dim=-1),
            torch.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], dim=-1),
        ],
        dim=-1,
    )


def euler_rotation(rotation_angles: Tensor) -> Tensor:
    """Create a rotation matrix with Euler angles."""
    alpha = rotation_angles[..., 0]
    beta = rotation_angles[..., 1]
    gamma = rotation_angles[..., 2]
    return torch.stack(
        [
            torch.stack(
                [
                    torch.cos(alpha) * torch.cos(beta),
                    torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma)
                    - torch.sin(alpha) * torch.cos(gamma),
                    torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma)
                    + torch.sin(alpha) * torch.sin(gamma),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.sin(alpha) * torch.cos(beta),
                    torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma)
                    + torch.cos(alpha) * torch.cos(gamma),
                    torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma)
                    - torch.cos(alpha) * torch.sin(gamma),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    -torch.sin(beta),
                    torch.cos(beta) * torch.sin(gamma),
                    torch.cos(beta) * torch.cos(gamma),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
