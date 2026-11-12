import torch
from torch import Tensor


def planar_rotation(phi: float | Tensor) -> Tensor:
    """Create a planar rotation matrix rotating by phi counter-clockwise."""
    phi = torch.as_tensor(phi)
    c = torch.cos(phi)
    s = torch.sin(phi)
    return torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-2,
    )


def axis_rotation(axis: Tensor, phi: float | Tensor) -> Tensor:
    """Create a rotation matrix rotating by phi around an axis, right-handed."""
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
        dim=-2,
    )


def euler_rotation(rotation_angles: Tensor) -> Tensor:
    """Create a rotation matrix from intrinsic z-y-x Euler angles."""
    ca, cb, cg = torch.cos(rotation_angles).unbind(-1)
    sa, sb, sg = torch.sin(rotation_angles).unbind(-1)
    row_1 = [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg]
    row_2 = [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg]
    row_3 = [-sb, cb * sg, cb * cg]
    return torch.stack([torch.stack(r, dim=-1) for r in (row_1, row_2, row_3)], dim=-2)
