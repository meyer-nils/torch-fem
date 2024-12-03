import torch
from torch import Tensor


def gyroid(points: Tensor) -> Tensor:
    # Extract x, y, z coordinates
    x = 2 * torch.pi * points[:, 0]
    y = 2 * torch.pi * points[:, 1]
    z = 2 * torch.pi * points[:, 2]

    # Gyroid implicit function
    gyroid_value = (
        torch.sin(x) * torch.cos(y)
        + torch.sin(y) * torch.cos(z)
        + torch.sin(z) * torch.cos(x)
    )

    # Approximation of the distance to the surface
    grad_x = torch.cos(x) * torch.cos(y) - torch.sin(z) * torch.cos(x)
    grad_y = torch.cos(y) * torch.cos(z) - torch.sin(x) * torch.cos(y)
    grad_z = torch.cos(z) * torch.cos(x) - torch.sin(y) * torch.cos(z)
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    return gyroid_value / grad_norm
