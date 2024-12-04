import torch
from torch import Tensor

EPS = 1e-10


def gyroid(points: Tensor) -> Tensor:
    # Extract x, y, z coordinates
    x = 2 * torch.pi * points[:, 0]
    y = 2 * torch.pi * points[:, 1]
    z = 2 * torch.pi * points[:, 2]

    # Implicit function
    f = (
        torch.sin(x) * torch.cos(y)
        + torch.sin(y) * torch.cos(z)
        + torch.sin(z) * torch.cos(x)
    )

    # Gradient
    grad_x = torch.cos(x) * torch.cos(y) - torch.sin(z) * torch.cos(x)
    grad_y = torch.cos(y) * torch.cos(z) - torch.sin(x) * torch.cos(y)
    grad_z = torch.cos(z) * torch.cos(x) - torch.sin(y) * torch.cos(z)

    # Gradient magnitude
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return f / (grad_norm + EPS)


def schwarz_primitive(points: Tensor, c: float = 0.0) -> Tensor:
    # Extract x, y, z coordinates
    x = 2 * torch.pi * points[:, 0]
    y = 2 * torch.pi * points[:, 1]
    z = 2 * torch.pi * points[:, 2]

    # Implicit function
    f = torch.cos(x) + torch.cos(y) + torch.cos(z)

    # Gradient
    grad_x = -torch.sin(x)
    grad_y = -torch.sin(y)
    grad_z = -torch.sin(z)

    # Gradient magnitude
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return (f - c) / (grad_norm + EPS)


def diamond(points: Tensor) -> Tensor:
    # Extract x, y, z coordinates
    x = 2 * torch.pi * points[:, 0]
    y = 2 * torch.pi * points[:, 1]
    z = 2 * torch.pi * points[:, 2]

    # Implicit function
    f = (
        torch.sin(x) * torch.sin(y) * torch.sin(z)
        + torch.sin(x) * torch.cos(y) * torch.cos(z)
        + torch.cos(x) * torch.sin(y) * torch.cos(z)
        + torch.cos(x) * torch.cos(y) * torch.sin(z)
    )

    # Gradient
    grad_x = (
        torch.cos(x) * torch.sin(y) * torch.sin(z)
        + torch.cos(x) * torch.cos(y) * torch.cos(z)
        - torch.sin(x) * torch.cos(y) * torch.cos(z)
        - torch.sin(x) * torch.sin(y) * torch.cos(z)
    )
    grad_y = (
        torch.sin(x) * torch.cos(y) * torch.sin(z)
        - torch.sin(x) * torch.sin(y) * torch.cos(z)
        + torch.cos(x) * torch.cos(y) * torch.cos(z)
        - torch.cos(x) * torch.sin(y) * torch.sin(z)
    )
    grad_z = (
        torch.sin(x) * torch.sin(y) * torch.cos(z)
        - torch.sin(x) * torch.cos(y) * torch.sin(z)
        + torch.cos(x) * torch.sin(y) * torch.cos(z)
        - torch.cos(x) * torch.cos(y) * torch.sin(z)
    )

    # Gradient magnitude
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return f / (grad_norm + EPS)


def lidinoid(points: Tensor) -> Tensor:
    # Extract x, y, z coordinates
    x = 2 * torch.pi * points[:, 0]
    y = 2 * torch.pi * points[:, 1]
    z = 2 * torch.pi * points[:, 2]

    # Implicit function
    f = (
        torch.sin(2 * x) * torch.cos(y) * torch.sin(z)
        + torch.sin(2 * y) * torch.cos(z) * torch.sin(x)
        + torch.sin(2 * z) * torch.cos(x) * torch.sin(y)
        - torch.cos(2 * x) * torch.cos(2 * y)
        - torch.cos(2 * y) * torch.cos(2 * z)
        - torch.cos(2 * z) * torch.cos(2 * x)
        + 0.3
    )

    # Gradient
    grad_x = (
        2 * torch.cos(2 * x) * torch.cos(y) * torch.sin(z)
        + torch.cos(x) * torch.sin(2 * z) * torch.sin(y)
        - 2 * torch.sin(2 * x) * torch.cos(2 * y)
        - 2 * torch.sin(2 * x) * torch.cos(2 * z)
    )
    grad_y = (
        2 * torch.cos(2 * y) * torch.cos(z) * torch.sin(x)
        + torch.cos(y) * torch.sin(2 * x) * torch.sin(z)
        - 2 * torch.sin(2 * y) * torch.cos(2 * x)
        - 2 * torch.sin(2 * y) * torch.cos(2 * z)
    )
    grad_z = (
        2 * torch.cos(2 * z) * torch.cos(x) * torch.sin(y)
        + torch.cos(z) * torch.sin(2 * y) * torch.sin(x)
        - 2 * torch.sin(2 * z) * torch.cos(2 * y)
        - 2 * torch.sin(2 * z) * torch.cos(2 * x)
    )

    # Gradient magnitude
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return f / (grad_norm + EPS)


def split_p(points: Tensor) -> Tensor:
    # Extract x, y, z coordinates
    x = 2 * torch.pi * points[:, 0]
    y = 2 * torch.pi * points[:, 1]
    z = 2 * torch.pi * points[:, 2]

    # Implicit function
    f = (
        1.1
        * (
            torch.sin(2 * x) * torch.sin(z) * torch.cos(y)
            + torch.sin(2 * y) * torch.sin(x) * torch.cos(z)
            + torch.sin(2 * z) * torch.sin(y) * torch.cos(x)
        )
        - 0.2
        * (
            torch.cos(2 * x) * torch.cos(2 * y)
            + torch.cos(2 * y) * torch.cos(2 * z)
            + torch.cos(2 * z) * torch.cos(2 * x)
        )
        - 0.4 * (torch.cos(2 * x) + torch.cos(2 * y) + torch.cos(2 * z))
    )

    # Gradient
    grad_x = (
        2.2 * torch.cos(2 * x) * torch.sin(z) * torch.cos(y)
        + torch.cos(x) * torch.sin(2 * y) * torch.cos(z)
        - 0.4 * torch.sin(2 * x)
        - 0.2
        * (
            2 * torch.sin(2 * x) * torch.cos(2 * y)
            + 2 * torch.sin(2 * x) * torch.cos(2 * z)
        )
    )
    grad_y = (
        2.2 * torch.cos(2 * y) * torch.sin(x) * torch.cos(z)
        + torch.cos(y) * torch.sin(2 * z) * torch.cos(x)
        - 0.4 * torch.sin(2 * y)
        - 0.2
        * (
            2 * torch.sin(2 * y) * torch.cos(2 * z)
            + 2 * torch.sin(2 * y) * torch.cos(2 * x)
        )
    )
    grad_z = (
        2.2 * torch.cos(2 * z) * torch.sin(y) * torch.cos(x)
        + torch.cos(z) * torch.sin(2 * x) * torch.cos(y)
        - 0.4 * torch.sin(2 * z)
        - 0.2
        * (
            2 * torch.sin(2 * z) * torch.cos(2 * x)
            + 2 * torch.sin(2 * z) * torch.cos(2 * y)
        )
    )

    # Gradient magnitude
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return f / (grad_norm + EPS)


def neovius(points: Tensor) -> Tensor:
    # Extract x, y, z coordinates
    x = 2 * torch.pi * points[:, 0]
    y = 2 * torch.pi * points[:, 1]
    z = 2 * torch.pi * points[:, 2]

    # Implicit function
    f = 3 * (torch.cos(x) + torch.cos(y) + torch.cos(z)) + 4 * torch.cos(x) * torch.cos(
        y
    ) * torch.cos(z)

    # Gradient
    grad_x = -3 * torch.sin(x) - 4 * torch.sin(x) * torch.cos(y) * torch.cos(z)
    grad_y = -3 * torch.sin(y) - 4 * torch.cos(x) * torch.sin(y) * torch.cos(z)
    grad_z = -3 * torch.sin(z) - 4 * torch.cos(x) * torch.cos(y) * torch.sin(z)

    # Gradient magnitude
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return f / (grad_norm + EPS)


def sphere(points: Tensor, center: Tensor, radius: float) -> Tensor:
    # Extract x, y, z coordinates
    x = points[:, 0] - center[0]
    y = points[:, 1] - center[1]
    z = points[:, 2] - center[2]

    # Implicit function
    f = torch.sqrt(x**2 + y**2 + z**2) - radius

    # Gradient
    grad_x = x / (torch.sqrt(x**2 + y**2 + z**2) + EPS)
    grad_y = y / (torch.sqrt(x**2 + y**2 + z**2) + EPS)
    grad_z = z / (torch.sqrt(x**2 + y**2 + z**2) + EPS)

    # Gradient magnitude
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    return f / (grad_norm + EPS)
