from abc import abstractmethod

import torch
from torch import Tensor

EPS = 1e-10


class SDF:
    def __init__(self, center: Tensor = torch.zeros(3)):
        self.center = center

    @abstractmethod
    def f(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def grad(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    def sdf(self, points: Tensor) -> Tensor:
        return self.f(points) / (torch.norm(self.grad(points), dim=1) + EPS)

    def _to_xyz(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (
            points[:, 0] - self.center[0],
            points[:, 1] - self.center[1],
            points[:, 2] - self.center[2],
        )


class Gyroid(SDF):
    def __init__(self, center: Tensor = torch.zeros(3), scale: float = 2 * torch.pi):
        super().__init__(center)
        self.scale = scale

    def f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
        return (
            torch.sin(x) * torch.cos(y)
            + torch.sin(y) * torch.cos(z)
            + torch.sin(z) * torch.cos(x)
        )

    def grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
        grad_x = torch.cos(x) * torch.cos(y) - torch.sin(z) * torch.cos(x)
        grad_y = torch.cos(y) * torch.cos(z) - torch.sin(x) * torch.cos(y)
        grad_z = torch.cos(z) * torch.cos(x) - torch.sin(y) * torch.cos(z)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class SchwarzP(SDF):
    def __init__(
        self,
        center: Tensor = torch.zeros(3),
        scale: float = 2 * torch.pi,
        c: float = 0.0,
    ):
        super().__init__(center)
        self.scale = scale
        self.c = c

    def f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
        return torch.cos(x) + torch.cos(y) + torch.cos(z) - self.c

    def grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
        grad_x = -torch.sin(x)
        grad_y = -torch.sin(y)
        grad_z = -torch.sin(z)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Diamond(SDF):
    def __init__(self, center: Tensor = torch.zeros(3), scale: float = 2 * torch.pi):
        super().__init__(center)
        self.scale = scale

    def f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
        return (
            torch.sin(x) * torch.sin(y) * torch.sin(z)
            + torch.sin(x) * torch.cos(y) * torch.cos(z)
            + torch.cos(x) * torch.sin(y) * torch.cos(z)
            + torch.cos(x) * torch.cos(y) * torch.sin(z)
        )

    def grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
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
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Lidinoid(SDF):
    def __init__(self, center: Tensor = torch.zeros(3), scale: float = 2 * torch.pi):
        super().__init__(center)
        self.scale = scale

    def f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
        return (
            torch.sin(2 * x) * torch.cos(y) * torch.sin(z)
            + torch.sin(2 * y) * torch.cos(z) * torch.sin(x)
            + torch.sin(2 * z) * torch.cos(x) * torch.sin(y)
            - torch.cos(2 * x) * torch.cos(2 * y)
            - torch.cos(2 * y) * torch.cos(2 * z)
            - torch.cos(2 * z) * torch.cos(2 * x)
            + 0.3
        )

    def grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
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
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class SplitP(SDF):
    def __init__(self, center: Tensor = torch.zeros(3), scale: float = 2 * torch.pi):
        super().__init__(center)
        self.scale = scale

    def f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
        return (
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

    def grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
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
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Neovius(SDF):
    def __init__(self, center: Tensor = torch.zeros(3), scale: float = 2 * torch.pi):
        super().__init__(center)
        self.scale = scale

    def f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
        return 3 * (torch.cos(x) + torch.cos(y) + torch.cos(z)) + 4 * torch.cos(
            x
        ) * torch.cos(y) * torch.cos(z)

    def grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(self.scale * points)
        grad_x = -3 * torch.sin(x) - 4 * torch.sin(x) * torch.cos(y) * torch.cos(z)
        grad_y = -3 * torch.sin(y) - 4 * torch.cos(x) * torch.sin(y) * torch.cos(z)
        grad_z = -3 * torch.sin(z) - 4 * torch.cos(x) * torch.cos(y) * torch.sin(z)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Sphere(SDF):
    def __init__(self, center: Tensor = torch.zeros(3), radius: float = 1.0):
        super().__init__(center)
        self.radius = radius

    def f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return torch.sqrt(x**2 + y**2 + z**2) - self.radius

    def grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        grad_x = x / (torch.sqrt(x**2 + y**2 + z**2) + EPS)
        grad_y = y / (torch.sqrt(x**2 + y**2 + z**2) + EPS)
        grad_z = z / (torch.sqrt(x**2 + y**2 + z**2) + EPS)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Cube(SDF):
    def __init__(self, center: Tensor = torch.zeros(3), size: Tensor = torch.ones(3)):
        super().__init__(center)
        self.size = size

    def f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return torch.max(
            torch.stack(
                [
                    torch.abs(x) - self.size[0],
                    torch.abs(y) - self.size[1],
                    torch.abs(z) - self.size[2],
                ],
            ),
            dim=0,
        ).values

    def grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        grad_x = torch.sign(x)
        grad_y = torch.sign(y)
        grad_z = torch.sign(z)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)
