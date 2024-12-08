from abc import abstractmethod
from typing import Callable

import torch
from torch import Tensor

EPS = 1e-10
torch.set_default_dtype(torch.float64)


class SDF:
    def __init__(self, center: Tensor = torch.zeros(3), scale: Tensor = torch.ones(3)):
        self.center = center
        self.transform = torch.diag(1 / scale)

    @abstractmethod
    def _f(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _grad(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    def __and__(self, other):
        return Intersection(self, other)

    def __or__(self, other):
        return Union(self, other)

    def __sub__(self, other):
        return Difference(self, other)

    def sdf(self, points: Tensor) -> Tensor:
        return self._f(points) / (torch.norm(self._grad(points), dim=1) + EPS)

    def scale(self, scale: Tensor):
        self.transform = self.transform @ torch.diag(1 / scale)
        return self

    def translate(self, center: Tensor):
        self.center += center
        return self

    def rotate(self, axis: Tensor, angle: Tensor):
        axis = axis / torch.norm(axis)
        x, y, z = axis
        c = torch.cos(angle)
        s = torch.sin(angle)
        t = 1 - c
        rotation = torch.tensor(
            [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ]
        )
        self.transform = rotation @ self.transform
        return self

    def _to_xyz(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p = (points - self.center) @ self.transform
        return p[:, 0], p[:, 1], p[:, 2]


class Boolean(SDF):
    def __init__(self, sdf1: SDF, sdf2: SDF):
        super().__init__((sdf1.center + sdf2.center) / 2)
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    def translate(self, center):
        self.sdf1.translate(center)
        self.sdf2.translate(center)
        return self

    def scale(self, scale):
        self.sdf1.scale(scale)
        self.sdf2.scale(scale)
        return self

    def rotate(self, axis, angle):
        self.sdf1.rotate(axis, angle)
        self.sdf2.rotate(axis, angle)
        return self


class Union(Boolean):

    def sdf(self, points: Tensor) -> Tensor:
        return torch.min(self.sdf1.sdf(points), self.sdf2.sdf(points))


class Intersection(Boolean):

    def sdf(self, points: Tensor) -> Tensor:
        return torch.max(self.sdf1.sdf(points), self.sdf2.sdf(points))


class Difference(Boolean):

    def sdf(self, points: Tensor) -> Tensor:
        return torch.max(self.sdf1.sdf(points), -self.sdf2.sdf(points))


class Gyroid(SDF):
    def __init__(
        self,
        center: Tensor = torch.zeros(3),
        scale: Tensor = 1 / (2 * torch.pi) * torch.ones(3),
    ):
        super().__init__(center, scale)

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return (
            torch.sin(x) * torch.cos(y)
            + torch.sin(y) * torch.cos(z)
            + torch.sin(z) * torch.cos(x)
        )

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        grad_x = torch.cos(x) * torch.cos(y) - torch.sin(z) * torch.cos(x)
        grad_y = torch.cos(y) * torch.cos(z) - torch.sin(x) * torch.cos(y)
        grad_z = torch.cos(z) * torch.cos(x) - torch.sin(y) * torch.cos(z)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class SchwarzP(SDF):
    def __init__(
        self,
        center: Tensor = torch.zeros(3),
        scale: Tensor = 1 / (2 * torch.pi) * torch.ones(3),
        c: float = 0.0,
    ):
        super().__init__(center, scale)
        self.c = c

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return torch.cos(x) + torch.cos(y) + torch.cos(z) - self.c

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        grad_x = -torch.sin(x)
        grad_y = -torch.sin(y)
        grad_z = -torch.sin(z)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Diamond(SDF):
    def __init__(
        self,
        center: Tensor = torch.zeros(3),
        scale: Tensor = 1 / (2 * torch.pi) * torch.ones(3),
    ):
        super().__init__(center, scale)

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return (
            torch.sin(x) * torch.sin(y) * torch.sin(z)
            + torch.sin(x) * torch.cos(y) * torch.cos(z)
            + torch.cos(x) * torch.sin(y) * torch.cos(z)
            + torch.cos(x) * torch.cos(y) * torch.sin(z)
        )

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
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
    def __init__(
        self,
        center: Tensor = torch.zeros(3),
        scale: Tensor = 1 / (2 * torch.pi) * torch.ones(3),
    ):
        super().__init__(center, scale)

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return (
            torch.sin(2 * x) * torch.cos(y) * torch.sin(z)
            + torch.sin(2 * y) * torch.cos(z) * torch.sin(x)
            + torch.sin(2 * z) * torch.cos(x) * torch.sin(y)
            - torch.cos(2 * x) * torch.cos(2 * y)
            - torch.cos(2 * y) * torch.cos(2 * z)
            - torch.cos(2 * z) * torch.cos(2 * x)
            + 0.3
        )

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
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
    def __init__(
        self,
        center: Tensor = torch.zeros(3),
        scale: Tensor = 1 / (2 * torch.pi) * torch.ones(3),
    ):
        super().__init__(center, scale)

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
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

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
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
    def __init__(
        self,
        center: Tensor = torch.zeros(3),
        scale: Tensor = 1 / (2 * torch.pi) * torch.ones(3),
    ):
        super().__init__(center, scale)

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return 3 * (torch.cos(x) + torch.cos(y) + torch.cos(z)) + 4 * torch.cos(
            x
        ) * torch.cos(y) * torch.cos(z)

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        grad_x = -3 * torch.sin(x) - 4 * torch.sin(x) * torch.cos(y) * torch.cos(z)
        grad_y = -3 * torch.sin(y) - 4 * torch.cos(x) * torch.sin(y) * torch.cos(z)
        grad_z = -3 * torch.sin(z) - 4 * torch.cos(x) * torch.cos(y) * torch.sin(z)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Sphere(SDF):
    def __init__(self, center: Tensor = torch.zeros(3), radius: float = 1.0):
        super().__init__(center)
        self.radius = radius

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return torch.sqrt(x**2 + y**2 + z**2) - self.radius

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        grad_x = x / (torch.sqrt(x**2 + y**2 + z**2) + EPS)
        grad_y = y / (torch.sqrt(x**2 + y**2 + z**2) + EPS)
        grad_z = z / (torch.sqrt(x**2 + y**2 + z**2) + EPS)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Torus(SDF):
    def __init__(
        self,
        center: Tensor = torch.zeros(3),
        radius: float = 1.0,
        tube_radius: float = 0.5,
    ):
        super().__init__(center)
        self.radius = radius
        self.tube_radius = tube_radius

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        q = torch.sqrt(x**2 + z**2) - self.radius
        return torch.sqrt(q**2 + y**2) - self.tube_radius

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        q = torch.sqrt(x**2 + z**2) - self.radius
        grad_x = x * (q / torch.sqrt(q**2 + y**2))
        grad_y = y / torch.sqrt(q**2 + y**2)
        grad_z = z * (q / torch.sqrt(q**2 + y**2))
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Box(SDF):
    def __init__(self, center: Tensor = torch.zeros(3), size: Tensor = torch.ones(3)):
        super().__init__(center)
        self.size = size

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return torch.max(
            torch.stack(
                [
                    torch.abs(x) - self.size[0] / 2,
                    torch.abs(y) - self.size[1] / 2,
                    torch.abs(z) - self.size[2] / 2,
                ],
            ),
            dim=0,
        ).values

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        grad_x = torch.sign(x)
        grad_y = torch.sign(y)
        grad_z = torch.sign(z)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Cylinder(SDF):
    def __init__(
        self,
        center: Tensor = torch.zeros(3),
        radius: float = 1.0,
        height: float = 1.0,
    ):
        super().__init__(center)
        self.radius = radius
        self.height = height

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return torch.max(
            torch.sqrt(x**2 + z**2) - self.radius,
            torch.abs(y) - self.height / 2,
        )

    def _grad(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        grad_x = x / (torch.sqrt(x**2 + z**2) + EPS)
        grad_y = torch.sign(y)
        grad_z = z / (torch.sqrt(x**2 + z**2) + EPS)
        return torch.stack([grad_x, grad_y, grad_z], dim=1)


class Plane(SDF):
    def __init__(
        self, center: Tensor = torch.zeros(3), normal: Tensor = torch.tensor([0, 0, 1])
    ):
        super().__init__(center)
        self.normal = normal / torch.norm(normal)

    def _f(self, points: Tensor) -> Tensor:
        x, y, z = self._to_xyz(points)
        return x * self.normal[0] + y * self.normal[1] + z * self.normal[2]

    def _grad(self, points: Tensor) -> Tensor:
        return self.normal * torch.ones_like(points)


class Shell(SDF):
    def __init__(self, other: SDF, thickness: float = 1.0, modulation: Callable = None):
        super().__init__(other.center)

        # Set modulation to identity function if not provided
        if modulation is None:

            def modulation(points: Tensor) -> Tensor:
                return torch.ones_like(points[:, 0])

        self.other = other
        self.thickness = thickness
        self.modulation = modulation

    def _f(self, points: Tensor) -> Tensor:
        return torch.abs(
            self.other.sdf(points)
        ) - 0.5 * self.thickness * self.modulation(points)

    def _grad(self, points: Tensor) -> Tensor:
        return self.other._grad(points)
