from abc import ABC, abstractmethod
from math import sqrt

import numpy as np
import torch
from torch import Tensor


class Element(ABC):

    def __init__(self):
        self.nodes: int

    @abstractmethod
    def N(self, xi: Tensor) -> Tensor:
        pass

    @abstractmethod
    def B(self, xi: Tensor) -> Tensor:
        pass

    @abstractmethod
    def ipoints(self) -> Tensor:
        pass

    @abstractmethod
    def iweights(self) -> Tensor:
        pass


class Bar1(Element):
    def __init__(self):
        self.nodes = 2

    def N(self, xi: Tensor) -> Tensor:
        N_1 = 1 - xi[..., 0]
        N_2 = 1 + xi[..., 0]
        return 1 / 2 * torch.stack([N_1, N_2], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        if xi.dim() == 1:
            return torch.tensor([[-0.5, 0.5]])
        else:
            N = xi.shape[0]
            return torch.tensor([[-0.5, 0.5]]).repeat(N, 1, 1)

    def ipoints(self) -> Tensor:
        return torch.tensor([[0.0]])

    def iweights(self) -> Tensor:
        return torch.tensor([2.0])


class Bar2(Element):
    def __init__(self):
        self.nodes = 3

    def N(self, xi: Tensor) -> Tensor:
        N_1 = 1 / 2 * xi[..., 0] * (xi[..., 0] - 1)
        N_2 = 1 / 2 * xi[..., 0] * (xi[..., 0] + 1)
        N_3 = 1 - xi[..., 0] ** 2
        return torch.stack([N_1, N_2, N_3], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        return torch.stack(
            [
                torch.stack(
                    [
                        0.5 * (2 * xi[..., 0] - 1),
                        0.5 * (2 * xi[..., 0] + 1),
                        -2 * xi[..., 0],
                    ],
                    dim=-1,
                )
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor([[-1 / sqrt(3.0)], [1 / sqrt(3.0)]])

    def iweights(self) -> Tensor:
        return torch.tensor([1.0, 1.0])


class Tria1(Element):
    def __init__(self):
        self.nodes = 3

    def N(self, xi: Tensor) -> Tensor:
        N_1 = 1.0 - xi[..., 0] - xi[..., 1]
        N_2 = xi[..., 0]
        N_3 = xi[..., 1]
        return torch.stack([N_1, N_2, N_3], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        if xi.dim() == 1:
            return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        else:
            N = xi.shape[0]
            return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]).repeat(N, 1, 1)

    def ipoints(self) -> Tensor:
        return torch.tensor([[1.0 / 3.0, 1.0 / 3.0]])

    def iweights(self) -> Tensor:
        return torch.tensor([0.5])


class Tria2(Element):
    def __init__(self):
        self.nodes = 6

    def N(self, xi: Tensor) -> Tensor:
        N_1 = (1 - xi[..., 0] - xi[..., 1]) * (1 - 2 * xi[..., 0] - 2 * xi[..., 1])
        N_2 = xi[..., 0] * (2 * xi[..., 0] - 1)
        N_3 = xi[..., 1] * (2 * xi[..., 1] - 1)
        N_4 = 4 * xi[..., 0] * (1 - xi[..., 0] - xi[..., 1])
        N_5 = 4 * xi[..., 0] * xi[..., 1]
        N_6 = 4 * xi[..., 1] * (1 - xi[..., 0] - xi[..., 1])
        return torch.stack([N_1, N_2, N_3, N_4, N_5, N_6], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        zeros = torch.zeros_like(xi[..., 0])
        return torch.stack(
            [
                torch.stack(
                    [
                        4 * xi[..., 0] + 4 * xi[..., 1] - 3,
                        4 * xi[..., 0] - 1,
                        zeros,
                        -4 * (2 * xi[..., 0] + xi[..., 1] - 1),
                        4 * xi[..., 1],
                        -4 * xi[..., 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        4 * xi[..., 0] + 4 * xi[..., 1] - 3,
                        zeros,
                        4 * xi[..., 1] - 1,
                        -4 * xi[..., 0],
                        4 * xi[..., 0],
                        -4 * (xi[..., 0] + 2 * xi[..., 1] - 1),
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]])

    def iweights(self) -> Tensor:
        return torch.tensor([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])


class Quad1(Element):
    def __init__(self):
        self.nodes = 4

    def N(self, xi: Tensor) -> Tensor:
        N_1 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1])
        N_2 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1])
        N_3 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1])
        N_4 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1])
        return 0.25 * torch.stack([N_1, N_2, N_3, N_4], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        return 0.25 * torch.stack(
            [
                torch.stack(
                    [
                        -(1 - xi[..., 1]),
                        (1 - xi[..., 1]),
                        (1.0 + xi[..., 1]),
                        -(1.0 + xi[..., 1]),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(1 - xi[..., 0]),
                        -(1 + xi[..., 0]),
                        (1.0 + xi[..., 0]),
                        (1.0 - xi[..., 0]),
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
                for xi_2 in [-1, 1]
                for xi_1 in [-1, 1]
            ]
        )

    def iweights(self) -> Tensor:
        return torch.tensor([1, 1, 1, 1])


class Quad2(Element):
    def __init__(self):
        self.nodes = 8

    def N(self, xi: Tensor) -> Tensor:
        N_1 = -(1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 0] + xi[..., 1])
        N_2 = -(1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 - xi[..., 0] + xi[..., 1])
        N_3 = -(1 + xi[..., 0]) * (1 + xi[..., 1]) * (1 - xi[..., 0] - xi[..., 1])
        N_4 = -(1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 + xi[..., 0] - xi[..., 1])
        N_5 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 0])
        N_6 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 1])
        N_7 = 2 * (1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 + xi[..., 0])
        N_8 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 1])
        return 0.25 * torch.stack([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        return 0.25 * torch.stack(
            [
                torch.stack(
                    [
                        -(xi[..., 1] - 1) * (2 * xi[..., 0] + xi[..., 1]),
                        -(xi[..., 1] - 1) * (2 * xi[..., 0] - xi[..., 1]),
                        +(xi[..., 1] + 1) * (2 * xi[..., 0] + xi[..., 1]),
                        +(xi[..., 1] + 1) * (2 * xi[..., 0] - xi[..., 1]),
                        +4 * xi[..., 0] * (xi[..., 1] - 1),
                        +2 - 2 * xi[..., 1] ** 2,
                        -4 * xi[..., 0] * (xi[..., 1] + 1),
                        -2 + 2 * xi[..., 1] ** 2,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(xi[..., 0] - 1) * (xi[..., 0] + 2 * xi[..., 1]),
                        -(xi[..., 0] + 1) * (xi[..., 0] - 2 * xi[..., 1]),
                        +(xi[..., 0] + 1) * (xi[..., 0] + 2 * xi[..., 1]),
                        +(xi[..., 0] - 1) * (xi[..., 0] - 2 * xi[..., 1]),
                        -2 + 2 * xi[..., 0] ** 2,
                        -4 * xi[..., 1] * (xi[..., 0] + 1),
                        +2 - 2 * xi[..., 0] ** 2,
                        +4 * (xi[..., 0] - 1) * xi[..., 1],
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
                for xi_2 in [-1, 1]
                for xi_1 in [-1, 1]
            ]
        )

    def iweights(self) -> Tensor:
        return torch.tensor([1, 1, 1, 1])


class Tetra1(Element):
    def __init__(self):
        self.nodes = 4

    def N(self, xi: Tensor) -> Tensor:
        N_1 = 1.0 - xi[..., 0] - xi[..., 1] - xi[..., 2]
        N_2 = xi[..., 0]
        N_3 = xi[..., 1]
        N_4 = xi[..., 2]
        return torch.stack([N_1, N_2, N_3, N_4], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        if xi.dim() == 1:
            return torch.tensor(
                [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]]
            )
        else:
            N = xi.shape[0]
            return torch.tensor(
                [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]]
            ).repeat(N, 1, 1)

    def ipoints(self) -> Tensor:
        return torch.tensor([[0.25, 0.25, 0.25]])

    def iweights(self) -> Tensor:
        return torch.tensor([1.0 / 6.0])


class Tetra2(Element):
    def __init__(self):
        self.nodes = 10

    def N(self, xi: Tensor) -> Tensor:
        N_1 = (1.0 - xi[..., 0] - xi[..., 1] - xi[..., 2]) * (
            2 * (1.0 - xi[..., 0] - xi[..., 1] - xi[..., 2]) - 1
        )
        N_2 = xi[..., 0] * (2 * xi[..., 0] - 1)
        N_3 = xi[..., 1] * (2 * xi[..., 1] - 1)
        N_4 = xi[..., 2] * (2 * xi[..., 2] - 1)
        N_5 = 4 * xi[..., 0] * (1 - xi[..., 0] - xi[..., 1] - xi[..., 2])
        N_6 = 4 * xi[..., 0] * xi[..., 1]
        N_7 = 4 * xi[..., 1] * (1 - xi[..., 0] - xi[..., 1] - xi[..., 2])
        N_8 = 4 * xi[..., 2] * (1 - xi[..., 0] - xi[..., 1] - xi[..., 2])
        N_9 = 4 * xi[..., 0] * xi[..., 2]
        N_10 = 4 * xi[..., 1] * xi[..., 2]
        return torch.stack([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8, N_9, N_10], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        zeros = torch.zeros_like(xi[..., 0])
        return torch.stack(
            [
                torch.stack(
                    [
                        4 * (xi[..., 0] + xi[..., 1] + xi[..., 2]) - 3,
                        4 * xi[..., 0] - 1,
                        zeros,
                        zeros,
                        -4 * (2 * xi[..., 0] + xi[..., 1] + xi[..., 2] - 1),
                        4 * xi[..., 1],
                        -4 * xi[..., 1],
                        -4 * xi[..., 2],
                        4 * xi[..., 2],
                        zeros,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        4 * (xi[..., 0] + xi[..., 1] + xi[..., 2]) - 3,
                        zeros,
                        4 * xi[..., 1] - 1,
                        zeros,
                        -4 * xi[..., 0],
                        4 * xi[..., 0],
                        -4 * (xi[..., 0] + 2 * xi[..., 1] + xi[..., 2] - 1),
                        -4 * xi[..., 2],
                        zeros,
                        4 * xi[..., 2],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        4 * (xi[..., 0] + xi[..., 1] + xi[..., 2]) - 3,
                        zeros,
                        zeros,
                        4 * xi[..., 2] - 1,
                        -4 * xi[..., 0],
                        zeros,
                        -4 * xi[..., 1],
                        -4 * (xi[..., 0] + xi[..., 1] + 2 * xi[..., 2] - 1),
                        4 * xi[..., 0],
                        4 * xi[..., 1],
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor(
            [
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660],
            ]
        )

    def iweights(self) -> Tensor:
        return torch.tensor([0.041666667, 0.041666667, 0.041666667, 0.041666667])


class Hexa1(Element):
    def __init__(self):
        self.nodes = 8

    def N(self, xi: Tensor) -> Tensor:
        N_1 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 - xi[..., 2])
        N_2 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 - xi[..., 2])
        N_3 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 - xi[..., 2])
        N_4 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 - xi[..., 2])
        N_5 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 + xi[..., 2])
        N_6 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 + xi[..., 2])
        N_7 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 + xi[..., 2])
        N_8 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 + xi[..., 2])
        return 0.125 * torch.stack([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8], dim=-1)

    def B(self, xi: Tensor) -> Tensor:
        return 0.125 * torch.stack(
            [
                torch.stack(
                    [
                        -(1.0 - xi[..., 1]) * (1.0 - xi[..., 2]),
                        (1.0 - xi[..., 1]) * (1.0 - xi[..., 2]),
                        (1.0 + xi[..., 1]) * (1.0 - xi[..., 2]),
                        -(1.0 + xi[..., 1]) * (1.0 - xi[..., 2]),
                        -(1.0 - xi[..., 1]) * (1.0 + xi[..., 2]),
                        (1.0 - xi[..., 1]) * (1.0 + xi[..., 2]),
                        (1.0 + xi[..., 1]) * (1.0 + xi[..., 2]),
                        -(1.0 + xi[..., 1]) * (1.0 + xi[..., 2]),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(1.0 - xi[..., 0]) * (1.0 - xi[..., 2]),
                        -(1.0 + xi[..., 0]) * (1.0 - xi[..., 2]),
                        (1.0 + xi[..., 0]) * (1.0 - xi[..., 2]),
                        (1.0 - xi[..., 0]) * (1.0 - xi[..., 2]),
                        -(1.0 - xi[..., 0]) * (1.0 + xi[..., 2]),
                        -(1.0 + xi[..., 0]) * (1.0 + xi[..., 2]),
                        (1.0 + xi[..., 0]) * (1.0 + xi[..., 2]),
                        (1.0 - xi[..., 0]) * (1.0 + xi[..., 2]),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(1.0 - xi[..., 0]) * (1.0 - xi[..., 1]),
                        -(1.0 + xi[..., 0]) * (1.0 - xi[..., 1]),
                        -(1.0 + xi[..., 0]) * (1.0 + xi[..., 1]),
                        -(1.0 - xi[..., 0]) * (1.0 + xi[..., 1]),
                        (1.0 - xi[..., 0]) * (1.0 - xi[..., 1]),
                        (1.0 + xi[..., 0]) * (1.0 - xi[..., 1]),
                        (1.0 + xi[..., 0]) * (1.0 + xi[..., 1]),
                        (1.0 - xi[..., 0]) * (1.0 + xi[..., 1]),
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0), xi_3 / sqrt(3.0)]
                for xi_3 in [-1.0, 1.0]
                for xi_2 in [-1.0, 1.0]
                for xi_1 in [-1.0, 1.0]
            ]
        )

    def iweights(self) -> Tensor:
        return torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


class Hexa2(Element):
    def __init__(self):
        self.nodes = 20

    def N(self, xi: Tensor) -> Tensor:
        N_1 = (
            (1 - xi[..., 0])
            * (1 - xi[..., 1])
            * (1 - xi[..., 2])
            * (-xi[..., 0] - xi[..., 1] - xi[..., 2] - 2)
        )
        N_2 = (
            (1 + xi[..., 0])
            * (1 - xi[..., 1])
            * (1 - xi[..., 2])
            * (+xi[..., 0] - xi[..., 1] - xi[..., 2] - 2)
        )
        N_3 = (
            (1 + xi[..., 0])
            * (1 + xi[..., 1])
            * (1 - xi[..., 2])
            * (+xi[..., 0] + xi[..., 1] - xi[..., 2] - 2)
        )
        N_4 = (
            (1 - xi[..., 0])
            * (1 + xi[..., 1])
            * (1 - xi[..., 2])
            * (-xi[..., 0] + xi[..., 1] - xi[..., 2] - 2)
        )

        N_5 = (
            (1 - xi[..., 0])
            * (1 - xi[..., 1])
            * (1 + xi[..., 2])
            * (-xi[..., 0] - xi[..., 1] + xi[..., 2] - 2)
        )
        N_6 = (
            (1 + xi[..., 0])
            * (1 - xi[..., 1])
            * (1 + xi[..., 2])
            * (+xi[..., 0] - xi[..., 1] + xi[..., 2] - 2)
        )
        N_7 = (
            (1 + xi[..., 0])
            * (1 + xi[..., 1])
            * (1 + xi[..., 2])
            * (+xi[..., 0] + xi[..., 1] + xi[..., 2] - 2)
        )
        N_8 = (
            (1 - xi[..., 0])
            * (1 + xi[..., 1])
            * (1 + xi[..., 2])
            * (-xi[..., 0] + xi[..., 1] + xi[..., 2] - 2)
        )
        N_9 = 2 * (1 - xi[..., 0] ** 2) * (1 - xi[..., 1]) * (1 - xi[..., 2])
        N_10 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1] ** 2) * (1 - xi[..., 2])
        N_11 = 2 * (1 - xi[..., 0] ** 2) * (1 + xi[..., 1]) * (1 - xi[..., 2])
        N_12 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1] ** 2) * (1 - xi[..., 2])
        N_13 = 2 * (1 - xi[..., 0] ** 2) * (1 - xi[..., 1]) * (1 + xi[..., 2])
        N_14 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1] ** 2) * (1 + xi[..., 2])
        N_15 = 2 * (1 - xi[..., 0] ** 2) * (1 + xi[..., 1]) * (1 + xi[..., 2])
        N_16 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1] ** 2) * (1 + xi[..., 2])
        N_17 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 - xi[..., 2] ** 2)
        N_18 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 - xi[..., 2] ** 2)
        N_19 = 2 * (1 + xi[..., 0]) * (1 + xi[..., 1]) * (1 - xi[..., 2] ** 2)
        N_20 = 2 * (1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 - xi[..., 2] ** 2)
        return 0.125 * torch.stack(
            [
                N_1,
                N_2,
                N_3,
                N_4,
                N_5,
                N_6,
                N_7,
                N_8,
                N_9,
                N_10,
                N_11,
                N_12,
                N_13,
                N_14,
                N_15,
                N_16,
                N_17,
                N_18,
                N_19,
                N_20,
            ],
            dim=-1,
        )

    def B(self, xi: Tensor) -> Tensor:
        return 0.125 * torch.stack(
            [
                torch.stack(
                    [
                        +(xi[..., 1] - 1)
                        * (xi[..., 2] - 1)
                        * (+2 * xi[..., 0] + xi[..., 1] + xi[..., 2] + 1),
                        -(xi[..., 1] - 1)
                        * (xi[..., 2] - 1)
                        * (-2 * xi[..., 0] + xi[..., 1] + xi[..., 2] + 1),
                        -(xi[..., 1] + 1)
                        * (xi[..., 2] - 1)
                        * (+2 * xi[..., 0] + xi[..., 1] - xi[..., 2] - 1),
                        +(xi[..., 1] + 1)
                        * (xi[..., 2] - 1)
                        * (-2 * xi[..., 0] + xi[..., 1] - xi[..., 2] - 1),
                        -(xi[..., 1] - 1)
                        * (xi[..., 2] + 1)
                        * (+2 * xi[..., 0] + xi[..., 1] - xi[..., 2] + 1),
                        +(xi[..., 1] - 1)
                        * (xi[..., 2] + 1)
                        * (-2 * xi[..., 0] + xi[..., 1] - xi[..., 2] + 1),
                        +(xi[..., 1] + 1)
                        * (xi[..., 2] + 1)
                        * (+2 * xi[..., 0] + xi[..., 1] + xi[..., 2] - 1),
                        -(xi[..., 1] + 1)
                        * (xi[..., 2] + 1)
                        * (-2 * xi[..., 0] + xi[..., 1] + xi[..., 2] - 1),
                        -4 * xi[..., 0] * (xi[..., 1] - 1) * (xi[..., 2] - 1),
                        +2 * (xi[..., 1] ** 2 - 1) * (xi[..., 2] - 1),
                        +4 * xi[..., 0] * (xi[..., 1] + 1) * (xi[..., 2] - 1),
                        -2 * (xi[..., 1] ** 2 - 1) * (xi[..., 2] - 1),
                        +4 * xi[..., 0] * (xi[..., 1] - 1) * (xi[..., 2] + 1),
                        -2 * (xi[..., 1] ** 2 - 1) * (xi[..., 2] + 1),
                        -4 * xi[..., 0] * (xi[..., 1] + 1) * (xi[..., 2] + 1),
                        +2 * (xi[..., 1] ** 2 - 1) * (xi[..., 2] + 1),
                        -2 * (xi[..., 1] - 1) * (xi[..., 2] ** 2 - 1),
                        +2 * (xi[..., 1] - 1) * (xi[..., 2] ** 2 - 1),
                        -2 * (xi[..., 1] + 1) * (xi[..., 2] ** 2 - 1),
                        +2 * (xi[..., 1] + 1) * (xi[..., 2] ** 2 - 1),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        +(xi[..., 0] - 1)
                        * (xi[..., 2] - 1)
                        * (xi[..., 0] + 2 * xi[..., 1] + xi[..., 2] + 1),
                        +(xi[..., 0] + 1)
                        * (xi[..., 2] - 1)
                        * (xi[..., 0] - 2 * xi[..., 1] - xi[..., 2] - 1),
                        -(xi[..., 0] + 1)
                        * (xi[..., 2] - 1)
                        * (xi[..., 0] + 2 * xi[..., 1] - xi[..., 2] - 1),
                        -(xi[..., 0] - 1)
                        * (xi[..., 2] - 1)
                        * (xi[..., 0] - 2 * xi[..., 1] + xi[..., 2] + 1),
                        -(xi[..., 0] - 1)
                        * (xi[..., 2] + 1)
                        * (xi[..., 0] + 2 * xi[..., 1] - xi[..., 2] + 1),
                        -(xi[..., 0] + 1)
                        * (xi[..., 2] + 1)
                        * (xi[..., 0] - 2 * xi[..., 1] + xi[..., 2] - 1),
                        +(xi[..., 0] + 1)
                        * (xi[..., 2] + 1)
                        * (xi[..., 0] + 2 * xi[..., 1] + xi[..., 2] - 1),
                        +(xi[..., 0] - 1)
                        * (xi[..., 2] + 1)
                        * (xi[..., 0] - 2 * xi[..., 1] - xi[..., 2] + 1),
                        -2 * (xi[..., 0] ** 2 - 1) * (xi[..., 2] - 1),
                        +4 * xi[..., 1] * (xi[..., 0] + 1) * (xi[..., 2] - 1),
                        +2 * (xi[..., 0] ** 2 - 1) * (xi[..., 2] - 1),
                        -4 * xi[..., 1] * (xi[..., 0] - 1) * (xi[..., 2] - 1),
                        +2 * (xi[..., 0] ** 2 - 1) * (xi[..., 2] + 1),
                        -4 * xi[..., 1] * (xi[..., 0] + 1) * (xi[..., 2] + 1),
                        -2 * (xi[..., 0] ** 2 - 1) * (xi[..., 2] + 1),
                        +4 * xi[..., 1] * (xi[..., 0] - 1) * (xi[..., 2] + 1),
                        -2 * (xi[..., 0] - 1) * (xi[..., 2] ** 2 - 1),
                        +2 * (xi[..., 0] + 1) * (xi[..., 2] ** 2 - 1),
                        -2 * (xi[..., 0] + 1) * (xi[..., 2] ** 2 - 1),
                        +2 * (xi[..., 0] - 1) * (xi[..., 2] ** 2 - 1),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        +(xi[..., 0] - 1)
                        * (xi[..., 1] - 1)
                        * (xi[..., 0] + xi[..., 1] + 2 * xi[..., 2] + 1),
                        +(xi[..., 0] + 1)
                        * (xi[..., 1] - 1)
                        * (xi[..., 0] - xi[..., 1] - 2 * xi[..., 2] - 1),
                        -(xi[..., 0] + 1)
                        * (xi[..., 1] + 1)
                        * (xi[..., 0] + xi[..., 1] - 2 * xi[..., 2] - 1),
                        -(xi[..., 0] - 1)
                        * (xi[..., 1] + 1)
                        * (xi[..., 0] - xi[..., 1] + 2 * xi[..., 2] + 1),
                        -(xi[..., 0] - 1)
                        * (xi[..., 1] - 1)
                        * (xi[..., 0] + xi[..., 1] - 2 * xi[..., 2] + 1),
                        -(xi[..., 0] + 1)
                        * (xi[..., 1] - 1)
                        * (xi[..., 0] - xi[..., 1] + 2 * xi[..., 2] - 1),
                        +(xi[..., 0] + 1)
                        * (xi[..., 1] + 1)
                        * (xi[..., 0] + xi[..., 1] + 2 * xi[..., 2] - 1),
                        +(xi[..., 0] - 1)
                        * (xi[..., 1] + 1)
                        * (xi[..., 0] - xi[..., 1] - 2 * xi[..., 2] + 1),
                        -2 * (xi[..., 0] ** 2 - 1) * (xi[..., 1] - 1),
                        +2 * (xi[..., 0] + 1) * (xi[..., 1] ** 2 - 1),
                        +2 * (xi[..., 0] ** 2 - 1) * (xi[..., 1] + 1),
                        -2 * (xi[..., 0] - 1) * (xi[..., 1] ** 2 - 1),
                        +2 * (xi[..., 0] ** 2 - 1) * (xi[..., 1] - 1),
                        -2 * (xi[..., 0] + 1) * (xi[..., 1] ** 2 - 1),
                        -2 * (xi[..., 0] ** 2 - 1) * (xi[..., 1] + 1),
                        +2 * (xi[..., 0] - 1) * (xi[..., 1] ** 2 - 1),
                        -4 * (xi[..., 0] - 1) * (xi[..., 1] - 1) * xi[..., 2],
                        +4 * (xi[..., 0] + 1) * (xi[..., 1] - 1) * xi[..., 2],
                        -4 * (xi[..., 0] + 1) * (xi[..., 1] + 1) * xi[..., 2],
                        +4 * (xi[..., 0] - 1) * (xi[..., 1] + 1) * xi[..., 2],
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    def ipoints(self) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0), xi_3 / sqrt(3.0)]
                for xi_3 in [-1.0, 1.0]
                for xi_2 in [-1.0, 1.0]
                for xi_1 in [-1.0, 1.0]
            ]
        )

    def iweights(self) -> Tensor:
        return torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def linear_to_quadratic(nodes: Tensor, elements: Tensor) -> tuple[Tensor, Tensor]:
    if elements.shape[1] == 2:
        # Bar1 element
        edges = torch.tensor([[0, 1]])
    elif elements.shape[1] == 3 and nodes.shape[1] == 2:
        # Tri1 element
        edges = torch.tensor([[0, 1], [1, 2], [2, 0]])
    elif elements.shape[1] == 4 and nodes.shape[1] == 2:
        # Quad1 element
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    elif elements.shape[1] == 4 and nodes.shape[1] == 3:
        # Tetra1 element
        edges = torch.tensor([[0, 1], [1, 2], [0, 2], [3, 0], [1, 3], [2, 3]])
    elif elements.shape[1] == 8 and nodes.shape[1] == 3:
        # Hexa1 element
        edges = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
        )
    else:
        raise Exception(
            "The element type is not supported for conversion to quadratic."
            "Maybe the element is already quadratic?"
        )

    # Vectorize edges for all elements and sort each pair with increasing node IDs
    edge_vector = elements[:, edges].reshape(-1, 2)
    edge_vector, _ = torch.sort(edge_vector, dim=1)

    # Find unique edges and get the inverse mapping
    if edge_vector.device.type == "cuda":
        # Default torch implementation for CUDA
        unique_edges, inv_indices = torch.unique(
            edge_vector, dim=0, return_inverse=True
        )
    else:
        # Numpy implementation for CPU (breaks gradient flow, but should not be needed)
        edge_vector_np = edge_vector.numpy()
        dt = np.dtype((np.void, edge_vector_np.dtype.itemsize * edge_vector.shape[1]))
        b = np.ascontiguousarray(edge_vector).view(dt)
        _, unique_indices, inv_indices_np = np.unique(
            b, return_index=True, return_inverse=True
        )
        unique_edges = edge_vector[unique_indices]
        inv_indices = torch.as_tensor(inv_indices_np)

    # Compute nodes
    mid_node_coords = (nodes[unique_edges[:, 0]] + nodes[unique_edges[:, 1]]) / 2.0
    new_nodes = torch.cat([nodes, mid_node_coords], dim=0)

    # Compute elements
    mid_node_map = inv_indices.view(elements.shape[0], -1) + nodes.shape[0]
    new_elements = torch.cat([elements, mid_node_map], dim=1)

    return new_nodes, new_elements
