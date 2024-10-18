from math import sqrt

import torch


class Tria1:
    def __init__(self):
        self.nodes = 3

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        N_1 = 1.0 - xi[..., 0] - xi[..., 1]
        N_2 = xi[..., 0]
        N_3 = xi[..., 1]
        return torch.stack([N_1, N_2, N_3], dim=-1)

    def B(self, xi: torch.Tensor) -> torch.Tensor:
        if xi.dim() == 1:
            return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        else:
            N = xi.shape[0]
            return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]).repeat(N, 1, 1)

    def ipoints(self) -> torch.Tensor:
        return torch.tensor([[1.0 / 3.0, 1.0 / 3.0]])

    def iweights(self) -> torch.Tensor:
        return torch.tensor([0.5])


class Tria2:
    def __init__(self):
        self.nodes = 6

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        N_1 = (1 - xi[..., 0] - xi[..., 1]) * (1 - 2 * xi[..., 0] - 2 * xi[..., 1])
        N_2 = xi[..., 0] * (2 * xi[..., 0] - 1)
        N_3 = xi[..., 1] * (2 * xi[..., 1] - 1)
        N_4 = 4 * xi[..., 0] * (1 - xi[..., 0] - xi[..., 1])
        N_5 = 4 * xi[..., 0] * xi[..., 1]
        N_6 = 4 * xi[..., 1] * (1 - xi[..., 0] - xi[..., 1])
        return torch.stack([N_1, N_2, N_3, N_4, N_5, N_6], dim=-1)

    def B(self, xi: torch.Tensor) -> torch.Tensor:
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

    def ipoints(self) -> torch.Tensor:
        return torch.tensor([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]])

    def iweights(self) -> torch.Tensor:
        return torch.tensor([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])


class Quad1:
    def __init__(self):
        self.nodes = 4

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        N_1 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1])
        N_2 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1])
        N_3 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1])
        N_4 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1])
        return 0.25 * torch.stack([N_1, N_2, N_3, N_4], dim=-1)

    def B(self, xi: torch.Tensor) -> torch.Tensor:
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

    def ipoints(self) -> torch.Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
                for xi_2 in [-1, 1]
                for xi_1 in [-1, 1]
            ]
        )

    def iweights(self) -> torch.Tensor:
        return torch.tensor([1, 1, 1, 1])


class Quad2:
    def __init__(self):
        self.nodes = 8

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        N_1 = -(1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 0] + xi[..., 1])
        N_2 = -(1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 - xi[..., 0] + xi[..., 1])
        N_3 = -(1 + xi[..., 0]) * (1 + xi[..., 1]) * (1 - xi[..., 0] - xi[..., 1])
        N_4 = -(1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 + xi[..., 0] - xi[..., 1])
        N_5 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 0])
        N_6 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 1])
        N_7 = 2 * (1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 + xi[..., 0])
        N_8 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 1])
        return 0.25 * torch.stack([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8], dim=-1)

    def B(self, xi: torch.Tensor) -> torch.Tensor:
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

    def ipoints(self) -> torch.Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
                for xi_2 in [-1, 1]
                for xi_1 in [-1, 1]
            ]
        )

    def iweights(self) -> torch.Tensor:
        return torch.tensor([1, 1, 1, 1])


class Tetra1:
    def __init__(self):
        self.nodes = 4

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        N_1 = 1.0 - xi[..., 0] - xi[..., 1] - xi[..., 2]
        N_2 = xi[..., 0]
        N_3 = xi[..., 1]
        N_4 = xi[..., 2]
        return torch.stack([N_1, N_2, N_3, N_4], dim=-1)

    def B(self, xi: torch.Tensor) -> torch.Tensor:
        if xi.dim() == 1:
            return torch.tensor(
                [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]]
            )
        else:
            N = xi.shape[0]
            return torch.tensor(
                [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]]
            ).repeat(N, 1, 1)

    def ipoints(self) -> torch.Tensor:
        return torch.tensor([[0.25, 0.25, 0.25]])

    def iweights(self) -> torch.Tensor:
        return torch.tensor([1.0 / 6.0])


class Tetra2:
    def __init__(self):
        self.nodes = 10

    def N(self, xi: torch.Tensor) -> torch.Tensor:
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

    def B(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    4 * (xi[..., 0] + xi[..., 1] + xi[..., 2]) - 3,
                    4 * xi[..., 0] - 1,
                    0,
                    0,
                    -4 * (2 * xi[..., 0] + xi[..., 1] + xi[..., 2] - 1),
                    4 * xi[..., 1],
                    -4 * xi[..., 1],
                    -4 * xi[..., 2],
                    4 * xi[..., 2],
                    0,
                ],
                [
                    4 * (xi[..., 0] + xi[..., 1] + xi[..., 2]) - 3,
                    0,
                    4 * xi[..., 1] - 1,
                    0,
                    -4 * xi[..., 0],
                    4 * xi[..., 0],
                    -4 * (xi[..., 0] + 2 * xi[..., 1] + xi[..., 2] - 1),
                    -4 * xi[..., 2],
                    0,
                    4 * xi[..., 2],
                ],
                [
                    4 * (xi[..., 0] + xi[..., 1] + xi[..., 2]) - 3,
                    0,
                    0,
                    4 * xi[..., 2] - 1,
                    -4 * xi[..., 0],
                    0,
                    -4 * xi[..., 1],
                    -4 * (xi[..., 0] + xi[..., 1] + 2 * xi[..., 2] - 1),
                    4 * xi[..., 0],
                    4 * xi[..., 1],
                ],
            ]
        )

    def ipoints(self) -> torch.Tensor:
        return torch.tensor(
            [
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660],
            ]
        )

    def iweights(self) -> torch.Tensor:
        return torch.tensor([0.041666667, 0.041666667, 0.041666667, 0.041666667])


class Hexa1:
    def __init__(self):
        self.nodes = 8

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        N_1 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 - xi[..., 2])
        N_2 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 - xi[..., 2])
        N_3 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 - xi[..., 2])
        N_4 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 - xi[..., 2])
        N_5 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 + xi[..., 2])
        N_6 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 + xi[..., 2])
        N_7 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 + xi[..., 2])
        N_8 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 + xi[..., 2])
        return 0.125 * torch.stack([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8], dim=-1)

    def B(self, xi: torch.Tensor) -> torch.Tensor:
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

    def ipoints(self) -> torch.Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0), xi_3 / sqrt(3.0)]
                for xi_3 in [-1.0, 1.0]
                for xi_2 in [-1.0, 1.0]
                for xi_1 in [-1.0, 1.0]
            ]
        )

    def iweights(self) -> torch.Tensor:
        return torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


class Hexa2:
    def __init__(self):
        self.nodes = 20

    def N(self, xi: torch.Tensor) -> torch.Tensor:
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

    def B(self, xi: torch.Tensor) -> torch.Tensor:
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

    def ipoints(self) -> torch.Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0), xi_3 / sqrt(3.0)]
                for xi_3 in [-1.0, 1.0]
                for xi_2 in [-1.0, 1.0]
                for xi_1 in [-1.0, 1.0]
            ]
        )

    def iweights(self) -> torch.Tensor:
        return torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def linear_to_quadratic(
    nodes: torch.Tensor, elements: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    new_nodes = nodes.tolist()
    midpoints = {}

    def get_midpoint_index(n1, n2):
        if (n1, n2) in midpoints:
            return midpoints[(n1, n2)]
        if (n2, n1) in midpoints:
            return midpoints[(n2, n1)]

        midpoint = (nodes[n1] + nodes[n2]) / 2
        mid_index = len(new_nodes)
        new_nodes.append(midpoint)
        midpoints[(n1, n2)] = mid_index
        return mid_index

    new_elements = []
    for element in elements:
        if len(element) == 3 and nodes.shape[1] == 2:
            n1, n2, n3 = element.numpy()
            m12 = get_midpoint_index(n1, n2)
            m23 = get_midpoint_index(n2, n3)
            m31 = get_midpoint_index(n3, n1)
            new_elements.append([n1, n2, n3, m12, m23, m31])
        if len(element) == 4 and nodes.shape[1] == 2:
            n1, n2, n3, n4 = element.numpy()
            m12 = get_midpoint_index(n1, n2)
            m23 = get_midpoint_index(n2, n3)
            m34 = get_midpoint_index(n3, n4)
            m41 = get_midpoint_index(n4, n1)
            new_elements.append([n1, n2, n3, n4, m12, m23, m34, m41])
        if len(element) == 4 and nodes.shape[1] == 3:
            n1, n2, n3, n4 = element.numpy()
            m12 = get_midpoint_index(n1, n2)
            m23 = get_midpoint_index(n2, n3)
            m13 = get_midpoint_index(n1, n3)
            m41 = get_midpoint_index(n4, n1)
            m24 = get_midpoint_index(n2, n4)
            m34 = get_midpoint_index(n3, n4)
            new_elements.append([n1, n2, n3, n4, m12, m23, m13, m41, m24, m34])
        if len(element) == 8 and nodes.shape[1] == 3:
            n1, n2, n3, n4, n5, n6, n7, n8 = element.numpy()
            m12 = get_midpoint_index(n1, n2)
            m23 = get_midpoint_index(n2, n3)
            m34 = get_midpoint_index(n3, n4)
            m41 = get_midpoint_index(n4, n1)
            m56 = get_midpoint_index(n5, n6)
            m67 = get_midpoint_index(n6, n7)
            m78 = get_midpoint_index(n7, n8)
            m85 = get_midpoint_index(n8, n5)
            m15 = get_midpoint_index(n1, n5)
            m26 = get_midpoint_index(n2, n6)
            m37 = get_midpoint_index(n3, n7)
            m48 = get_midpoint_index(n4, n8)
            new_elements.append(
                [
                    n1,
                    n2,
                    n3,
                    n4,
                    n5,
                    n6,
                    n7,
                    n8,
                    m12,
                    m23,
                    m34,
                    m41,
                    m56,
                    m67,
                    m78,
                    m85,
                    m15,
                    m26,
                    m37,
                    m48,
                ]
            )

    return torch.tensor(new_nodes), torch.tensor(new_elements)
