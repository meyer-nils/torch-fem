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

    def B(self, _: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    def ipoints(self) -> torch.Tensor:
        return torch.tensor([[1.0 / 3.0, 1.0 / 3.0]])

    def iweights(self) -> torch.Tensor:
        return torch.tensor([0.5])


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
        return 0.25 * torch.tensor(
            [
                [-(1 - xi[1]), (1 - xi[1]), (1.0 + xi[1]), -(1.0 + xi[1])],
                [-(1 - xi[0]), -(1 + xi[0]), (1.0 + xi[0]), (1.0 - xi[0])],
            ]
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

    def B(self, _: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]]
        )

    def ipoints(self) -> torch.Tensor:
        return torch.tensor([[0.25, 0.25, 0.25]])

    def iweights(self) -> torch.Tensor:
        return torch.tensor([1.0 / 6.0])


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
        return 0.125 * torch.tensor(
            [
                [
                    -(1.0 - xi[1]) * (1.0 - xi[2]),
                    (1.0 - xi[1]) * (1.0 - xi[2]),
                    (1.0 + xi[1]) * (1.0 - xi[2]),
                    -(1.0 + xi[1]) * (1.0 - xi[2]),
                    -(1.0 - xi[1]) * (1.0 + xi[2]),
                    (1.0 - xi[1]) * (1.0 + xi[2]),
                    (1.0 + xi[1]) * (1.0 + xi[2]),
                    -(1.0 + xi[1]) * (1.0 + xi[2]),
                ],
                [
                    -(1.0 - xi[0]) * (1.0 - xi[2]),
                    -(1.0 + xi[0]) * (1.0 - xi[2]),
                    (1.0 + xi[0]) * (1.0 - xi[2]),
                    (1.0 - xi[0]) * (1.0 - xi[2]),
                    -(1.0 - xi[0]) * (1.0 + xi[2]),
                    -(1.0 + xi[0]) * (1.0 + xi[2]),
                    (1.0 + xi[0]) * (1.0 + xi[2]),
                    (1.0 - xi[0]) * (1.0 + xi[2]),
                ],
                [
                    -(1.0 - xi[0]) * (1.0 - xi[1]),
                    -(1.0 + xi[0]) * (1.0 - xi[1]),
                    -(1.0 + xi[0]) * (1.0 + xi[1]),
                    -(1.0 - xi[0]) * (1.0 + xi[1]),
                    (1.0 - xi[0]) * (1.0 - xi[1]),
                    (1.0 + xi[0]) * (1.0 - xi[1]),
                    (1.0 + xi[0]) * (1.0 + xi[1]),
                    (1.0 - xi[0]) * (1.0 + xi[1]),
                ],
            ]
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
