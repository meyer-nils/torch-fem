import torch
from torch import Tensor


def stress2voigt(sigma: Tensor) -> Tensor:
    """Convert a stress tensor to Voigt notation."""
    if sigma.shape[-1] == 2 and sigma.shape[-2] == 2:
        return torch.stack(
            [
                sigma[..., 0, 0],
                sigma[..., 1, 1],
                sigma[..., 0, 1],
            ],
            dim=-1,
        )
    elif sigma.shape[-1] == 3 and sigma.shape[-2] == 3:
        return torch.stack(
            [
                sigma[..., 0, 0],
                sigma[..., 1, 1],
                sigma[..., 2, 2],
                sigma[..., 1, 2],
                sigma[..., 0, 2],
                sigma[..., 0, 1],
            ],
            dim=-1,
        )
    else:
        raise ValueError("Invalid shape for stress tensor.")


def strain2voigt(epsilon: Tensor) -> Tensor:
    """Convert a strain tensor to Voigt notation."""
    if epsilon.shape[-1] == 2 and epsilon.shape[-2] == 2:
        return torch.stack(
            [
                epsilon[..., 0, 0],
                epsilon[..., 1, 1],
                2 * epsilon[..., 0, 1],
            ],
            dim=-1,
        )
    elif epsilon.shape[-1] == 3 and epsilon.shape[-2] == 3:
        return torch.stack(
            [
                epsilon[..., 0, 0],
                epsilon[..., 1, 1],
                epsilon[..., 2, 2],
                2 * epsilon[..., 1, 2],
                2 * epsilon[..., 0, 2],
                2 * epsilon[..., 0, 1],
            ],
            dim=-1,
        )
    else:
        raise ValueError("Invalid shape for strain tensor.")


def stiffness2voigt(C: Tensor) -> Tensor:
    """Convert a stiffness tensor to Voigt notation."""
    if C.shape[-1] == 2 and C.shape[-2] == 2:
        return torch.stack(
            [
                torch.stack(
                    [C[..., 0, 0, 0, 0], C[..., 0, 0, 1, 1], C[..., 0, 0, 0, 1]], dim=-1
                ),
                torch.stack(
                    [C[..., 1, 1, 0, 0], C[..., 1, 1, 1, 1], C[..., 1, 1, 0, 1]], dim=-1
                ),
                torch.stack(
                    [C[..., 0, 1, 0, 0], C[..., 0, 1, 1, 1], C[..., 0, 1, 0, 1]], dim=-1
                ),
            ],
            dim=-1,
        )
    elif C.shape[-1] == 3 and C.shape[-2] == 3:
        return torch.stack(
            [
                torch.stack(
                    [
                        C[..., 0, 0, 0, 0],
                        C[..., 0, 0, 1, 1],
                        C[..., 0, 0, 2, 2],
                        C[..., 0, 0, 1, 2],
                        C[..., 0, 0, 0, 2],
                        C[..., 0, 0, 0, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        C[..., 1, 1, 0, 0],
                        C[..., 1, 1, 1, 1],
                        C[..., 1, 1, 2, 2],
                        C[..., 1, 1, 1, 2],
                        C[..., 1, 1, 0, 2],
                        C[..., 1, 1, 0, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        C[..., 2, 2, 0, 0],
                        C[..., 2, 2, 1, 1],
                        C[..., 2, 2, 2, 2],
                        C[..., 2, 2, 1, 2],
                        C[..., 2, 2, 0, 2],
                        C[..., 2, 2, 0, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        C[..., 1, 2, 0, 0],
                        C[..., 1, 2, 1, 1],
                        C[..., 1, 2, 2, 2],
                        C[..., 1, 2, 1, 2],
                        C[..., 1, 2, 0, 2],
                        C[..., 1, 2, 0, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        C[..., 0, 2, 0, 0],
                        C[..., 0, 2, 1, 1],
                        C[..., 0, 2, 2, 2],
                        C[..., 0, 2, 1, 2],
                        C[..., 0, 2, 0, 2],
                        C[..., 0, 2, 0, 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        C[..., 0, 1, 0, 0],
                        C[..., 0, 1, 1, 1],
                        C[..., 0, 1, 2, 2],
                        C[..., 0, 1, 1, 2],
                        C[..., 0, 1, 0, 2],
                        C[..., 0, 1, 0, 1],
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )


def voigt2stress(voigt: Tensor) -> Tensor:
    """Convert a stress tensor from Voigt notation."""
    if voigt.shape[-1] == 3:
        return torch.stack(
            [
                torch.stack([voigt[..., 0], voigt[..., 2]], dim=-1),
                torch.stack([voigt[..., 2], voigt[..., 1]], dim=-1),
            ],
            dim=-1,
        )
    elif voigt.shape[-1] == 6:
        return torch.stack(
            [
                torch.stack([voigt[..., 0], voigt[..., 5], voigt[..., 4]], dim=-1),
                torch.stack([voigt[..., 5], voigt[..., 1], voigt[..., 3]], dim=-1),
                torch.stack([voigt[..., 4], voigt[..., 3], voigt[..., 2]], dim=-1),
            ],
            dim=-1,
        )
    else:
        raise ValueError("Invalid shape for Voigt notation.")


def voigt2strain(voigt: Tensor) -> Tensor:
    """Convert a strain tensor from Voigt notation."""
    if voigt.shape[-1] == 3:
        return torch.stack(
            [
                torch.stack([voigt[..., 0], 0.5 * voigt[..., 2]], dim=-1),
                torch.stack([0.5 * voigt[..., 2], voigt[..., 1]], dim=-1),
            ],
            dim=-1,
        )
    elif voigt.shape[-1] == 6:
        return torch.stack(
            [
                torch.stack(
                    [voigt[..., 0], 0.5 * voigt[..., 5], 0.5 * voigt[..., 4]], dim=-1
                ),
                torch.stack(
                    [0.5 * voigt[..., 5], voigt[..., 1], 0.5 * voigt[..., 3]], dim=-1
                ),
                torch.stack(
                    [0.5 * voigt[..., 4], 0.5 * voigt[..., 3], voigt[..., 2]], dim=-1
                ),
            ],
            dim=-1,
        )
    else:
        raise ValueError("Invalid shape for Voigt notation.")


def voigt2stiffness(voigt: Tensor) -> Tensor:
    """Convert a stiffness tensor from Voigt notation."""
    if voigt.shape[-1] == 3:
        C = torch.zeros(voigt.shape[0], 2, 2, 2, 2)
        C[..., 0, 0, 0, 0] = voigt[..., 0, 0]
        C[..., 1, 1, 1, 1] = voigt[..., 1, 1]
        C[..., 0, 0, 1, 1] = voigt[..., 0, 1]
        C[..., 1, 1, 0, 0] = voigt[..., 1, 0]
        C[..., 0, 0, 0, 1] = voigt[..., 0, 2]
        C[..., 0, 0, 1, 0] = voigt[..., 0, 2]
        C[..., 1, 0, 0, 0] = voigt[..., 2, 0]
        C[..., 0, 1, 0, 0] = voigt[..., 2, 0]
        C[..., 1, 0, 1, 1] = voigt[..., 2, 1]
        C[..., 0, 1, 1, 1] = voigt[..., 2, 1]
        C[..., 1, 1, 1, 0] = voigt[..., 1, 2]
        C[..., 1, 1, 0, 1] = voigt[..., 1, 2]
        C[..., 1, 0, 0, 1] = voigt[..., 2, 2]
        C[..., 0, 1, 0, 1] = voigt[..., 2, 2]
        C[..., 1, 0, 1, 0] = voigt[..., 2, 2]
        C[..., 0, 1, 1, 0] = voigt[..., 2, 2]
        return C
    elif voigt.shape[-1] == 6:
        C = torch.zeros(voigt.shape[0], 3, 3, 3, 3)
        C[..., 0, 0, 0, 0] = voigt[..., 0, 0]
        C[..., 1, 1, 1, 1] = voigt[..., 1, 1]
        C[..., 2, 2, 2, 2] = voigt[..., 2, 2]

        C[..., 0, 0, 1, 1] = voigt[..., 0, 1]
        C[..., 1, 1, 0, 0] = voigt[..., 1, 0]

        C[..., 0, 0, 2, 2] = voigt[..., 0, 2]
        C[..., 2, 2, 0, 0] = voigt[..., 2, 0]

        C[..., 1, 2, 1, 2] = voigt[..., 3, 3]
        C[..., 2, 1, 1, 2] = voigt[..., 3, 3]
        C[..., 1, 2, 2, 1] = voigt[..., 3, 3]
        C[..., 2, 1, 2, 1] = voigt[..., 3, 3]

        C[..., 0, 2, 0, 2] = voigt[..., 4, 4]
        C[..., 2, 0, 0, 2] = voigt[..., 4, 4]
        C[..., 0, 2, 2, 0] = voigt[..., 4, 4]
        C[..., 2, 0, 2, 0] = voigt[..., 4, 4]

        C[..., 0, 1, 0, 1] = voigt[..., 5, 5]
        C[..., 1, 0, 0, 1] = voigt[..., 5, 5]
        C[..., 0, 1, 1, 0] = voigt[..., 5, 5]
        C[..., 1, 0, 1, 0] = voigt[..., 5, 5]

        C[..., 0, 0, 1, 2] = voigt[..., 0, 3]
        C[..., 0, 0, 2, 1] = voigt[..., 0, 3]
        C[..., 1, 2, 0, 0] = voigt[..., 3, 0]
        C[..., 2, 1, 0, 0] = voigt[..., 3, 0]

        C[..., 0, 0, 0, 2] = voigt[..., 0, 4]
        C[..., 0, 0, 2, 0] = voigt[..., 0, 4]
        C[..., 2, 0, 0, 0] = voigt[..., 4, 0]
        C[..., 0, 2, 0, 0] = voigt[..., 4, 0]

        C[..., 0, 0, 0, 1] = voigt[..., 0, 5]
        C[..., 0, 0, 1, 0] = voigt[..., 0, 5]
        C[..., 1, 0, 0, 0] = voigt[..., 5, 0]
        C[..., 0, 1, 0, 0] = voigt[..., 5, 0]

        C[..., 1, 1, 1, 2] = voigt[..., 1, 3]
        C[..., 1, 1, 2, 1] = voigt[..., 1, 3]
        C[..., 1, 2, 1, 1] = voigt[..., 3, 1]
        C[..., 2, 1, 1, 1] = voigt[..., 3, 1]

        C[..., 1, 1, 0, 2] = voigt[..., 1, 4]
        C[..., 1, 1, 2, 0] = voigt[..., 1, 4]
        C[..., 0, 2, 1, 1] = voigt[..., 4, 1]
        C[..., 2, 0, 1, 1] = voigt[..., 4, 1]

        C[..., 1, 1, 0, 1] = voigt[..., 1, 5]
        C[..., 1, 1, 1, 0] = voigt[..., 1, 5]
        C[..., 0, 1, 1, 1] = voigt[..., 5, 1]
        C[..., 1, 0, 1, 1] = voigt[..., 5, 1]

        C[..., 2, 2, 1, 2] = voigt[..., 2, 3]
        C[..., 2, 2, 2, 1] = voigt[..., 2, 3]
        C[..., 1, 2, 2, 2] = voigt[..., 3, 2]
        C[..., 2, 1, 2, 2] = voigt[..., 3, 2]

        C[..., 2, 2, 0, 2] = voigt[..., 2, 4]
        C[..., 2, 2, 2, 0] = voigt[..., 2, 4]
        C[..., 0, 2, 2, 2] = voigt[..., 4, 2]
        C[..., 2, 0, 2, 2] = voigt[..., 4, 2]

        C[..., 2, 2, 0, 1] = voigt[..., 2, 5]
        C[..., 2, 2, 1, 0] = voigt[..., 2, 5]
        C[..., 0, 1, 2, 2] = voigt[..., 5, 2]
        C[..., 1, 0, 2, 2] = voigt[..., 5, 2]

        C[..., 1, 2, 0, 2] = voigt[..., 3, 4]
        C[..., 1, 2, 2, 0] = voigt[..., 3, 4]
        C[..., 2, 1, 0, 2] = voigt[..., 3, 4]
        C[..., 2, 1, 2, 0] = voigt[..., 3, 4]
        C[..., 0, 2, 1, 2] = voigt[..., 4, 3]
        C[..., 0, 2, 2, 1] = voigt[..., 4, 3]
        C[..., 2, 0, 1, 2] = voigt[..., 4, 3]
        C[..., 2, 0, 2, 1] = voigt[..., 4, 3]

        C[..., 1, 2, 0, 1] = voigt[..., 3, 5]
        C[..., 1, 2, 1, 0] = voigt[..., 3, 5]
        C[..., 2, 1, 0, 1] = voigt[..., 3, 5]
        C[..., 2, 1, 1, 0] = voigt[..., 3, 5]
        C[..., 0, 1, 1, 2] = voigt[..., 5, 3]
        C[..., 0, 1, 2, 1] = voigt[..., 5, 3]
        C[..., 1, 0, 1, 2] = voigt[..., 5, 3]
        C[..., 1, 0, 2, 1] = voigt[..., 5, 3]

        C[..., 0, 2, 0, 1] = voigt[..., 4, 5]
        C[..., 0, 2, 1, 0] = voigt[..., 4, 5]
        C[..., 2, 0, 0, 1] = voigt[..., 4, 5]
        C[..., 2, 0, 1, 0] = voigt[..., 4, 5]
        C[..., 0, 1, 0, 2] = voigt[..., 5, 4]
        C[..., 0, 1, 2, 0] = voigt[..., 5, 4]
        C[..., 1, 0, 0, 2] = voigt[..., 5, 4]
        C[..., 1, 0, 2, 0] = voigt[..., 5, 4]

        return C
