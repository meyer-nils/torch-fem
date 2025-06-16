from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
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
    else:
        raise ValueError("Invalid shape for stiffness tensor.")


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
    else:
        raise ValueError("Invalid shape for Voigt notation.")


def plot_contours(
    x: Tensor,
    f: Tensor,
    opti: Tensor | List = [],
    figsize: Tuple[float, float] = (8, 6),
    levels: int = 25,
    title: str = "",
    box: List[Tensor] | None = None,
    paths: dict[str, list] = {},
    colorbar: bool = False,
):
    """Function to plot contours of a function f(x) in 2D.
    Only for educational purposes.

    Args:
        x: 2D tensor of shape (N, 2) representing the coordinates.
        f: 2D tensor of shape (N,) representing the function values.
        opti: 2D tensor of shape (2,) representing the optimal point.
        figsize: Tuple representing the figure size.
        levels: Number of contour levels.
        title: Title of the plot.
        box: Tuple representing the box coordinates for the contour plot.
        paths: Dictionary of paths to plot.
        colorbar: Boolean indicating whether to show the colorbar."""
    with torch.no_grad():
        plt.figure(figsize=figsize)
        plt.contour(x[..., 0], x[..., 1], f, levels=levels, colors="k", linewidths=0.5)
        if box is not None:
            cond = (
                (x[..., 0] > box[0][0])
                & (x[..., 0] < box[1][0])
                & (x[..., 1] > box[0][1])
                & (x[..., 1] < box[1][1])
            )
            rect = patches.Rectangle(
                (box[0][0].item(), box[0][1].item()),
                box[1][0].item() - box[0][0].item(),
                box[1][1].item() - box[0][1].item(),
                edgecolor="k",
                facecolor="none",
                zorder=2,
            )
            plt.gca().add_patch(rect)
            plt.contourf(
                x[..., 0],
                x[..., 1],
                f,
                levels=levels,
                cmap="plasma",
                alpha=0.5,
                vmin=f.min(),
                vmax=f.max(),
            )
            plt.contourf(
                torch.where(cond, x[..., 0], torch.nan),
                torch.where(cond, x[..., 1], torch.nan),
                torch.where(cond, f, torch.nan),
                levels=levels,
                cmap="plasma",
                vmin=f.min(),
                vmax=f.max(),
            )
        else:
            plt.contourf(
                x[..., 0],
                x[..., 1],
                f,
                levels=levels,
                cmap="plasma",
                vmin=f.min(),
                vmax=f.max(),
            )
        if colorbar:
            plt.colorbar()
        for label, path in paths.items():
            xp = [p[0] for p in path]
            yp = [p[1] for p in path]
            plt.plot(xp, yp, "o-", linewidth=3, label=label)
            plt.legend()
        if opti:
            plt.plot(opti[0], opti[1], "ow")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.axis("equal")
        plt.title(title)
        plt.tight_layout()
