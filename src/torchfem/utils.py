import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch import Tensor

# Row and column of each Voigt component, per spatial dimension
_VOIGT = {
    2: ((0, 0), (1, 1), (0, 1)),
    3: ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)),
}


def _indices(dim: int) -> tuple[Tensor, Tensor]:
    """Row and column index of every Voigt component in `dim` dimensions."""
    rows, cols = zip(*_VOIGT[dim])
    return torch.tensor(rows), torch.tensor(cols)


def _tensor_dim(tensor: Tensor, rank: int) -> int:
    """Spatial dimension of a square tensor of the given rank."""
    dim = tensor.shape[-1]
    if dim not in _VOIGT or tuple(tensor.shape[-rank:]) != rank * (dim,):
        raise ValueError(f"Invalid shape {tuple(tensor.shape)} for a tensor.")
    return dim


def _voigt_dim(voigt: Tensor, rank: int) -> int:
    """Spatial dimension that a Voigt array of the given rank belongs to."""
    for dim, pairs in _VOIGT.items():
        if tuple(voigt.shape[-rank:]) == rank * (len(pairs),):
            return dim
    raise ValueError(f"Invalid shape {tuple(voigt.shape)} for Voigt notation.")


def stress2voigt(sigma: Tensor) -> Tensor:
    """Convert a stress tensor to Voigt notation."""
    i, j = _indices(_tensor_dim(sigma, 2))
    return sigma[..., i, j]


def strain2voigt(epsilon: Tensor) -> Tensor:
    """Convert a strain tensor to Voigt notation, doubling the shear components."""
    i, j = _indices(_tensor_dim(epsilon, 2))
    return torch.where(i == j, 1.0, 2.0).to(epsilon) * epsilon[..., i, j]


def stiffness2voigt(C: Tensor) -> Tensor:
    """Convert a stiffness tensor to Voigt notation."""
    i, j = _indices(_tensor_dim(C, 4))
    return C[..., i[:, None], j[:, None], i, j]


def voigt2stress(voigt: Tensor) -> Tensor:
    """Convert a stress tensor from Voigt notation."""
    dim = _voigt_dim(voigt, 1)
    i, j = _indices(dim)
    sigma = torch.zeros(*voigt.shape[:-1], dim, dim).to(voigt)
    sigma[..., i, j] = voigt
    sigma[..., j, i] = voigt
    return sigma


def voigt2strain(voigt: Tensor) -> Tensor:
    """Convert a strain tensor from Voigt notation, halving the shear components."""
    dim = _voigt_dim(voigt, 1)
    i, j = _indices(dim)
    shear = torch.where(i == j, 1.0, 0.5).to(voigt) * voigt
    epsilon = torch.zeros(*voigt.shape[:-1], dim, dim).to(voigt)
    epsilon[..., i, j] = shear
    epsilon[..., j, i] = shear
    return epsilon


def voigt2stiffness(voigt: Tensor) -> Tensor:
    """Convert a stiffness tensor from Voigt notation, restoring its symmetries."""
    dim = _voigt_dim(voigt, 2)
    i, j = _indices(dim)
    C = torch.zeros(*voigt.shape[:-2], *(4 * (dim,))).to(voigt)
    for rows, cols in ((i, j), (j, i)):
        for k, m in ((i, j), (j, i)):
            C[..., rows[:, None], cols[:, None], k, m] = voigt
    return C


def plot_contours(
    x: Tensor,
    f: Tensor,
    opti: Tensor | list | None = None,
    figsize: tuple[float, float] = (8, 6),
    levels: int = 25,
    title: str = "",
    box: list[Tensor] | None = None,
    paths: dict[str, list] | None = None,
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
    opti = [] if opti is None else opti
    paths = {} if paths is None else paths

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
