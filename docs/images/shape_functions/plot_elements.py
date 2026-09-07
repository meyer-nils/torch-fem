"""Generate shape function plots.

Usage: python docs/images/shape_functions/plot_elements.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from torchfem.elements import Bar1, Bar2, Element, Quad1, Quad2, Tria1, Tria2
from torchfem.plot_utils import THEMES

# Write next to this script, which does not move with an installed torchfem.
IMAGES_DIR = Path(__file__).parent

Size = tuple[float, float]


def plot_bar(cls: type[Element], figsize: Size, n_points: int = 100) -> Figure:
    """Shape functions of a line element over its reference coordinate."""
    # Compute shape functions at evenly spaced points in reference space
    xi = torch.linspace(-1.0, 1.0, n_points).unsqueeze(-1)
    N = cls.N(xi)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(cls.nodes):
        ax.plot(xi, N[:, i], linewidth=2.0, label=f"$N_{i}$")
    ax.set_xlabel("$\\xi$")
    ax.set_ylabel("$N_i(\\xi)$")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_tria(cls: type[Element], figsize: Size, n_points: int = 30) -> Figure:
    """Shape functions of a triangle, one panel each, in rows of three."""
    # Sample inside triangular reference domain (ξ₁ ≥ 0, ξ₂ ≥ 0, ξ₁+ξ₂ ≤ 1)
    t = np.linspace(0.0, 1.0, n_points)
    xi1, xi2 = np.meshgrid(t, t)
    mask = (xi1 + xi2) <= 1.0
    xi1f, xi2f = xi1[mask], xi2[mask]
    xi = torch.tensor(np.stack([xi1f, xi2f], axis=-1), dtype=torch.float32)
    N = cls.N(xi).detach().cpu().numpy()

    fig, axes = plt.subplots(
        cls.nodes // 3, 3, figsize=figsize, subplot_kw={"projection": "3d"}
    )
    for i, ax in enumerate(np.asarray(axes).ravel()):
        ax.plot_trisurf(xi1f, xi2f, N[:, i], color=f"C{i}", alpha=0.9)
        ax.set_xlabel("$\\xi_1$")
        ax.set_ylabel("$\\xi_2$")
        ax.set_title(f"$N_{i}$")

    fig.tight_layout()
    return fig


def plot_quad(cls: type[Element], figsize: Size, n_points: int = 30) -> Figure:
    """Shape functions of a quadrilateral, one panel each, in two rows."""
    # Sample on the square reference domain (ξ₁, ξ₂ ∈ [-1, 1])
    t = np.linspace(-1.0, 1.0, n_points)
    xi1, xi2 = np.meshgrid(t, t)
    xi = torch.tensor(
        np.stack([xi1.ravel(), xi2.ravel()], axis=-1), dtype=torch.float32
    )
    N = cls.N(xi).detach().cpu().numpy()

    fig, axes = plt.subplots(
        2, cls.nodes // 2, figsize=figsize, subplot_kw={"projection": "3d"}
    )
    for i, ax in enumerate(np.asarray(axes).ravel()):
        ax.plot_surface(
            xi1,
            xi2,
            N[:, i].reshape(n_points, n_points),
            color=f"C{i}",
            alpha=0.9,
            linewidth=0,
        )
        ax.set_xlabel("$\\xi_1$")
        ax.set_ylabel("$\\xi_2$")
        ax.set_title(f"$N_{i}$")

    fig.tight_layout()
    return fig


# Each element with the function that draws it and a figure size fitting its panels.
ELEMENTS = [
    (Bar1, plot_bar, (6.0, 4.0)),
    (Bar2, plot_bar, (6.0, 4.0)),
    (Tria1, plot_tria, (10.0, 4.0)),
    (Tria2, plot_tria, (10.0, 8.0)),
    (Quad1, plot_quad, (8.0, 8.0)),
    (Quad2, plot_quad, (14.0, 8.0)),
]


def main(path: Path = IMAGES_DIR):
    """Write <Element>_<theme>.png for every element and color scheme."""
    for cls, plot, figsize in ELEMENTS:
        for theme, style in THEMES:
            with plt.style.context(style):
                fig = plot(cls, figsize)
                fig.savefig(
                    path / f"{cls.__name__}_{theme}.png",
                    dpi=200,
                    bbox_inches="tight",
                    transparent=True,
                )
                plt.close(fig)
        print(f"Saved {cls.__name__} shape functions")


if __name__ == "__main__":
    main()
