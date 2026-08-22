"""Generate laminate figures.

Usage: python docs/images/laminate/plot_laminates.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torchfem.elements import THEMES
from torchfem.laminate import Laminate
from torchfem.materials import OrthotropicElasticityPlaneStress

# Write next to this script, which does not move with an installed torchfem.
IMAGES_DIR = Path(__file__).parent

DEG = torch.pi / 180.0

ply = OrthotropicElasticityPlaneStress(
    E_1=54000.0, E_2=9400.0, nu_12=0.33, G_12=5500.0, G_13=5500.0, G_23=3000.0
)


def laminate(angles, **kwargs) -> Laminate:
    """A laminate of equally thick plies at the given angles, in degrees."""
    return Laminate(
        materials=[ply] * len(angles),
        thicknesses=[0.25] * len(angles),
        angles=[a * DEG for a in angles],
        **kwargs,
    )


def figure(name, cases):
    """Save a row of stacking sequences, one figure per color scheme."""
    for theme, style in THEMES:
        with plt.style.context(style):
            fig, axes = plt.subplots(1, len(cases), figsize=(2.6 * len(cases), 4.2))
            for ax, (title, layup) in zip(fig.axes, cases):
                layup.plot(ax=ax)
                ax.set_title(title, fontsize=9)
            for ax in axes[1:]:
                ax.set_ylabel("")
            fig.savefig(
                IMAGES_DIR / f"{name}_{theme}.png",
                dpi=200,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close(fig)
    print(f"Saved {name}")


def main():
    # The stacking order, the mirrored half-stack, and the shifted reference
    # surface, which are the three arguments that move the plies around.
    figure(
        "laminate_stacking",
        [
            ("[0, 90, 90, 0]", laminate([0, 90, 90, 0])),
            ("[0, 45] symmetric", laminate([0, 45], symmetric=True)),
            ("[0, 45] offset='top'", laminate([0, 45], offset="top")),
        ],
    )


if __name__ == "__main__":
    main()
