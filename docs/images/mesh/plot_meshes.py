"""Generate mesh figures.

Usage: python docs/images/mesh/plot_meshes.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from torchfem.elements import THEMES, linear_to_quadratic
from torchfem.mesh import cube_hexa, cube_tetra, mesh_to_lattice, rect_quad, rect_tri

# Write next to this script, which does not move with an installed torchfem.
IMAGES_DIR = Path(__file__).parent


def draw(ax, nodes, bars, extra=None):
    """Draw a lattice as line segments, marking its nodes and any extra ones."""
    for bar in bars:
        ax.plot(*nodes[bar].T, color="C0", linewidth=1.5, zorder=1)
    ax.scatter(*nodes.T, s=12, color="C0", zorder=2)
    if extra is not None:
        ax.scatter(*extra.T, s=24, color="C1", zorder=3)
    ax.set_axis_off()
    ax.set_aspect("equal")


def generated(func, *args, **kwargs):
    """A panel of the lattice of a generated mesh, titled by the call."""
    shown = [*map(repr, args), *(f"{k}={v!r}" for k, v in kwargs.items())]
    title = f"{func.__name__}({', '.join(shown)})"
    return title, mesh_to_lattice(*func(*args, **kwargs))


def braced(mesh, variant):
    """A panel of a braced lattice, titled by the `mesh_to_lattice` call."""
    return f"mesh_to_lattice(..., {variant!r})", mesh_to_lattice(*mesh, variant)


def figure(name, cases, projection=None):
    """Save a row of panels, one figure per color scheme.

    Each case pairs a title with the arguments of `draw` for that panel.
    """
    for theme, style in THEMES:
        with plt.style.context(style):
            fig, _ = plt.subplots(
                1,
                len(cases),
                figsize=(3.2 * len(cases), 3.4),
                subplot_kw={"projection": projection},
            )
            for ax, (title, panel) in zip(fig.axes, cases):
                draw(ax, *panel)
                ax.set_title(title, fontsize=9)
            # A common range draws every panel at the same scale, so that panels
            # differing only in their lengths stay comparable.
            for axis in "xyz" if projection == "3d" else "xy":
                lo, hi = zip(*(getattr(ax, f"get_{axis}lim")() for ax in fig.axes))
                for ax in fig.axes:
                    getattr(ax, f"set_{axis}lim")(min(lo), max(hi))
            fig.savefig(
                IMAGES_DIR / f"{name}_{theme}.png",
                dpi=200,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close(fig)
    print(f"Saved {name}")


def main():
    figure(
        "rect_quad",
        [generated(rect_quad, 4, 4, *lengths) for lengths in [(), (2.0, 1.0)]],
    )
    figure(
        "rect_tri",
        [
            generated(rect_tri, 4, 4, variant=v)
            for v in ("up", "down", "zigzag", "center")
        ],
    )
    figure(
        "cube_hexa",
        [generated(cube_hexa, 3, 3, 3, *lengths) for lengths in [(), (2.0, 1.0, 0.5)]],
        projection="3d",
    )
    # Two hexahedra, so that the checkerboard alternation of the split is
    # visible without the edges piling up.
    figure(
        "cube_tetra",
        [generated(cube_tetra, 3, 2, 2, *lengths) for lengths in [(), (2.0, 1.0, 0.5)]],
        projection="3d",
    )
    figure(
        "mesh_to_lattice",
        [braced(rect_quad(4, 4), v) for v in ("simple", "up", "down", "cross")],
    )
    # Two hexahedra, so that both the bracing per face and the face they
    # share are visible.
    figure(
        "mesh_to_lattice_hexa",
        [braced(cube_hexa(3, 2, 2), v) for v in ("simple", "cross")],
        projection="3d",
    )
    # The quadratic mesh keeps the corner nodes and appends the mid-side ones.
    linear = rect_quad(3, 3)
    bars = mesh_to_lattice(*linear)
    midside = linear_to_quadratic(*linear)[0][len(linear[0]) :]
    figure(
        "linear_to_quadratic",
        [("rect_quad(3, 3)", bars), ("linear_to_quadratic(...)", (*bars, midside))],
    )


if __name__ == "__main__":
    main()
