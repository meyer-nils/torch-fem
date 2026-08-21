import base64
import os
import tempfile
from typing import cast

import pyvista
import torch
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.transforms import ScaledTranslation
from torch import Tensor

# vtk.js builds each actor with its own class defaults and applies the serialized
# properties without redrawing, so the first frame shows an unconfigured orientation
# triad until something triggers a redraw. Quotes would break the iframe srcdoc.
RERENDER = (
    "<script>let i=0,t=setInterval(()=>{"
    "window.global?.renderWindow?.render();"
    "if(++i>10)clearInterval(t)},200)</script>"
)


def show_html(plotter):
    """Display a plotter in a notebook, redrawing it while the scene loads."""
    viewer = plotter.show(jupyter_backend="html", return_viewer=True)
    if viewer is None:
        # Outside a notebook, show() already opened a window.
        return

    # IPython ships with the notebook extra, so it is imported where it is used
    from IPython.display import display

    viewer.value = viewer.value.replace("</body>", RERENDER + "</body>")
    display(viewer)


def embed_animation_gif(ani, fps=20):
    """
    Convert a matplotlib FuncAnimation into an embedded GIF for notebooks/GitHub.

    Works cross-platform (Windows, macOS, Linux).

    Parameters
    ----------
    ani : matplotlib.animation.FuncAnimation
        The animation object.
    fps : int, optional
        Frames per second for the GIF output (default: 20).

    Returns
    -------
    IPython.display.HTML
        An HTML object embedding the GIF inline.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()  # important: close before writing on Windows

    try:
        ani.save(tmp.name, writer="pillow", fps=fps)
        with open(tmp.name, "rb") as f:
            gif_base64 = base64.b64encode(f.read()).decode("utf-8")
    finally:
        os.remove(tmp.name)

    from IPython.display import HTML

    return HTML(f'<img src="data:image/gif;base64,{gif_base64}">')


def embed_pyvista_animation(
    plotter, update_plotter, frames, framerate=24, plotter_args=()
):
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()  # important: close before writing on Windows

    plotter.open_gif(tmp.name)
    for i in frames:
        update_plotter(i, *plotter_args)
        plotter.render()
        plotter.write_frame()
    plotter.close()

    try:
        with open(tmp.name, "rb") as f:
            gif_base64 = base64.b64encode(f.read()).decode("utf-8")
    finally:
        os.remove(tmp.name)

    from IPython.display import HTML

    return HTML(f'<img src="data:image/gif;base64,{gif_base64}">')


# Boundary condition glyphs. Each takes geometry that is already scaled, so a
# model sets its own sizes and assembles a picture from these blocks.

ARROW = pyvista.Arrow()

# A moment or a rotation carries a doubled head, the usual convention
MOMENT = ARROW + pyvista.Cone(
    center=(0.625, 0.0, 0.0), direction=(1.0, 0.0, 0.0), height=0.25, radius=0.1
)


def glyphs(
    pl: pyvista.Plotter,
    points: Tensor,
    geom,
    directions: Tensor | None = None,
    color: str | None = "gray",
):
    """Put a copy of `geom` on every point, along and scaled by directions.

    A color of None leaves the geometry in the default color of the plotter.
    """
    if points.numel() == 0:
        return
    data = pyvista.PolyData(points.cpu().numpy())
    if directions is not None:
        length = torch.linalg.norm(directions, dim=1, keepdim=True)
        data["dir"] = (directions / length).cpu().numpy()
        data["len"] = length.ravel().cpu().numpy()
    orient = "dir" if directions is not None else False
    glyph = data.glyph(geom=geom, orient=orient, scale=orient and "len")
    pl.add_mesh(cast(pyvista.DataSet, glyph), color=color)


def arrows(
    pl: pyvista.Plotter,
    points: Tensor,
    vectors: Tensor,
    span: float | None = None,
    doubled: bool = False,
):
    """Draw the nonzero vectors as arrows, the largest one spanning `span`.

    Without a span they are drawn to scale, as a prescribed displacement is.
    A doubled head marks a moment or a rotation.
    """
    magnitude = torch.linalg.norm(vectors, dim=1)
    if magnitude.numel() == 0 or magnitude.max() == 0.0:
        return
    if span is not None:
        vectors = span / magnitude.max() * vectors
    drawn = magnitude > 0.0
    glyphs(pl, points[drawn], MOMENT if doubled else ARROW, vectors[drawn])


def cones(
    pl: pyvista.Plotter,
    points: Tensor,
    fixed: Tensor,
    height: float,
    doubled: bool = False,
):
    """Draw a cone of that height along the axis of every constrained DOF."""
    geom = pyvista.Cone(
        center=(-0.5 * height, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        height=height,
        radius=0.5 * height,
        resolution=12,
    )
    if doubled:
        geom = geom + geom.translate((-height, 0.0, 0.0), inplace=False)
    node, dof = torch.nonzero(fixed).T
    glyphs(pl, points[node], geom, torch.eye(3, dtype=points.dtype)[dof])


def dots(pl: pyvista.Plotter, points: Tensor, radius: float):
    """Draw a sphere on every point, marking where a node is pulled to."""
    geom = pyvista.Sphere(radius=radius, theta_resolution=10, phi_resolution=10)
    glyphs(pl, points, geom)


# Boundary conditions in a matplotlib plot. Every glyph is gray and outlined in
# the foreground of the style, sized in points so it looks the same on any mesh.

MARKERSIZE = 6.0

# A node label sits diagonally off its node, clear of the marker drawn there
LABEL_OFFSET = {"textcoords": "offset points", "xytext": (MARKERSIZE, MARKERSIZE)}


def _glyph(size: float = MARKERSIZE, zorder: float = 10, **kwargs) -> dict:
    """Keywords drawing a marker, taking its outline from the current style."""
    return {
        "markerfacecolor": "gray",
        "markeredgecolor": rcParams["text.color"],
        "markeredgewidth": 1.0,
        "markersize": size,
        "zorder": zorder,
        "clip_on": False,
        **kwargs,
    }


def dots2d(ax: Axes, points: Tensor, zorder: float = 12):
    """Draw a dot where a glyph attaches to a node, above the arrow it caps.

    A constraint passes the zorder of its own markers instead, so that an arrow
    crossing a constrained edge stays visible.
    """
    if points.numel() == 0:
        return
    ax.plot(*points.T, "o", **_glyph(zorder=zorder))


def arrows2d(
    ax: Axes, points: Tensor, vectors: Tensor, width: float, span: float | None = None
) -> Tensor:
    """Draw the nonzero vectors as arrows and return their tips.

    Without a span they are drawn to scale, as a prescribed displacement is.
    """
    magnitude = torch.linalg.norm(vectors, dim=1)
    if magnitude.numel() == 0 or magnitude.max() == 0.0:
        return points[:0]
    if span is not None:
        vectors = span / magnitude.max() * vectors
    drawn = magnitude > 0.0
    style = {
        "width": width,
        "length_includes_head": span is None,
        "facecolor": "gray",
        "edgecolor": rcParams["text.color"],
        "linewidth": 1.0,
        # Above the constraint markers a load may share a node with
        "zorder": 11,
    }
    for start, vector in zip(points[drawn], vectors[drawn]):
        ax.arrow(*start.tolist(), *vector.tolist(), **style)
    dots2d(ax, points[drawn])
    return (points + vectors)[drawn]


def markers2d(ax: Axes, points: Tensor, fixed: Tensor):
    """Mark every constrained DOF with a marker sitting just outside its node."""
    # A temperature has no direction, so its constraint is a square on the node
    if fixed.shape[1] == 1:
        markers = [("s", 0.0, 0.0)]
    else:
        markers = [(">", -MARKERSIZE, 0.0), ("^", 0.0, -MARKERSIZE)]
        # A dot at the node, so that the offset markers point at something
        dots2d(ax, points[fixed.any(dim=1)], zorder=10)

    for dof, (marker, dx, dy) in enumerate(markers):
        drawn = points[fixed[:, dof]]
        if drawn.numel() == 0:
            continue
        # Shifted in points, so the marker clears the node at any mesh size
        shift = ScaledTranslation(dx / 72, dy / 72, ax.figure.dpi_scale_trans)
        ax.plot(*drawn.T, marker, **_glyph(transform=ax.transData + shift))


def signs2d(ax: Axes, points: Tensor, values: Tensor):
    """Mark every nonzero value with its sign, as a heat source or a sink."""
    for marker, drawn in [("+", values > 0.0), ("_", values < 0.0)]:
        shown = points[drawn]
        if shown.numel() == 0:
            continue
        # A line marker carries no fill, so it is drawn larger and heavier
        style = _glyph(1.5 * MARKERSIZE, markeredgecolor="gray", markeredgewidth=2.0)
        ax.plot(*shown.T, marker, **style)
