import base64
import os
import tempfile
from typing import cast

import pyvista
import torch
from IPython.display import HTML, display
from matplotlib.axes import Axes
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
    # Drawn on top of the mesh, with no edge around the head
    style = {
        "width": width,
        "length_includes_head": span is None,
        "facecolor": "gray",
        "linewidth": 0.0,
        "zorder": 10,
    }
    for start, vector in zip(points[drawn], vectors[drawn]):
        ax.arrow(*start.tolist(), *vector.tolist(), **style)
    return (points + vectors)[drawn]


def markers2d(ax: Axes, points: Tensor, fixed: Tensor, offset: float):
    """Mark every constrained DOF with a marker clearing its node by `offset`."""
    # A temperature has no direction, so its constraint is a square
    if fixed.shape[1] == 1:
        for x, y in points[fixed[:, 0]].tolist():
            ax.plot(x, y, "s", color="gray")
        return

    for (x, y), dof in zip(points.tolist(), fixed):
        if dof[0]:
            ax.plot(x - offset, y, ">", color="gray")
        if dof[1]:
            ax.plot(x, y - offset, "^", color="gray")
