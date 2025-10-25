import base64
import tempfile
import os
from IPython.display import HTML


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
