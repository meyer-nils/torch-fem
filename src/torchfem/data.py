from importlib import resources
from pathlib import Path


def get_data(file: str) -> Path:
    """Resolves the path to an example mesh bundled with torch-fem.

    The meshes live in `torchfem/data` and are meant to be passed straight to
    `import_mesh(...)`, e.g. `import_mesh(get_data("plate_hole.vtk"), material)`.

    Args:
        file (str): File name of the bundled mesh, including its suffix.

    Returns:
        Path: Absolute path to the file inside the installed package. The path is
            returned without checking that the file exists.
    """
    data_dir = resources.files("torchfem").joinpath("data")
    return Path(str(data_dir.joinpath(file)))
