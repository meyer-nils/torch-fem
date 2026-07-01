from importlib import resources
from pathlib import Path


def get_data(file: str) -> Path:
    data_dir = resources.files("torchfem").joinpath("data")
    return Path(str(data_dir.joinpath(file)))
