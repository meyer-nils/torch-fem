from importlib import resources
from os import PathLike


def get_example_file(file: str) -> PathLike:
    with resources.path("torchfem.data", file) as f:
        data_file_path = f
    return data_file_path
