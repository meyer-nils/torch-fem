from importlib import resources


def get_data(file: str) -> str:
    data_dir = resources.files("torchfem").joinpath("data")
    return str(data_dir.joinpath(file))
