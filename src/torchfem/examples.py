from importlib import resources


def get_example_file(file):
    with resources.path("torchfem.data", file) as f:
        data_file_path = f
    return data_file_path
