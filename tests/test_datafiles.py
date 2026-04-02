from pathlib import Path

import pytest

from torchfem.data import get_data

EXAMPLE_FILES = [
    "clamped_plate_uniform_S8R.vtk",
    "extruded_hex_mesh.inp",
    "fillet.vtu",
    "ge_bracket.vtu",
    "iso37.vtu",
    "plate_hole.vtk",
    "quarter_plate_hole_3d.vtu",
]


class TestGetData:
    @pytest.mark.parametrize("file_name", EXAMPLE_FILES)
    def test_returns_existing_example_path(self, file_name):
        path = Path(get_data(file_name))
        assert path.exists()
        assert path.name == file_name

    def test_missing_file_returns_nonexistent_path(self):
        path = Path(get_data("missing-example-file.vtu"))
        assert not path.exists()
