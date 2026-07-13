import tempfile
from pathlib import Path

import torch

import pytest

from torchfem import Planar, Solid
from torchfem.io import (
    export_mesh,
    import_mesh,
    import_planar,
    import_shell,
    import_solid,
)
from torchfem.materials import IsotropicElasticity3D, IsotropicElasticityPlaneStress
from torchfem.mesh import cube_hexa, rect_quad


class TestExportMesh:
    def test_export_vtu(self):
        nodes, elements = cube_hexa(3, 3, 3)
        mat = IsotropicElasticity3D(1000.0, 0.3)
        model = Solid(nodes, elements, mat)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtu"
            export_mesh(model, str(path))
            assert path.exists()
            assert path.stat().st_size > 0

    def test_export_with_nodal_data(self):
        nodes, elements = cube_hexa(3, 3, 3)
        mat = IsotropicElasticity3D(1000.0, 0.3)
        model = Solid(nodes, elements, mat)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtu"
            u = torch.randn(len(nodes), 3)
            export_mesh(model, str(path), nodal_data={"displacement": u})
            assert path.exists()

    def test_export_uncompressed(self):
        nodes, elements = cube_hexa(2, 2, 2)
        mat = IsotropicElasticity3D(1000.0, 0.3)
        model = Solid(nodes, elements, mat)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtu"
            export_mesh(model, str(path), compress=False)
            assert path.exists()


class TestImportMesh:
    def test_import_3d_mesh(self):
        """Export then re-import a 3D mesh."""
        nodes, elements = cube_hexa(3, 3, 3)
        mat = IsotropicElasticity3D(1000.0, 0.3)
        model = Solid(nodes, elements, mat)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtu"
            export_mesh(model, str(path))
            reimported = import_mesh(path, mat)
            assert isinstance(reimported, Solid)
            assert reimported.n_elem == model.n_elem

    def test_import_2d_mesh(self):
        """Export then re-import a 2D mesh."""
        nodes_2d, elements = rect_quad(3, 3)
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        model = Planar(nodes_2d, elements, mat)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtu"
            export_mesh(model, str(path))
            reimported = import_mesh(path, mat)
            assert isinstance(reimported, Planar)
            assert reimported.n_elem == model.n_elem


class TestTypedImport:
    def test_import_planar_narrows(self):
        """The planar wrapper returns a `Planar` for a 2D mesh."""
        nodes_2d, elements = rect_quad(3, 3)
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        model = Planar(nodes_2d, elements, mat)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtu"
            export_mesh(model, str(path))
            assert isinstance(import_planar(path, mat), Planar)

    def test_import_solid_narrows(self):
        """The solid wrapper returns a `Solid` for a 3D mesh."""
        nodes, elements = cube_hexa(3, 3, 3)
        mat = IsotropicElasticity3D(1000.0, 0.3)
        model = Solid(nodes, elements, mat)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtu"
            export_mesh(model, str(path))
            assert isinstance(import_solid(path, mat), Solid)

    def test_type_mismatch_raises(self):
        """A wrapper raises `TypeError` when the mesh is of another type."""
        nodes, elements = cube_hexa(3, 3, 3)
        mat = IsotropicElasticity3D(1000.0, 0.3)
        model = Solid(nodes, elements, mat)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtu"
            export_mesh(model, str(path))
            with pytest.raises(TypeError):
                import_shell(path, mat)
            with pytest.raises(TypeError):
                import_planar(path, mat)
