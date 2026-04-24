import tempfile
from pathlib import Path

import torch

from torchfem import Planar, Solid
from torchfem.io import export_mesh, import_mesh
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
