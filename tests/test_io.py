import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from meshio import Mesh

from torchfem import Laminate, Planar, PlanarHeat, Shell, Solid, SolidHeat
from torchfem.elements import Quad1
from torchfem.io import (
    export_mesh,
    import_mesh,
    import_shell,
)
from torchfem.materials import (
    IsotropicConductivity2D,
    IsotropicConductivity3D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStress,
)
from torchfem.mesh import cube_hexa, rect_quad

# Corners of a tetrahedron, so no cell built from them lies in the z=0 plane.
NON_PLANAR_POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
PLANAR_POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
)


def write_mesh(path, cells, points=NON_PLANAR_POINTS):
    """Write a mesh torch-fem cannot export itself."""
    Mesh(points, cells).write(path)


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
    def test_type_mismatch_raises(self):
        """`import_shell` raises `TypeError` when the mesh is of another type."""
        nodes, elements = cube_hexa(3, 3, 3)
        mat = IsotropicElasticity3D(1000.0, 0.3)
        model = Solid(nodes, elements, mat)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtu"
            export_mesh(model, str(path))
            with pytest.raises(TypeError):
                import_shell(path, mat)


class TestImportNonPlanarMesh:
    """Meshes torch-fem cannot export itself, written directly with meshio."""

    def test_non_planar_triangle_mesh_returns_shell(self):
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shell.vtu"
            write_mesh(path, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])
            model = import_mesh(path, mat, thickness=0.1)
            assert isinstance(model, Shell)
            assert model.n_elem == 2
            assert isinstance(import_shell(path, mat), Shell)

    def test_rejects_multiple_element_types(self):
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mixed.vtu"
            write_mesh(
                path,
                [
                    ("triangle", np.array([[0, 1, 2]])),
                    ("quad", np.array([[0, 1, 2, 3]])),
                ],
            )
            with pytest.raises(ValueError, match="single element types"):
                import_mesh(path, mat)

    def test_non_planar_quad_mesh_returns_shell(self):
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shell.vtu"
            write_mesh(path, [("quad", np.array([[0, 1, 2, 3]]))])
            model = import_mesh(path, mat, thickness=0.1)
            assert isinstance(model, Shell)
            assert model.etype is Quad1
            assert isinstance(import_shell(path, mat), Shell)

    def test_rejects_element_type_without_a_model(self):
        """A quadratic triangle is a supported element, but only in the z=0 plane."""
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        points = np.vstack([NON_PLANAR_POINTS, [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tria6_3d.vtu"
            write_mesh(path, [("triangle6", np.array([[0, 1, 3, 4, 5, 2]]))], points)
            with pytest.raises(ValueError, match="Cannot interpret element type"):
                import_mesh(path, mat)


class TestImportMeshPhysics:
    """The material decides the physics of the imported model."""

    def test_planar_mesh_with_a_conductivity_gives_planar_heat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "planar.vtu"
            write_mesh(path, [("quad", np.array([[0, 1, 2, 3]]))], PLANAR_POINTS)
            model = import_mesh(path, IsotropicConductivity2D(400.0), thickness=0.1)
            assert isinstance(model, PlanarHeat)
            assert torch.allclose(model.thickness, torch.full((1,), 0.1))

    def test_solid_mesh_with_a_conductivity_gives_solid_heat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solid.vtu"
            write_mesh(path, [("tetra", np.array([[0, 1, 2, 3]]))])
            assert isinstance(
                import_mesh(path, IsotropicConductivity3D(400.0)), SolidHeat
            )

    def test_surface_mesh_has_no_heat_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shell.vtu"
            write_mesh(path, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])
            with pytest.raises(ValueError, match="no heat model"):
                import_mesh(path, IsotropicConductivity2D(400.0))

    def test_a_laminate_section_imports_as_a_shell(self):
        """A `Laminate` stands in for a material without subclassing one."""
        layup = Laminate([IsotropicElasticityPlaneStress(1000.0, 0.3)], [1.0], [0.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shell.vtu"
            write_mesh(path, [("triangle", np.array([[0, 1, 2], [0, 1, 3]]))])
            assert isinstance(import_mesh(path, layup), Shell)

    def test_a_laminate_needs_a_surface_mesh(self):
        layup = Laminate([IsotropicElasticityPlaneStress(1000.0, 0.3)], [1.0], [0.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solid.vtu"
            write_mesh(path, [("tetra", np.array([[0, 1, 2, 3]]))])
            with pytest.raises(ValueError, match="laminate section needs a surface"):
                import_mesh(path, layup)

    def test_a_flat_surface_mesh_imports_as_a_shell(self):
        """A flat shell is requested through `import_shell`, not `import_mesh`."""
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        flat = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flat.vtu"
            write_mesh(path, [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))], flat)
            assert isinstance(import_mesh(path, mat), Planar)
            shell = import_shell(path, mat, thickness=0.5)
            assert isinstance(shell, Shell)
            assert shell.nodes.shape == (4, 3)
            assert torch.allclose(shell.thickness, torch.full((2,), 0.5))

    def test_a_solid_mesh_is_not_a_surface(self):
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "solid.vtu"
            write_mesh(path, [("tetra", np.array([[0, 1, 2, 3]]))])
            with pytest.raises(TypeError, match="not a surface mesh, but tetra"):
                import_shell(path, mat)
