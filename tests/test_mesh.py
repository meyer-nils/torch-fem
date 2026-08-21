from typing import Any

import pytest
import torch

from torchfem.elements import Hexa1
from torchfem.mesh import (
    cube_hexa,
    cube_tetra,
    mesh_to_lattice,
    rect_quad,
    rect_tri,
)


class TestRectQuad:
    def test_basic_shape(self):
        nodes, elements = rect_quad(3, 3)
        assert nodes.shape == (9, 2)
        assert elements.shape == (4, 4)

    def test_custom_dimensions(self):
        nodes, elements = rect_quad(4, 3, Lx=2.0, Ly=3.0)
        assert nodes.shape == (12, 2)
        assert elements.shape == (6, 4)
        assert torch.allclose(nodes[:, 0].max(), torch.tensor(2.0))
        assert torch.allclose(nodes[:, 1].max(), torch.tensor(3.0))

    def test_single_element(self):
        nodes, elements = rect_quad(2, 2)
        assert elements.shape == (1, 4)
        assert nodes.shape == (4, 2)

    def test_node_bounds(self):
        Lx, Ly = 5.0, 3.0
        nodes, _ = rect_quad(10, 8, Lx=Lx, Ly=Ly)
        assert torch.allclose(nodes[:, 0].min(), torch.tensor(0.0))
        assert torch.allclose(nodes[:, 1].min(), torch.tensor(0.0))
        assert torch.allclose(nodes[:, 0].max(), torch.tensor(Lx))
        assert torch.allclose(nodes[:, 1].max(), torch.tensor(Ly))

    def test_connectivity_valid(self):
        nodes, elements = rect_quad(5, 5)
        assert elements.min() >= 0
        assert elements.max() < len(nodes)


class TestRectTri:
    @pytest.mark.parametrize("variant", ["up", "down", "zigzag", "center"])
    def test_variant_shapes(self, variant):
        nodes, elements = rect_tri(3, 3, variant=variant)
        assert elements.shape[1] == 3  # triangles have 3 nodes
        assert nodes.ndim == 2
        assert nodes.shape[1] == 2

    def test_up_element_count(self):
        nodes, elements = rect_tri(3, 3, variant="up")
        # 2x2 quads = 4 quads, each split into 2 triangles = 8
        assert elements.shape[0] == 8

    def test_down_element_count(self):
        nodes, elements = rect_tri(3, 3, variant="down")
        assert elements.shape[0] == 8

    def test_zigzag_element_count(self):
        nodes, elements = rect_tri(3, 3, variant="zigzag")
        assert elements.shape[0] == 8

    def test_center_element_count(self):
        nodes, elements = rect_tri(3, 3, variant="center")
        # 4 quads, each split into 4 triangles = 16
        assert elements.shape[0] == 16

    def test_center_adds_nodes(self):
        nodes_quad, _ = rect_quad(3, 3)
        nodes_tri, _ = rect_tri(3, 3, variant="center")
        # center variant adds one node per quad
        assert len(nodes_tri) == len(nodes_quad) + 4

    def test_invalid_variant(self):
        variant: Any = "invalid"
        with pytest.raises(ValueError, match="Unknown variant"):
            rect_tri(3, 3, variant=variant)

    def test_connectivity_valid(self):
        nodes, elements = rect_tri(5, 5, variant="zigzag")
        assert elements.min() >= 0
        assert elements.max() < len(nodes)


class TestCubeHexa:
    def test_basic_shape(self):
        nodes, elements = cube_hexa(3, 3, 3)
        assert nodes.shape == (27, 3)
        assert elements.shape == (8, 8)

    def test_custom_dimensions(self):
        Lx, Ly, Lz = 2.0, 3.0, 4.0
        nodes, elements = cube_hexa(4, 3, 2, Lx=Lx, Ly=Ly, Lz=Lz)
        assert torch.allclose(nodes[:, 0].max(), torch.tensor(Lx))
        assert torch.allclose(nodes[:, 1].max(), torch.tensor(Ly))
        assert torch.allclose(nodes[:, 2].max(), torch.tensor(Lz))

    def test_single_element(self):
        nodes, elements = cube_hexa(2, 2, 2)
        assert elements.shape == (1, 8)

    def test_element_count(self):
        nodes, elements = cube_hexa(4, 3, 5)
        assert elements.shape == (3 * 2 * 4, 8)

    def test_connectivity_valid(self):
        nodes, elements = cube_hexa(5, 5, 5)
        assert elements.min() >= 0
        assert elements.max() < len(nodes)


class TestCubeTetra:
    def test_basic_shape(self):
        nodes, elements = cube_tetra(3, 3, 3)
        assert nodes.shape == (27, 3)
        assert elements.shape[1] == 4  # tetrahedral

    def test_five_tets_per_hex(self):
        nodes, elements = cube_tetra(2, 2, 2)
        # 1 hex → 5 tets
        assert elements.shape == (5, 4)

    def test_element_count(self):
        nodes, elements = cube_tetra(3, 3, 3)
        # 8 hexes → 5*8 = 40 tets
        assert elements.shape[0] == 40

    def test_custom_dimensions(self):
        Lx, Ly, Lz = 5.0, 2.0, 3.0
        nodes, _ = cube_tetra(3, 3, 3, Lx=Lx, Ly=Ly, Lz=Lz)
        assert torch.allclose(nodes[:, 0].max(), torch.tensor(Lx))
        assert torch.allclose(nodes[:, 1].max(), torch.tensor(Ly))
        assert torch.allclose(nodes[:, 2].max(), torch.tensor(Lz))

    def test_connectivity_valid(self):
        nodes, elements = cube_tetra(4, 4, 4)
        assert elements.min() >= 0
        assert elements.max() < len(nodes)


class TestMeshToLattice:
    @pytest.mark.parametrize(
        "mesh",
        [rect_tri(3, 3), rect_quad(3, 3), cube_tetra(3, 3, 3), cube_hexa(3, 3, 3)],
    )
    def test_bars_are_unique_and_valid(self, mesh):
        nodes, bars = mesh_to_lattice(*mesh)
        assert bars.shape[1] == 2
        assert torch.equal(nodes, mesh[0])
        assert (bars[:, 0] < bars[:, 1]).all()
        assert len(bars.unique(dim=0)) == len(bars)
        assert bars.max() < len(nodes)

    def test_simple_hexa_has_only_axis_aligned_bars(self):
        nodes, bars = mesh_to_lattice(*cube_hexa(3, 3, 3))
        # A 3x3x3 grid has 3 * 2 * 3 * 3 = 54 axis-aligned edges
        assert bars.shape == (54, 2)
        assert torch.allclose(
            torch.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], dim=1),
            torch.tensor(0.5),
        )

    @pytest.mark.parametrize("variant", ["up", "down"])
    def test_neighbors_agree_on_shared_faces(self, variant):
        """Each face gets exactly one diagonal, else neighbors braced it twice."""
        nodes, elements = cube_hexa(3, 3, 3)
        faces = elements[:, Hexa1.facets].reshape(-1, 4)
        n_faces = len(faces.sort(dim=1).values.unique(dim=0))
        _, bars = mesh_to_lattice(nodes, elements, variant)
        assert len(bars) == 54 + n_faces

    def test_cross_is_the_union_of_up_and_down(self):
        mesh = cube_hexa(3, 3, 3)
        sets = {
            v: {tuple(bar) for bar in mesh_to_lattice(*mesh, v)[1].tolist()}
            for v in ["simple", "up", "down", "cross"]
        }
        assert sets["up"] | sets["down"] == sets["cross"]
        assert sets["up"] & sets["down"] == sets["simple"]

    def test_up_matches_the_rect_tri_diagonal(self):
        nodes, bars = mesh_to_lattice(*rect_quad(3, 3), "up")
        _, tris = rect_tri(3, 3, variant="up")
        diagonals = {tuple(sorted([t[0], t[2]])) for t in tris.tolist()}
        assert diagonals <= {tuple(bar) for bar in bars.tolist()}

    @pytest.mark.parametrize("mesh", [rect_tri(3, 3), cube_tetra(3, 3, 3)])
    def test_simplices_reject_bracing(self, mesh):
        with pytest.raises(ValueError, match="no quadrilaterals"):
            mesh_to_lattice(*mesh, "cross")
