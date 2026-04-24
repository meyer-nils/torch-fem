from typing import Any

import pytest
import torch

from torchfem.mesh import cube_hexa, cube_tetra, rect_quad, rect_tri


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
