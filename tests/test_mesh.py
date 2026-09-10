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

# One coarse and one refined grid per generator, with the node count and the
# element table they must produce.
GRIDS = [
    pytest.param(rect_quad, (2, 2), 4, (1, 4), id="quad-single"),
    pytest.param(rect_quad, (4, 3), 12, (6, 4), id="quad"),
    pytest.param(cube_hexa, (2, 2, 2), 8, (1, 8), id="hexa-single"),
    pytest.param(cube_hexa, (4, 3, 5), 60, (24, 8), id="hexa"),
    # Five tetrahedra per hexahedron
    pytest.param(cube_tetra, (2, 2, 2), 8, (5, 4), id="tetra-single"),
    pytest.param(cube_tetra, (3, 3, 3), 27, (40, 4), id="tetra"),
]

BOXES = [
    pytest.param(rect_quad, (10, 8), (5.0, 3.0), id="quad"),
    pytest.param(cube_hexa, (4, 3, 2), (2.0, 3.0, 4.0), id="hexa"),
    pytest.param(cube_tetra, (3, 3, 3), (5.0, 2.0, 3.0), id="tetra"),
]

MESHES = [
    pytest.param(rect_quad(5, 5), id="quad"),
    pytest.param(rect_tri(5, 5), id="tria"),
    pytest.param(cube_hexa(5, 5, 5), id="hexa"),
    pytest.param(cube_tetra(4, 4, 4), id="tetra"),
]

# Two triangles per quad, except for "center", which adds a node and makes four.
TRI_VARIANTS = [("up", 8), ("down", 8), ("zigzag", 8), ("center", 16)]


@pytest.mark.parametrize("gen, grid, n_nodes, element_shape", GRIDS)
def test_grid_has_the_expected_nodes_and_elements(gen, grid, n_nodes, element_shape):
    nodes, elements = gen(*grid)
    assert nodes.shape == (n_nodes, len(grid))
    assert elements.shape == element_shape


@pytest.mark.parametrize("gen, grid, lengths", BOXES)
def test_nodes_span_the_requested_box(gen, grid, lengths):
    nodes, _ = gen(*grid, *lengths)
    for axis, length in enumerate(lengths):
        assert torch.allclose(nodes[:, axis].min(), torch.tensor(0.0))
        assert torch.allclose(nodes[:, axis].max(), torch.tensor(length))


@pytest.mark.parametrize("mesh", MESHES)
def test_connectivity_stays_within_the_nodes(mesh):
    nodes, elements = mesh
    assert elements.min() >= 0
    assert elements.max() < len(nodes)


class TestRectTri:
    @pytest.mark.parametrize("variant, n_elem", TRI_VARIANTS)
    def test_variant_splits_every_quad(self, variant, n_elem):
        nodes, elements = rect_tri(3, 3, variant=variant)
        assert nodes.shape[1] == 2
        assert elements.shape == (n_elem, 3)

    def test_center_adds_one_node_per_quad(self):
        nodes_quad, _ = rect_quad(3, 3)
        nodes_tri, _ = rect_tri(3, 3, variant="center")
        assert len(nodes_tri) == len(nodes_quad) + 4

    def test_invalid_variant(self):
        variant: Any = "invalid"
        with pytest.raises(ValueError, match="Unknown variant"):
            rect_tri(3, 3, variant=variant)


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
        nodes, elements = mesh
        sets = {
            v: {tuple(bar) for bar in mesh_to_lattice(nodes, elements, v)[1].tolist()}
            for v in ("simple", "up", "down", "cross")
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
            mesh_to_lattice(mesh[0], mesh[1], "cross")
