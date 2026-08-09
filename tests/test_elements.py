import tempfile
from pathlib import Path

import pytest
import torch
from matplotlib import pyplot as plt

from torchfem.elements import (
    ELEMENT_REGISTRY,
    Bar1,
    Bar2,
    Quad1,
    Quad2,
    Tria1,
    Tria2,
    linear_to_quadratic,
)
from torchfem.mesh import cube_hexa, cube_tetra, rect_quad, rect_tri


@pytest.mark.parametrize(
    "elem",
    ELEMENT_REGISTRY,
)
def test_jacobian(elem):
    volume = torch.tensor([0.0])
    for w, q in zip(elem.iweights, elem.ipoints):
        J = elem.B(q) @ elem.iso_coords
        detJ = torch.linalg.det(J)
        volume += w * detJ
    assert torch.allclose(volume, torch.tensor(elem.iso_volume), atol=1e-5)


@pytest.mark.parametrize(
    "elem",
    ELEMENT_REGISTRY,
)
def test_gradient(elem):
    for q in elem.ipoints:
        q.requires_grad = True
        for i in range(elem.nodes):
            grad = torch.autograd.grad(elem.N(q)[i], q)[0]
            assert torch.allclose(grad, elem.B(q)[:, i], atol=1e-5)


@pytest.mark.parametrize(
    "elem",
    ELEMENT_REGISTRY,
)
def test_completeness(elem):
    N = elem.N(elem.iso_coords)
    assert torch.allclose(
        N - torch.eye(elem.nodes), torch.zeros(elem.nodes, elem.nodes), atol=1e-5
    )


@pytest.mark.parametrize(
    "elem",
    ELEMENT_REGISTRY,
)
def test_quadrature_weights(elem):
    assert torch.allclose(
        elem.iweights.sum() - torch.tensor(elem.iso_volume), torch.zeros(1), atol=1e-5
    )


@pytest.mark.parametrize("elem", [Bar1, Bar2, Tria1, Tria2, Quad1, Quad2])
def test_plot(elem):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        elem.plot(path=path)
        for theme in ["light", "dark"]:
            result = path / f"{elem.__name__}_{theme}.png"
            assert result.exists()
    plt.close("all")


def _bar_mesh():
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    elements = torch.tensor([[0, 1], [1, 2]])
    return nodes, elements


# One case per supported topology, with the expected quadratic node count.
LINEAR_MESHES = [
    pytest.param(_bar_mesh(), 3, id="bar"),
    pytest.param(rect_tri(3, 3), 6, id="tria"),
    pytest.param(rect_quad(3, 3), 8, id="quad"),
    pytest.param(cube_tetra(2, 2, 2), 10, id="tetra"),
    pytest.param(cube_hexa(2, 2, 2), 20, id="hexa"),
]


class TestLinearToQuadratic:
    @pytest.mark.parametrize("mesh, n_quad_nodes", LINEAR_MESHES)
    def test_extends_connectivity_and_keeps_corner_nodes(self, mesh, n_quad_nodes):
        nodes, elements = mesh
        new_nodes, new_elements = linear_to_quadratic(nodes, elements)
        n_lin = elements.shape[1]
        assert new_elements.shape == (elements.shape[0], n_quad_nodes)
        assert torch.equal(new_elements[:, :n_lin], elements)
        assert torch.allclose(new_nodes[: nodes.shape[0]], nodes)

    @pytest.mark.parametrize("mesh, n_quad_nodes", LINEAR_MESHES)
    def test_adds_one_node_per_unique_edge(self, mesh, n_quad_nodes):
        """Elements sharing an edge must reference the same midside node."""
        nodes, elements = mesh
        new_nodes, new_elements = linear_to_quadratic(nodes, elements)
        n_lin = elements.shape[1]
        midside = new_elements[:, n_lin:]
        assert new_nodes.shape[0] - nodes.shape[0] == len(midside.unique())
        assert new_nodes.unique(dim=0).shape[0] == new_nodes.shape[0]
        # Within an element, every edge gets its own midside node.
        for elem in midside:
            assert len(elem.unique()) == n_quad_nodes - n_lin

    @pytest.mark.parametrize("mesh, n_quad_nodes", LINEAR_MESHES)
    def test_new_nodes_sit_at_edge_midpoints(self, mesh, n_quad_nodes):
        nodes, elements = mesh
        new_nodes, new_elements = linear_to_quadratic(nodes, elements)
        n_lin = elements.shape[1]
        i, j = torch.triu_indices(n_lin, n_lin, offset=1)
        for elem in new_elements:
            corners = new_nodes[elem[:n_lin]]
            midpoints = 0.5 * (corners[i] + corners[j])
            for node in new_nodes[elem[n_lin:]]:
                assert torch.isclose(midpoints, node, atol=1e-10).all(dim=-1).any()

    def test_rejects_unsupported_topology(self):
        nodes, elements = rect_quad(3, 3)
        new_nodes, new_elements = linear_to_quadratic(nodes, elements)
        with pytest.raises(Exception, match="not supported"):
            linear_to_quadratic(new_nodes, new_elements)
