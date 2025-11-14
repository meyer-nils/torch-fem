import pytest
import torch

from torchfem.elements import ELEMENT_REGISTRY


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
