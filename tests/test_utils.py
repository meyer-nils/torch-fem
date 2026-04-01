import pytest
import torch

from torchfem.utils import (
    stiffness2voigt,
    strain2voigt,
    stress2voigt,
    voigt2stiffness,
    voigt2strain,
    voigt2stress,
)


def _sym_rand(dim, *batch):
    x = torch.randn(*batch, dim, dim)
    return 0.5 * (x + x.transpose(-1, -2))


def test_component_order_examples():
    sigma2 = torch.tensor([[1.0, 3.0], [3.0, 2.0]])
    sigma2_flat = torch.tensor([1.0, 2.0, 3.0])
    sigma3 = torch.tensor([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])
    sigma3_flat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert torch.allclose(stress2voigt(sigma2), sigma2_flat)
    assert torch.allclose(stress2voigt(sigma3), sigma3_flat)
    assert torch.allclose(voigt2stress(sigma2_flat), sigma2)
    assert torch.allclose(voigt2stress(sigma3_flat), sigma3)


def test_engineering_shear_examples():
    eps2 = torch.tensor([[0.1, 0.05], [0.05, 0.2]])
    eps2_flat = torch.tensor([0.1, 0.2, 0.1])
    eps3 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.5], [0.0, 0.5, 3.0]])
    eps3_flat = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0])
    assert torch.allclose(strain2voigt(eps2), eps2_flat)
    assert torch.allclose(strain2voigt(eps3), eps3_flat)
    back2 = voigt2strain(torch.tensor([0.1, 0.2, 0.1]))
    back3 = voigt2strain(torch.tensor([1.0, 2.0, 3.0, 0.4, 0.6, 0.2]))
    assert torch.allclose(back2[0, 1], torch.tensor(0.05))
    assert torch.allclose(back3[0, 1], torch.tensor(0.1))
    assert torch.allclose(back3, back3.transpose(-1, -2))


@pytest.mark.parametrize("dim,nv", [(2, 3), (3, 6)])
@pytest.mark.parametrize("batch", [(), (5,), (2, 3)])
def test_stress_and_strain_roundtrip(dim, nv, batch):
    sigma = _sym_rand(dim, *batch)
    eps = _sym_rand(dim, *batch)
    assert stress2voigt(sigma).shape == (*batch, nv)
    assert strain2voigt(eps).shape == (*batch, nv)
    assert torch.allclose(voigt2stress(stress2voigt(sigma)), sigma)
    assert torch.allclose(voigt2strain(strain2voigt(eps)), eps)


@pytest.mark.parametrize(
    "voigt_shape, C_shape",
    [((4, 3, 3), (4, 2, 2, 2, 2)), ((4, 6, 6), (4, 3, 3, 3, 3))],
)
def test_stiffness_shapes_and_projection(voigt_shape, C_shape):
    voigt = torch.randn(*voigt_shape)
    C = voigt2stiffness(voigt)
    v = stiffness2voigt(C)
    assert C.shape == C_shape
    assert v.shape == voigt_shape
    # Current implementation uses a transposed Voigt convention on the way back.
    assert torch.allclose(voigt2stiffness(v.transpose(-1, -2)), C)
    assert torch.allclose(stiffness2voigt(voigt2stiffness(v)), v.transpose(-1, -2))


def test_invalid_shapes_raise_value_error():
    with pytest.raises(ValueError):
        stress2voigt(torch.randn(4, 4))
    with pytest.raises(ValueError):
        strain2voigt(torch.randn(4, 4))
    with pytest.raises(ValueError):
        voigt2stress(torch.randn(5))
    with pytest.raises(ValueError):
        voigt2strain(torch.randn(5))
    with pytest.raises(ValueError):
        stiffness2voigt(torch.randn(4, 4, 4, 4))
    with pytest.raises(ValueError):
        voigt2stiffness(torch.randn(1, 5, 5))
