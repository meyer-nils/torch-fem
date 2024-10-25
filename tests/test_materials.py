import pytest

from torchfem.materials import Isotropic


@pytest.mark.parametrize("material, shape", [(Isotropic(1000.0, 0.3), (10, 3))])
def test_vectorization(material, shape):
    mat_vec = material.vectorize(*shape)
    assert mat_vec.E.shape == shape
    assert mat_vec.nu.shape == shape
    assert mat_vec.C.shape == shape + material.C.shape
