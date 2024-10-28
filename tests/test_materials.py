import pytest

from torchfem.materials import Isotropic


@pytest.mark.parametrize("material, n_elem", [(Isotropic(1000.0, 0.3), 10)])
def test_vectorization(material, n_elem):
    mat_vec = material.vectorize(n_elem)
    assert mat_vec.E.shape == (n_elem,)
    assert mat_vec.nu.shape == (n_elem,)
    assert mat_vec.C.shape == (n_elem,) + material.C.shape
