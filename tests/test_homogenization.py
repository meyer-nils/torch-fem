import pytest
import torch

from torchfem.homogenization import (
    IBOF_closure,
    compute_orientation_average,
    symm,
    tandon_weng_homogenization,
)
from torchfem.materials import IsotropicElasticity3D

ISOTROPIC = torch.eye(3) / 3.0
ALIGNED = torch.diag(torch.tensor([1.0, 0.0, 0.0]))


def _matrix():
    return IsotropicElasticity3D(3500.0, 0.35)


def _fiber():
    return IsotropicElasticity3D(72000.0, 0.2)


class TestSymm:
    def test_symmetrizing_twice_changes_nothing(self):
        A4 = torch.randn(2, 3, 3, 3, 3)
        assert torch.allclose(symm(symm(A4)), symm(A4), atol=1e-12)

    def test_output_has_full_index_symmetry(self):
        result = symm(torch.randn(2, 3, 3, 3, 3))
        # Averaging over all 24 permutations, so any of them leaves it alone
        assert torch.allclose(result, result.permute(0, 2, 1, 3, 4), atol=1e-10)
        assert torch.allclose(result, result.permute(0, 3, 4, 1, 2), atol=1e-10)


class TestIBOFClosure:
    @pytest.mark.parametrize("A2", [ISOTROPIC, ALIGNED], ids=["isotropic", "aligned"])
    def test_contracts_back_to_the_second_order_tensor(self, A2):
        """A4_ijkk = A2_ij, since the closure averages the same distribution."""
        batched = A2.unsqueeze(0)
        A4 = IBOF_closure(batched)
        assert A4.shape == (1, 3, 3, 3, 3)
        assert torch.allclose(torch.einsum("nijkk->nij", A4), batched, atol=1e-10)

    def test_batched(self):
        A2 = ISOTROPIC.expand(4, 3, 3)
        assert IBOF_closure(A2).shape == (4, 3, 3, 3, 3)


class TestComputeOrientationAverage:
    def test_an_isotropic_stiffness_is_unchanged_by_any_orientation(self):
        """Averaging rotations of an isotropic tensor gives it back."""
        C = IsotropicElasticity3D(1000.0, 0.3).C
        A2 = ISOTROPIC.unsqueeze(0)
        C_avg = compute_orientation_average(C, A2, IBOF_closure(A2))
        assert torch.allclose(C_avg[0], C, atol=1e-10)

    def test_batched(self):
        C = IsotropicElasticity3D(1000.0, 0.3).C
        A2 = ISOTROPIC.expand(3, 3, 3)
        C_avg = compute_orientation_average(C, A2, IBOF_closure(A2))
        assert C_avg.shape == (3, 3, 3, 3, 3)


class TestTandonWengHomogenization:
    def test_fibers_stiffen_along_their_axis(self):
        result = tandon_weng_homogenization(_matrix(), _fiber(), a=20.0, volfrac=0.3)
        assert result.C.shape == (3, 3, 3, 3)
        # Slender fibers carry the load, so E_1 rises well above the matrix
        assert result.E_1 > _matrix().E
        # ... while the transverse directions stay far softer and equal
        assert result.E_2 < result.E_1
        assert torch.allclose(result.E_2, result.E_3)

    def test_a_vanishing_fiber_fraction_returns_the_matrix(self):
        result = tandon_weng_homogenization(_matrix(), _fiber(), a=20.0, volfrac=0.0)
        assert torch.allclose(result.E_1, _matrix().E, rtol=1e-6)
        assert torch.allclose(result.nu_12, _matrix().nu, rtol=1e-6)
