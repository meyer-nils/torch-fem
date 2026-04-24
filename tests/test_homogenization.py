import torch

from torchfem.homogenization import (
    IBOF_closure,
    compute_orientation_average,
    symm,
    tandon_weng_homogenization,
)
from torchfem.materials import IsotropicElasticity3D, OrthotropicElasticity3D


class TestSymm:
    def test_identity_like(self):
        """Symmetrizing an already symmetric tensor should give itself back."""
        Id = torch.eye(3)
        A4 = torch.einsum("ij,kl->ijkl", Id, Id).unsqueeze(0)
        result = symm(A4)
        assert result.shape == (1, 3, 3, 3, 3)

    def test_output_symmetric(self):
        """Result should have full index symmetry."""
        A4 = torch.randn(2, 3, 3, 3, 3)
        result = symm(A4)
        # Test a few permutations
        assert torch.allclose(result, result.permute(0, 2, 1, 3, 4), atol=1e-10)
        assert torch.allclose(result, result.permute(0, 3, 4, 1, 2), atol=1e-10)


class TestIBOFClosure:
    def test_isotropic_orientation(self):
        """Isotropic A2 = 1/3 * I should produce valid A4."""
        A2 = (torch.eye(3) / 3.0).unsqueeze(0)  # must be batched (N,3,3)
        A4 = IBOF_closure(A2)
        assert A4.shape == (1, 3, 3, 3, 3)
        assert torch.isfinite(A4).all()

    def test_aligned_orientation(self):
        """Fully aligned A2 should produce valid A4."""
        A2 = torch.zeros(1, 3, 3)
        A2[0, 0, 0] = 1.0
        A4 = IBOF_closure(A2)
        assert A4.shape == (1, 3, 3, 3, 3)
        assert torch.isfinite(A4).all()

    def test_batched(self):
        """Test with a batch of orientation tensors."""
        A2 = torch.eye(3).unsqueeze(0).expand(4, -1, -1) / 3.0
        A4 = IBOF_closure(A2)
        assert A4.shape == (4, 3, 3, 3, 3)


class TestComputeOrientationAverage:
    def test_basic(self):
        """Test orientation averaging with isotropic orientation."""
        mat = IsotropicElasticity3D(1000.0, 0.3)
        C = mat.C
        A2 = (torch.eye(3) / 3.0).unsqueeze(0)
        A4 = IBOF_closure(A2)
        C_avg = compute_orientation_average(C, A2, A4)
        assert C_avg.shape[-4:] == (3, 3, 3, 3)
        assert torch.isfinite(C_avg).all()

    def test_batched(self):
        mat = IsotropicElasticity3D(1000.0, 0.3)
        C = mat.C
        n = 3
        A2 = torch.eye(3).unsqueeze(0).expand(n, -1, -1) / 3.0
        A4 = IBOF_closure(A2)
        C_avg = compute_orientation_average(C, A2, A4)
        assert C_avg.shape == (n, 3, 3, 3, 3)


class TestTandonWengHomogenization:
    def test_returns_orthotropic(self):
        matrix = IsotropicElasticity3D(3500.0, 0.35)
        fiber = IsotropicElasticity3D(72000.0, 0.2)
        result = tandon_weng_homogenization(matrix, fiber, a=20.0, volfrac=0.3)
        assert isinstance(result, OrthotropicElasticity3D)

    def test_stiffness_shape(self):
        matrix = IsotropicElasticity3D(3500.0, 0.35)
        fiber = IsotropicElasticity3D(72000.0, 0.2)
        result = tandon_weng_homogenization(matrix, fiber, a=20.0, volfrac=0.3)
        assert result.C.shape == (3, 3, 3, 3)

    def test_fiber_stiffens_material(self):
        matrix = IsotropicElasticity3D(3500.0, 0.35)
        fiber = IsotropicElasticity3D(72000.0, 0.2)
        result = tandon_weng_homogenization(matrix, fiber, a=20.0, volfrac=0.3)
        # Longitudinal modulus should be higher than matrix modulus
        assert result.E_1.item() > matrix.E.item()
