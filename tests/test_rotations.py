import pytest
import torch

from torchfem.materials import OrthotropicElasticity3D, OrthotropicElasticityPlaneStress
from torchfem.rotations import (
    axis_rotation,
    euler_rotation,
    planar_rotation,
    voigt_strain_rotation,
    voigt_stress_rotation,
)
from torchfem.utils import strain2voigt, stress2voigt, voigt2strain, voigt2stress

mat_2D = OrthotropicElasticityPlaneStress(1.0, 2.0, 0.3, 1.0)
mat_2D_rot = OrthotropicElasticityPlaneStress(2.0, 1.0, 0.6, 1.0)

mat_3D = OrthotropicElasticity3D(1.0, 2.0, 3.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0)
mat_3D_rot = OrthotropicElasticity3D(1.0, 3.0, 2.0, 0.3, 0.3, 0.45, 1.0, 1.0, 1.0)

X = torch.tensor([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "R",
    [
        planar_rotation(20.0),
        axis_rotation(X, 20.0),
        euler_rotation(torch.tensor([20.0, 30.0, 40.0])),
    ],
)
def test_orthogonality(R):
    assert torch.allclose(R.transpose(-1, -2), torch.linalg.inv(R))


@pytest.mark.parametrize(
    "mat, mat_rot, R",
    [
        (mat_3D, mat_3D_rot, axis_rotation(X, torch.pi / 2)),
        (mat_2D, mat_2D_rot, planar_rotation(torch.pi / 2)),
    ],
)
def test_stiffness_rotation(mat, mat_rot, R):
    rotated_material = mat.rotate(R)
    assert torch.allclose(rotated_material.C, mat_rot.C, atol=1e-6)


@pytest.mark.parametrize(
    "voigt, R",
    [
        (torch.rand(6), axis_rotation(X, torch.rand(1))),
        (torch.rand(3), planar_rotation(torch.rand(1))),
    ],
)
def test_stress_rotation(voigt, R):
    voigt_rot = voigt_stress_rotation(R) @ voigt
    sigma_rot = stress2voigt(R @ voigt2stress(voigt) @ R.transpose(-1, -2))
    assert torch.allclose(sigma_rot, voigt_rot, atol=1e-6)


@pytest.mark.parametrize(
    "voigt, R",
    [
        (torch.rand(6), axis_rotation(X, torch.rand(1))),
        (torch.rand(3), planar_rotation(torch.rand(1))),
    ],
)
def test_strain_rotation(voigt, R):
    voigt_rot = voigt_strain_rotation(R) @ voigt
    sigma_rot = strain2voigt(R @ voigt2strain(voigt) @ R.transpose(-1, -2))
    assert torch.allclose(sigma_rot, voigt_rot, atol=1e-6)
