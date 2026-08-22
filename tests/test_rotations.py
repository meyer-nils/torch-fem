import pytest
import torch

from torchfem.materials import OrthotropicElasticity3D, OrthotropicElasticityPlaneStress
from torchfem.rotations import axis_rotation, euler_rotation, planar_rotation

mat_2D = OrthotropicElasticityPlaneStress(1.0, 2.0, 0.3, 1.0)
mat_2D_rot = OrthotropicElasticityPlaneStress(2.0, 1.0, 0.6, 1.0)

mat_3D = OrthotropicElasticity3D(1.0, 2.0, 3.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0)
mat_3D_rot = OrthotropicElasticity3D(1.0, 3.0, 2.0, 0.3, 0.3, 0.45, 1.0, 1.0, 1.0)

X = torch.tensor([1.0, 0.0, 0.0])
Y = torch.tensor([0.0, 1.0, 0.0])
Z = torch.tensor([0.0, 0.0, 1.0])

QUARTER = torch.tensor(torch.pi / 2)


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
    "R",
    [
        planar_rotation(20.0),
        axis_rotation(X, 20.0),
        euler_rotation(torch.tensor([20.0, 30.0, 40.0])),
    ],
)
def test_proper_rotation(R):
    # A reflection satisfies orthogonality too, so the determinant is checked.
    assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0))


def test_planar_rotation_turns_counter_clockwise():
    rotated = planar_rotation(QUARTER) @ torch.tensor([1.0, 0.0])
    assert torch.allclose(rotated, torch.tensor([0.0, 1.0]), atol=1e-15)


@pytest.mark.parametrize(
    "axis, vector, expected", [(Z, X, Y), (X, Y, Z), (Y, Z, X), (Y, X, -Z)]
)
def test_axis_rotation_follows_the_right_hand_rule(axis, vector, expected):
    assert torch.allclose(axis_rotation(axis, QUARTER) @ vector, expected, atol=1e-15)


@pytest.mark.parametrize("index, axis", [(0, Z), (1, Y), (2, X)])
def test_euler_rotation_composes_axis_rotations(index, axis):
    # Intrinsic z-y-x, so a single angle is the rotation about its own axis.
    angles = torch.zeros(3)
    angles[index] = 0.4
    assert torch.allclose(euler_rotation(angles), axis_rotation(axis, angles[index]))


def test_rotated_ply_is_stiff_along_its_own_angle():
    ply = OrthotropicElasticityPlaneStress(100.0, 1.0, 0.3, 1.0)
    ply.rotate(planar_rotation(torch.tensor(torch.pi / 4)))

    def modulus(direction):
        strain = torch.outer(direction, direction)
        return torch.einsum("ijkl,kl,ij->", ply.C, strain, strain)

    diagonal = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
    assert modulus(diagonal) > 10 * modulus(diagonal * torch.tensor([1.0, -1.0]))


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
