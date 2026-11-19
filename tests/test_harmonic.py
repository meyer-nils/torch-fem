import math

import pytest
import torch

from torchfem import Planar, Solid, Truss
from torchfem.materials import (
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStress,
)
from torchfem.mesh import cube_hexa, rect_quad


def test_harmonic_scalar_and_vector_frequency():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True
    model.forces[-1, 1] = 100.0

    # 1. Scalar frequency input
    u_scalar = model.solve_harmonic(torch.tensor(1.0e6), n_modes=4)
    assert u_scalar.shape == (model.n_nod, 2)
    assert u_scalar.dtype == torch.complex128

    # 2. Vector of frequencies input
    omegas = torch.tensor([1.0e5, 1.0e6, 2.0e6])
    u_vector = model.solve_harmonic(omegas, n_modes=4)
    assert u_vector.shape == (3, model.n_nod, 2)
    assert u_vector.dtype == torch.complex128
    assert torch.allclose(u_vector[1], u_scalar, atol=1e-12)


def test_harmonic_single_dof_analytical():
    """Verify single-mode response against textbook analytical SDOF FRF."""
    n_elem = 5
    E = 1000.0
    rho = 10.0
    F0 = 50.0
    zeta = 0.05

    x = torch.linspace(0.0, 1.0, n_elem + 1)
    nodes = torch.stack([x, torch.zeros_like(x)], dim=1)
    elements = torch.stack([torch.arange(n_elem), torch.arange(1, n_elem + 1)], dim=1)

    mat = IsotropicElasticity1D(E=E, rho=rho)
    model = Truss(nodes, elements, mat)
    # Constrain left end (node 0) and constrain all y-displacement
    model.constraints[0, :] = True
    model.constraints[:, 1] = True
    model.forces[-1, 0] = F0

    omega_sq, modes = model.solve_modes(1)
    omega_n = math.sqrt(float(omega_sq[0]))

    # Test at three frequency ratios: r = 0.5, r = 1.0 (resonance), r = 1.5
    ratios = torch.tensor([0.5, 1.0, 1.5])
    omegas = ratios * omega_n

    u = model.solve_harmonic(omegas, n_modes=1, damping=zeta)

    # For a modal coordinate with M-normalized mode phi:
    # q = (phi * F0) / ((omega_n^2 - Omega^2) + 2j * zeta * omega_n * Omega)
    # Physical displacement at tip: u_tip = phi * q
    phi_tip = float(modes[0, -1, 0])
    p = phi_tip * F0

    for i, Om_t in enumerate(omegas):
        Om = float(Om_t)
        denom = (omega_n**2 - Om**2) + 2j * zeta * omega_n * Om
        q_analytic = p / denom
        u_analytic = q_analytic * phi_tip

        u_tip_fem = u[i, -1, 0]
        assert torch.isclose(u_tip_fem.real, torch.tensor(u_analytic.real), atol=1e-10)
        assert torch.isclose(u_tip_fem.imag, torch.tensor(u_analytic.imag), atol=1e-10)


def test_harmonic_precomputed_modes():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True
    model.forces[-1, 1] = 50.0

    modes = model.solve_modes(4)
    omegas = torch.linspace(1e5, 1e6, 5)

    u1 = model.solve_harmonic(omegas, n_modes=4)
    u2 = model.solve_harmonic(omegas, modes=modes)

    assert torch.allclose(u1, u2, atol=1e-12)


def test_harmonic_rayleigh_damping():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True
    model.forces[-1, 1] = 50.0

    omega_sq, modes = model.solve_modes(3)
    omega_n = torch.sqrt(omega_sq)

    alpha, beta = 10.0, 1.0e-7
    # Equivalent modal damping: zeta_j = alpha / (2 * omega_j) + beta * omega_j / 2
    zeta_equiv = 0.5 * (alpha / omega_n + beta * omega_n)

    omegas = torch.linspace(5e5, 1.5e6, 4)
    u_rayleigh = model.solve_harmonic(
        omegas, n_modes=3, rayleigh=(alpha, beta), modes=(omega_sq, modes)
    )
    u_modal = model.solve_harmonic(
        omegas, n_modes=3, damping=zeta_equiv, modes=(omega_sq, modes)
    )

    assert torch.allclose(u_rayleigh, u_modal, atol=1e-12)


def test_harmonic_forces_override():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True
    model.forces[:] = 0.0

    custom_forces = torch.zeros_like(model.forces)
    custom_forces[-1, 0] = 75.0

    u = model.solve_harmonic(torch.tensor(1.0e6), n_modes=3, forces=custom_forces)
    assert u.abs().max() > 0.0

    with pytest.raises(ValueError, match="Forces must have shape"):
        model.solve_harmonic(torch.tensor(1.0e6), forces=torch.zeros(5, 5))


def test_harmonic_negative_frequency_raises():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True

    with pytest.raises(ValueError, match="non-negative"):
        model.solve_harmonic(torch.tensor([-10.0, 50.0]))


def test_harmonic_differentiability_wrt_material_and_density():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    E_param = torch.tensor(210e3, requires_grad=True)
    rho_param = torch.tensor(7.8e-9, requires_grad=True)

    material = IsotropicElasticityPlaneStress(E=E_param, nu=0.3, rho=rho_param)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True
    model.forces[-1, 1] = 100.0

    omegas = torch.tensor([1.0e6, 2.0e6])
    u = model.solve_harmonic(omegas, n_modes=4, damping=0.03)

    # Objective: sum of squared displacement magnitudes across frequencies
    loss = (u.abs() ** 2).sum()
    loss.backward()

    assert E_param.grad is not None
    assert rho_param.grad is not None
    assert not torch.isnan(E_param.grad)
    assert not torch.isnan(rho_param.grad)
    assert E_param.grad.abs() > 0.0
    assert rho_param.grad.abs() > 0.0


def test_harmonic_solid_3d():
    nodes, elements = cube_hexa(2, 2, 2, 1.0, 1.0, 1.0)
    material = IsotropicElasticity3D(E=210e3, nu=0.3, rho=7.8e-9)
    model = Solid(nodes, elements, material)
    # Clamp bottom surface z=0
    model.constraints[model.nodes[:, 2] == 0.0, :] = True
    # Harmonic load on top node
    model.forces[-1, 2] = 500.0

    omegas = torch.linspace(5e5, 2e6, 3)
    u = model.solve_harmonic(omegas, n_modes=4, damping=0.02)

    assert u.shape == (3, model.n_nod, 3)
    assert u.dtype == torch.complex128
    assert not torch.isnan(u.abs()).any()
    assert (u.abs() > 0.0).any()
