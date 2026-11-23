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
from torchfem.sparse import differentiable_sparse_solve


def test_harmonic_scalar_and_vector_frequency():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True
    model.forces[-1, 1] = 100.0

    # 1. Scalar frequency input
    u_scalar = model.solve_harmonic(torch.tensor(1.0e4), damping=0.02)
    assert u_scalar.shape == (model.n_nod, 2)
    assert u_scalar.dtype == torch.complex128

    # 2. Vector of frequencies input
    omegas = torch.tensor([5.0e3, 1.0e4, 2.0e4])
    u_vector = model.solve_harmonic(omegas, damping=0.02)
    assert u_vector.shape == (3, model.n_nod, 2)
    assert u_vector.dtype == torch.complex128
    assert torch.allclose(u_vector[1], u_scalar, atol=1e-12)


def test_harmonic_single_dof_analytical():
    """Verify direct dynamic solve against textbook analytical SDOF FRF."""
    E = 1000.0
    rho = 10.0
    F0 = 50.0
    zeta = 0.05
    L = 1.0

    nodes = torch.tensor([[0.0, 0.0], [L, 0.0]])
    elements = torch.tensor([[0, 1]])
    mat = IsotropicElasticity1D(E=E, rho=rho)
    model = Truss(nodes, elements, mat)
    model.constraints[0, :] = True
    model.constraints[1, 1] = True
    model.forces[1, 0] = F0

    k_val = E * 1.0 / L
    m_e = model.integrate_mass()
    con = torch.nonzero(model.constraints.ravel(), as_tuple=False).ravel()
    M_mat = model.assemble_matrix(m_e, con)
    m_val = float(M_mat.values()[model.diag_map[2]])

    omega_n = math.sqrt(k_val / m_val)
    ratios = torch.tensor([0.5, 1.0, 1.5])
    omegas = ratios * omega_n

    beta = 2.0 * zeta / omega_n
    u = model.solve_harmonic(omegas, rayleigh=(0.0, beta))

    for i, Om_t in enumerate(omegas):
        Om = float(Om_t)
        denom = (k_val - Om**2 * m_val) + 1j * Om * (beta * k_val)
        u_analytic = F0 / denom
        u_tip_fem = u[i, 1, 0]
        assert torch.isclose(u_tip_fem.real, torch.tensor(u_analytic.real), atol=1e-10)
        assert torch.isclose(u_tip_fem.imag, torch.tensor(u_analytic.imag), atol=1e-10)


def test_harmonic_rayleigh_and_loss_damping():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True
    model.forces[-1, 1] = 50.0

    omegas = torch.linspace(5e3, 2e4, 3)

    # Rayleigh damping
    u_rayleigh = model.solve_harmonic(omegas, rayleigh=(10.0, 1.0e-6))
    assert u_rayleigh.shape == (3, model.n_nod, 2)
    assert not torch.isnan(u_rayleigh.abs()).any()
    assert (u_rayleigh.abs() > 0.0).any()

    # Structural / loss factor damping
    u_loss = model.solve_harmonic(omegas, damping=0.05)
    assert u_loss.shape == (3, model.n_nod, 2)
    assert not torch.isnan(u_loss.abs()).any()
    assert (u_loss.abs() > 0.0).any()


def test_harmonic_zero_frequency_matches_static():
    """At zero frequency without damping, harmonic solve matches static solve."""
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True
    model.forces[-1, 1] = 100.0

    con = torch.nonzero(model.constraints.ravel(), as_tuple=False).ravel()
    k_e = model.k0()
    K = model.assemble_matrix(k_e, con)
    f_vec = model.forces.ravel().clone()
    f_vec[con] = 0.0
    u_static = differentiable_sparse_solve(K, f_vec).reshape(
        model.n_nod, model.n_dof_per_node
    )

    u_harmonic = model.solve_harmonic(torch.tensor(0.0), damping=0.0)

    assert torch.allclose(u_harmonic.real, u_static, atol=1e-12)
    assert torch.allclose(
        u_harmonic.imag, torch.zeros_like(u_harmonic.imag), atol=1e-12
    )


def test_harmonic_forces_override():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True
    model.forces[:] = 0.0

    custom_forces = torch.zeros_like(model.forces)
    custom_forces[-1, 0] = 75.0

    u = model.solve_harmonic(torch.tensor(1.0e4), damping=0.02, forces=custom_forces)
    assert u.abs().max() > 0.0

    with pytest.raises(ValueError, match="Forces must have shape"):
        model.solve_harmonic(torch.tensor(1.0e4), forces=torch.zeros(5, 5))


def test_harmonic_negative_frequency_raises():
    nodes, elements = rect_quad(3, 3, 1.0, 1.0)
    material = IsotropicElasticityPlaneStress(E=210e3, nu=0.3, rho=7.8e-9)
    model = Planar(nodes, elements, material, thickness=0.01)
    model.constraints[model.nodes[:, 0] == 0.0, :] = True

    with pytest.raises(ValueError, match="non-negative"):
        model.solve_harmonic(torch.tensor([-10.0, 50.0]))


def test_harmonic_gradient_finite_differences():
    """Verify autograd sensitivities against central finite differences.

    Verifies sensitivities for uniform E, uniform rho, and per-element thickness
    under direct dynamic stiffness solve as requested by maintainer review.
    """

    def run_fwd(E_val, rho_val, t_val):
        nodes, elements = rect_quad(3, 3, 1.0, 1.0)
        mat = IsotropicElasticityPlaneStress(E=E_val, nu=0.3, rho=rho_val)
        model = Planar(nodes, elements, mat, thickness=t_val)
        model.constraints[model.nodes[:, 0] == 0.0, :] = True
        model.forces[-1, 1] = 100.0
        omegas = torch.tensor([1000.0, 2000.0])
        u = model.solve_harmonic(omegas, rayleigh=(10.0, 1e-7))
        return (u.abs() ** 2).sum()

    # 1. Uniform E sensitivity
    E = torch.tensor(210e3, requires_grad=True)
    rho = torch.tensor(7.8e-9)
    t = torch.tensor([0.01, 0.015, 0.02, 0.012])
    loss = run_fwd(E, rho, t)
    loss.backward()
    assert E.grad is not None
    grad_E = E.grad.item()

    eps_E = 1e-1
    fd_E = (
        run_fwd(torch.tensor(210e3 + eps_E), rho, t)
        - run_fwd(torch.tensor(210e3 - eps_E), rho, t)
    ).item() / (2 * eps_E)
    rel_err_E = abs(grad_E - fd_E) / abs(fd_E)
    assert rel_err_E < 1e-4

    # 2. Uniform rho sensitivity
    E = torch.tensor(210e3)
    rho = torch.tensor(7.8e-9, requires_grad=True)
    loss = run_fwd(E, rho, t)
    loss.backward()
    assert rho.grad is not None
    grad_rho = rho.grad.item()

    eps_rho = 1e-12
    fd_rho = (
        run_fwd(E, torch.tensor(7.8e-9 + eps_rho), t)
        - run_fwd(E, torch.tensor(7.8e-9 - eps_rho), t)
    ).item() / (2 * eps_rho)
    rel_err_rho = abs(grad_rho - fd_rho) / abs(fd_rho)
    assert rel_err_rho < 1e-4

    # 3. Per-element thickness sensitivity
    E = torch.tensor(210e3)
    rho = torch.tensor(7.8e-9)
    t = torch.tensor([0.01, 0.015, 0.02, 0.012], requires_grad=True)
    loss = run_fwd(E, rho, t)
    loss.backward()
    assert t.grad is not None
    grad_t = t.grad.detach().clone()

    eps_t = 1e-6
    fd_t = []
    for i in range(4):
        tp = torch.tensor([0.01, 0.015, 0.02, 0.012])
        tm = torch.tensor([0.01, 0.015, 0.02, 0.012])
        tp[i] += eps_t
        tm[i] -= eps_t
        fd_t.append((run_fwd(E, rho, tp) - run_fwd(E, rho, tm)).item() / (2 * eps_t))
    fd_t = torch.tensor(fd_t)
    rel_err_t = (grad_t - fd_t).abs() / fd_t.abs()
    assert (rel_err_t < 1e-4).all()


def test_harmonic_solid_3d():
    nodes, elements = cube_hexa(2, 2, 2, 1.0, 1.0, 1.0)
    material = IsotropicElasticity3D(E=210e3, nu=0.3, rho=7.8e-9)
    model = Solid(nodes, elements, material)
    # Clamp bottom surface z=0
    model.constraints[model.nodes[:, 2] == 0.0, :] = True
    # Harmonic load on top node
    model.forces[-1, 2] = 500.0

    omegas = torch.linspace(5e3, 2e4, 3)
    u = model.solve_harmonic(omegas, rayleigh=(10.0, 1e-6))

    assert u.shape == (3, model.n_nod, 3)
    assert u.dtype == torch.complex128
    assert not torch.isnan(u.abs()).any()
    assert (u.abs() > 0.0).any()
