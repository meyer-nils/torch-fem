import math

import torch

from torchfem import Planar, PlanarHeat, Solid
from torchfem.materials import (
    Hyperelastic3D,
    IsotropicConductivity2D,
    IsotropicElasticityPlaneStress,
)
from torchfem.mesh import cube_hexa, rect_quad


def _build_minimal_planar_cantilever() -> Planar:
    material = IsotropicElasticityPlaneStress(E=1000.0, nu=0.3)
    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    )
    elements = torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]])

    cantilever = Planar(nodes, elements, material)
    cantilever.forces[5, 1] = -1.0
    cantilever.constraints[[0, 3], :] = True
    return cantilever


def test_gradients_incremental_final_matches_single_step():
    cantilever = _build_minimal_planar_cantilever()
    cantilever.thickness.requires_grad = True

    # Full load in one solve.
    u_single, f_single, _, _, _ = cantilever.solve(
        differentiable_parameters=cantilever.thickness
    )
    compliance_single = torch.inner(f_single.ravel(), u_single.ravel())
    grad_single = torch.autograd.grad(compliance_single, cantilever.thickness)[0]

    # Same load reached through incremental loading.
    increments = torch.linspace(0.1, 1.0, 5)
    u_inc, f_inc, _, _, _ = cantilever.solve(
        increments=increments,
        return_intermediate=True,
        differentiable_parameters=cantilever.thickness,
    )
    compliance_final = torch.inner(f_inc[-1].ravel(), u_inc[-1].ravel())
    grad_final = torch.autograd.grad(compliance_final, cantilever.thickness)[0]

    assert torch.allclose(grad_final, grad_single, atol=1e-9, rtol=1e-7)


def test_gradients_incremental_force_matches_single_step():
    # Load-side parameters: du_n depends on the accumulated state, so the
    # incremental gradient is only correct if sensitivities chain across
    # increments through the previous state.
    grads = {}
    for increments in [None, torch.linspace(0.1, 1.0, 5)]:
        cantilever = _build_minimal_planar_cantilever()
        cantilever.forces = torch.zeros_like(cantilever.nodes)
        cantilever.forces[5, 1] = -1.0
        cantilever.forces.requires_grad = True

        if increments is None:
            u, _, _, _, _ = cantilever.solve(
                differentiable_parameters=cantilever.forces
            )
        else:
            u, _, _, _, _ = cantilever.solve(
                increments=increments,
                return_intermediate=True,
                differentiable_parameters=cantilever.forces,
            )
            u = u[-1]
        key = "single" if increments is None else "incremental"
        grads[key] = torch.autograd.grad(u.sum(), cantilever.forces)[0]

    assert torch.allclose(grads["incremental"], grads["single"], atol=1e-9, rtol=1e-7)


def test_gradients_incremental_nonlinear_matches_analytical():
    # Uniaxial Neo-Hookean stretch of a unit cube: the deformation is
    # homogeneous, so the reaction force sensitivities w.r.t. the Lamé
    # parameters follow from the analytical uniaxial response.
    E = 1000.0
    nu = 0.3
    lbd = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    U = 2.0

    def psi(F, params):
        MU = params[0]
        LBD = params[1]
        C = F.transpose(-1, -2) @ F
        logJ = 0.5 * torch.logdet(C)
        return MU / 2 * (torch.trace(C) - 3.0) - MU * logJ + LBD / 2 * logJ**2

    nodes, elements = cube_hexa(3, 3, 3)
    params = torch.tensor([mu, lbd], requires_grad=True)
    box = Solid(nodes, elements, Hyperelastic3D(psi, params))
    right = nodes[:, 0] == 1.0
    box.constraints[nodes[:, 0] == 0.0, 0] = True
    box.constraints[right, 0] = True
    box.constraints[nodes[:, 1] == 0.5, 1] = True
    box.constraints[nodes[:, 2] == 0.5, 2] = True
    box.displacements[right, 0] = U

    # Geometric stretch increments and reaction force sensitivity
    lam_pts = torch.logspace(0, math.log10(1.0 + U), 8)
    u, f, _, _, _ = box.solve(
        increments=(lam_pts - 1.0) / U, nlgeom=True, differentiable_parameters=params
    )
    reaction = f[right, 0].sum()
    grad = torch.autograd.grad(reaction, params)[0]

    # Analytical solution: lateral stretch from mu * (J / lam - 1) + lbd * ln J = 0
    # via Newton iterations, then implicit differentiation of the reaction
    # R = mu * (lam - 1 / lam) + lbd * ln(J) / lam.
    lam = 1.0 + U
    J = 1.0
    for _ in range(100):
        J -= (mu * (J / lam - 1.0) + lbd * math.log(J)) / (mu / lam + lbd / J)
    denominator = mu / lam + lbd / J
    dJ_dmu = -(J / lam - 1.0) / denominator
    dJ_dlbd = -math.log(J) / denominator
    dR_dmu = (lam - 1.0 / lam) + lbd / (lam * J) * dJ_dmu
    dR_dlbd = math.log(J) / lam + lbd / (lam * J) * dJ_dlbd

    reference = torch.tensor([dR_dmu, dR_dlbd])
    assert torch.allclose(grad, reference, rtol=1e-6, atol=1e-8)


def test_gradients_planar_heat_topology_parameter_autograd():
    model = PlanarHeat(*rect_quad(5, 5, 1.0, 1.0), IsotropicConductivity2D(kappa=400.0))
    west = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].min())
    north = torch.isclose(model.nodes[:, 1], model.nodes[:, 1].max())
    model.constraints[west | north] = True
    model.displacements[west | north] = 0.0

    element_volume = model.integrate_field()
    model.forces[:, 0] = (
        model.assemble_rhs(
            (1000.0 * element_volume / element_volume.sum())
            .unsqueeze(1)
            .repeat(1, model.etype.nodes)
        )
        / model.etype.nodes
    )

    rho_nodes = 0.4 * torch.ones(len(model.nodes), requires_grad=True)
    N, _, _ = model.eval_shape_functions(model.etype.ipoints.sum(dim=0))
    model.thickness = torch.einsum("EN, N -> E", rho_nodes[model.elements], N) ** 3.0

    temperature, internal_force, _, _, _ = model.solve(
        differentiable_parameters=rho_nodes
    )
    compliance = torch.inner(internal_force.ravel(), temperature.ravel())
    sensitivity = torch.autograd.grad(compliance, rho_nodes)[0]

    assert torch.isfinite(sensitivity).all()
