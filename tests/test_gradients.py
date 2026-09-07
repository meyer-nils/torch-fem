import math
from functools import partial

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
    model.temperatures[west | north] = 0.0

    element_volume = model.integrate_field()
    model.heat_flux[:, 0] = (
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


def test_solve_outputs_detached_without_differentiable_parameters():
    # Guard the differentiable_parameters trap: if a design variable requires
    # grad but is not declared to solve(), the adjoint cannot account for it.
    # The outputs must then be fully detached, so a forgotten declaration
    # fails loudly (no grad) instead of returning a silently wrong gradient.
    cantilever = _build_minimal_planar_cantilever()
    rho = torch.ones(cantilever.n_elem, requires_grad=True)
    cantilever.thickness = rho**3

    u, f, sigma, F, alpha = cantilever.solve()

    for out in (u, f, sigma, F, alpha):
        assert not out.requires_grad

    compliance = torch.inner(f.ravel(), u.ravel())
    assert not compliance.requires_grad


# Transient sensitivities. `time_integration(...)` differentiates by unrolling its
# Newton loop into the graph, where `solve(...)` uses an implicit adjoint. These
# pin the gradients against finite differences so that path can be changed safely.

T_OUTPUT = torch.tensor([0.0, 0.5, 1.0])
# Two internal steps per output interval, so a gradient has to chain across them.
DELTA_T = 0.25


def _build_transient_plate(
    thickness: torch.Tensor | float = 1.0, kappa: torch.Tensor | float = 400.0
) -> PlanarHeat:
    """A plate held cold on one edge and hot on the other, small enough to perturb."""
    material = IsotropicConductivity2D(kappa=kappa, rho=1.0e3)
    plate = PlanarHeat(*rect_quad(4, 4, 1.0, 1.0), material, thickness=thickness)
    west = torch.isclose(plate.nodes[:, 0], plate.nodes[:, 0].min())
    east = torch.isclose(plate.nodes[:, 0], plate.nodes[:, 0].max())
    plate.constraints[west | east] = True
    plate.temperatures[east, 0] = 100.0
    return plate


def _central_difference(loss, x0: float, h: float) -> float:
    """Slope of `loss` at `x0`, from a symmetric perturbation."""
    return (loss(x0 + h) - loss(x0 - h)) / (2.0 * h)


def test_time_integration_gradients_match_finite_differences_for_design_field():
    # A per-element design field enters both the conductivity and the capacity,
    # and the stored temperatures depend on it through every internal step.
    n_elem = _build_transient_plate().n_elem
    rho_0 = 0.6
    rho = torch.full((n_elem,), rho_0, requires_grad=True)

    plate = _build_transient_plate(thickness=rho)
    temperature, _, _, _, _ = plate.time_integration(
        T_OUTPUT, DELTA_T, differentiable_parameters=rho
    )
    gradient = torch.autograd.grad(temperature.sum(), rho)[0]

    def loss(element: int, value: float) -> float:
        field = torch.full((n_elem,), rho_0)
        field[element] = value
        perturbed = _build_transient_plate(thickness=field)
        temperature, _, _, _, _ = perturbed.time_integration(T_OUTPUT, DELTA_T)
        return float(temperature.sum())

    for element in (0, 4):
        slope = partial(loss, element)
        reference = _central_difference(slope, rho_0, 1e-6)
        assert torch.allclose(gradient[element], torch.tensor(reference), rtol=1e-6)


def test_time_integration_gradients_match_finite_differences_for_material():
    kappa_0 = 400.0
    kappa = torch.tensor(kappa_0, requires_grad=True)

    plate = _build_transient_plate(kappa=kappa)
    temperature, _, _, _, _ = plate.time_integration(
        T_OUTPUT, DELTA_T, differentiable_parameters=kappa
    )
    gradient = torch.autograd.grad(temperature.sum(), kappa)[0]

    def loss(value: float) -> float:
        perturbed = _build_transient_plate(kappa=value)
        temperature, _, _, _, _ = perturbed.time_integration(T_OUTPUT, DELTA_T)
        return float(temperature.sum())

    reference = _central_difference(loss, kappa_0, 1e-2)
    assert torch.allclose(gradient, torch.tensor(reference), rtol=1e-6)


def test_time_integration_gradients_reach_every_output_time():
    # Only the last output time would still pass a final-state check, so each
    # time is weighted differently and compared on its own.
    n_elem = _build_transient_plate().n_elem
    rho_0 = 0.6
    weights = torch.tensor([0.0, 1.0, 3.0])

    rho = torch.full((n_elem,), rho_0, requires_grad=True)
    plate = _build_transient_plate(thickness=rho)
    temperature, _, _, _, _ = plate.time_integration(
        T_OUTPUT, DELTA_T, differentiable_parameters=rho
    )
    gradient = torch.autograd.grad(
        (weights * temperature.sum(dim=1).ravel()).sum(), rho
    )[0]

    def loss(value: float) -> float:
        field = torch.full((n_elem,), rho_0)
        field[0] = value
        perturbed = _build_transient_plate(thickness=field)
        temperature, _, _, _, _ = perturbed.time_integration(T_OUTPUT, DELTA_T)
        return float((weights * temperature.sum(dim=1).ravel()).sum())

    reference = _central_difference(loss, rho_0, 1e-6)
    assert torch.allclose(gradient[0], torch.tensor(reference), rtol=1e-6)


def test_time_integration_differentiates_without_declared_parameters():
    # Characterizes the unrolled path: `solve(...)` detaches its outputs unless a
    # parameter is declared, because its adjoint cannot account for an undeclared
    # one, while `time_integration(...)` differentiates through the graph and so
    # returns a gradient either way, ignoring the argument in its time loop.
    # Moving it onto the adjoint is expected to change this.
    n_elem = _build_transient_plate().n_elem
    rho = torch.full((n_elem,), 0.6, requires_grad=True)

    plate = _build_transient_plate(thickness=rho)
    temperature, _, _, _, _ = plate.time_integration(T_OUTPUT, DELTA_T)

    assert temperature.requires_grad
    gradient = torch.autograd.grad(temperature.sum(), rho)[0]
    assert torch.isfinite(gradient).all()
    assert (gradient != 0.0).any()
