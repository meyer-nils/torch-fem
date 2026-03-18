import torch

from torchfem import Planar, PlanarHeat
from torchfem.materials import IsotropicConductivity2D, IsotropicElasticityPlaneStress
from torchfem.mesh import rect_quad


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


def test_gradients_incremental_final_matches_single_step(float64):
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


def test_gradients_planar_heat_topology_parameter_autograd(float64):
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
