import torch

from torchfem import Planar
from torchfem.materials import IsotropicElasticityPlaneStress

torch.set_default_dtype(torch.float64)


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
