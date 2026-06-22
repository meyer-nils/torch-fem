import torch

from torchfem import Laminate, Shell
from torchfem.materials import (
    IsotropicElasticityPlaneStress,
    OrthotropicElasticityPlaneStress,
)


def square_plate():
    """A unit square in the z=0 plane meshed with two triangles."""
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return nodes, elements


def cantilever_tip_displacement(shell, differentiable_parameters=None):
    """Clamp the x=0 edge, pull the x=1 edge in +z, return tip deflection."""
    constraints = torch.zeros(len(shell.nodes), 6, dtype=torch.bool)
    constraints[[0, 3]] = True  # clamp x = 0 edge (all 6 dofs)
    shell.constraints = constraints
    forces = torch.zeros(len(shell.nodes), 6)
    forces[[1, 2], 2] = 0.5  # transverse load on x = 1 edge
    shell.forces = forces
    u, _, _, _, _ = shell.solve(differentiable_parameters=differentiable_parameters)
    return u[:, 2]


def test_single_layer_matches_homogeneous():
    """A 1-layer laminate must reproduce the homogeneous shell exactly."""
    nodes, elements = square_plate()
    mat = IsotropicElasticityPlaneStress(E=70000.0, nu=0.3)
    t = 2.0

    homog = Shell(nodes, elements, mat, thickness=t)
    lam = Shell(nodes, elements, Laminate([mat], [t], [0.0]))

    u_homog = cantilever_tip_displacement(homog)
    u_lam = cantilever_tip_displacement(lam)

    assert torch.allclose(u_homog, u_lam, atol=1e-10)


def test_single_layer_mass_matches_homogeneous():
    nodes, elements = square_plate()
    mat = IsotropicElasticityPlaneStress(E=70000.0, nu=0.3, rho=2.7e-9)
    t = 1.5

    homog = Shell(nodes, elements, mat, thickness=t)
    lam = Shell(nodes, elements, Laminate([mat], [t], [0.0]))

    assert torch.allclose(homog.integrate_mass(), lam.integrate_mass(), atol=1e-12)


def test_abd_against_clt_reference():
    """Closed-form ABD of an isotropic single layer about the mid-plane."""
    E, nu, t = 70000.0, 0.3, 2.0
    mat = IsotropicElasticityPlaneStress(E=E, nu=nu)
    lam = Laminate([mat], [t], [0.0]).vectorize(1)
    A, B, D = lam.abd

    # Isotropic plane-stress in-plane stiffness
    q = E / (1 - nu**2)
    A_ref = t * torch.tensor(
        [[q, nu * q, 0.0], [nu * q, q, 0.0], [0.0, 0.0, 0.5 * (1 - nu) * q]]
    )
    D_ref = (t**3 / 12.0) * torch.tensor(
        [[q, nu * q, 0.0], [nu * q, q, 0.0], [0.0, 0.0, 0.5 * (1 - nu) * q]]
    )

    assert torch.allclose(A[0], A_ref, atol=1e-6)
    assert torch.allclose(B[0], torch.zeros(3, 3), atol=1e-8)
    assert torch.allclose(D[0], D_ref, atol=1e-6)


def test_symmetric_laminate_has_zero_coupling():
    """A symmetric stacking sequence has B = 0."""
    gfrp = OrthotropicElasticityPlaneStress(
        E_1=40000.0, E_2=10000.0, nu_12=0.3, G_12=5000.0, G_13=5000.0, G_23=4000.0
    )
    # Symmetric [0 / 90 / 90 / 0]
    lam = Laminate(
        materials=[gfrp, gfrp, gfrp, gfrp],
        thicknesses=[0.25, 0.25, 0.25, 0.25],
        angles=[0.0, torch.pi / 2, torch.pi / 2, 0.0],
    ).vectorize(1)
    _, B, _ = lam.abd
    assert torch.allclose(B[0], torch.zeros(3, 3), atol=1e-6)


def test_unsymmetric_laminate_has_coupling():
    """An unsymmetric stacking sequence has nonzero B."""
    gfrp = OrthotropicElasticityPlaneStress(
        E_1=40000.0, E_2=10000.0, nu_12=0.3, G_12=5000.0, G_13=5000.0, G_23=4000.0
    )
    lam = Laminate(
        materials=[gfrp, gfrp],
        thicknesses=[0.5, 0.5],
        angles=[0.0, torch.pi / 2],
    ).vectorize(1)
    _, B, _ = lam.abd
    assert B[0].abs().max() > 1e-3


def test_glare_style_laminate_solves():
    """A GLARE-style fiber-metal laminate solves and stays finite."""
    nodes, elements = square_plate()
    gfrp = OrthotropicElasticityPlaneStress(
        E_1=54000.0, E_2=9400.0, nu_12=0.33, G_12=5500.0, G_13=5500.0, G_23=3000.0
    )
    alu = IsotropicElasticityPlaneStress(E=72000.0, nu=0.33)
    layup = Laminate(
        materials=[alu, gfrp, gfrp, alu],
        thicknesses=[0.3, 0.125, 0.125, 0.3],
        angles=[0.0, 0.0, torch.pi / 2, 0.0],
    )
    plate = Shell(nodes, elements, layup)
    u = cantilever_tip_displacement(plate)
    assert torch.all(torch.isfinite(u))
    assert u.abs().max() > 0.0


def test_laminate_gradient_flows():
    """Gradients flow through a layer thickness to a displacement objective."""
    nodes, elements = square_plate()
    mat = IsotropicElasticityPlaneStress(E=70000.0, nu=0.3)
    t = torch.tensor(2.0, requires_grad=True)
    layup = Laminate([mat], [t], [0.0])
    plate = Shell(nodes, elements, layup)
    u = cantilever_tip_displacement(plate, differentiable_parameters=t)
    obj = u.abs().sum()
    obj.backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad)


def test_unsymmetric_tangent_is_consistent():
    """Adjoint gradient matches finite differences for an unsymmetric layup.

    An unsymmetric stacking sequence has a nonzero coupling matrix B, so the
    element tangent must include the membrane-bending coupling block. Without
    it the assembled stiffness is inconsistent with the internal force and the
    adjoint sensitivity (used by autograd) is wrong, which this test detects by
    comparing against a central finite difference.
    """
    nodes, elements = square_plate()
    gfrp = OrthotropicElasticityPlaneStress(
        E_1=40000.0, E_2=10000.0, nu_12=0.3, G_12=5000.0, G_13=5000.0, G_23=4000.0
    )

    def tip_objective(t_top):
        layup = Laminate(
            materials=[gfrp, gfrp],
            thicknesses=[torch.tensor(0.5), t_top],
            angles=[0.0, torch.pi / 2],
        )
        shell = Shell(nodes, elements, layup)
        u = cantilever_tip_displacement(shell, differentiable_parameters=t_top)
        return u.abs().sum()

    # Autograd gradient through the adjoint solve
    t = torch.tensor(0.5, requires_grad=True)
    obj = tip_objective(t)
    obj.backward()
    g_autograd = t.grad.item()

    # Central finite difference reference
    h = 1e-4
    with torch.no_grad():
        f_plus = tip_objective(torch.tensor(0.5 + h)).item()
        f_minus = tip_objective(torch.tensor(0.5 - h)).item()
    g_fd = (f_plus - f_minus) / (2.0 * h)

    assert abs(g_autograd - g_fd) <= 1e-4 * abs(g_fd)
