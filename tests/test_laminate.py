import torch

from torchfem import Laminate, Shell
from torchfem.materials import (
    IsotropicElasticityPlaneStress,
    IsotropicPlasticityPlaneStress,
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


def test_symmetric_laminate_decouples_membrane_and_bending():
    """Symmetric stacks have no membrane-bending coupling; unsymmetric ones do.

    Validated through the assembled shell response rather than a precomputed
    coupling matrix: an in-plane stretch produces essentially no out-of-plane
    deflection for a symmetric laminate, but a clearly nonzero one for an
    unsymmetric laminate (nonzero coupling).
    """
    nodes, elements = square_plate()
    gfrp = OrthotropicElasticityPlaneStress(
        E_1=40000.0, E_2=10000.0, nu_12=0.3, G_12=5000.0, G_13=5000.0, G_23=4000.0
    )

    def out_of_plane_under_stretch(angles, thicknesses):
        layup = Laminate(
            materials=[gfrp] * len(angles),
            thicknesses=thicknesses,
            angles=angles,
        )
        plate = Shell(nodes, elements, layup)
        plate.constraints[[0, 3]] = True  # clamp x = 0 edge (all 6 dofs)
        plate.constraints[[1, 2], 0] = True  # prescribe in-plane u_x on x = 1 edge
        plate.displacements[[1, 2], 0] = 0.01
        u, _, _, _, _ = plate.solve()
        return u[:, 2].abs().max()

    uz_sym = out_of_plane_under_stretch(
        [0.0, torch.pi / 2, torch.pi / 2, 0.0], [0.25, 0.25, 0.25, 0.25]
    )
    uz_unsym = out_of_plane_under_stretch([0.0, torch.pi / 2], [0.5, 0.5])

    assert uz_sym < 1e-9
    assert uz_unsym > 1e-4


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


def test_metal_ply_plasticizes():
    """A fiber-metal laminate yields in its metal plies under in-plane stretch.

    The elastoplastic aluminium layers accumulate equivalent plastic strain and
    soften the membrane response relative to a purely elastic laminate. This
    exercises state-bearing layers integrated through the thickness during the
    analysis (no precomputed ABD).
    """
    nodes, elements = square_plate()

    sigma_y, hardening = 200.0, 1000.0
    alu = IsotropicPlasticityPlaneStress(
        E=70000.0,
        nu=0.33,
        sigma_f=lambda q: sigma_y + hardening * q,
        sigma_f_prime=lambda q: hardening * torch.ones_like(q),
    )
    alu_elastic = IsotropicElasticityPlaneStress(E=70000.0, nu=0.33)
    gfrp = OrthotropicElasticityPlaneStress(
        E_1=54000.0, E_2=9400.0, nu_12=0.33, G_12=5500.0, G_13=5500.0, G_23=3000.0
    )

    def stretch(metal):
        # Al / 0 deg GFRP / Al fiber-metal laminate
        layup = Laminate(
            materials=[metal, gfrp, metal],
            thicknesses=[0.4, 0.25, 0.4],
            angles=[0.0, 0.0, 0.0],
        )
        plate = Shell(nodes, elements, layup)
        # Clamp x = 0 edge, prescribe in-plane stretch on the x = 1 edge
        plate.constraints[[0, 3]] = True
        plate.constraints[[1, 2], 0] = True
        plate.displacements[[1, 2], 0] = 0.02  # ~2% strain, well past yield
        u, f, _, _, state = plate.solve(
            increments=torch.linspace(0.0, 1.0, 6),
            aggregate_integration_points=False,
        )
        reaction = f[[1, 2], 0].sum()
        return u, state, reaction

    u, state, reaction_plastic = stretch(alu)
    _, _, reaction_elastic = stretch(alu_elastic)

    assert torch.all(torch.isfinite(u))
    # Equivalent plastic strain accumulated in the metal plies
    assert state.abs().max() > 0.0
    # Plastic laminate carries less load than the elastic one (yield softening)
    assert reaction_plastic < reaction_elastic


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


def _anisotropic_shell(nodes, elements, orientation=None):
    mat = OrthotropicElasticityPlaneStress(
        E_1=130000.0, E_2=8000.0, nu_12=0.3, G_12=4000.0, G_13=4000.0, G_23=3000.0
    )
    return Shell(
        nodes,
        elements,
        mat,
        thickness=1.0,
        transverse_G=[4000.0, 3000.0],
        orientation=orientation,
    )


def test_shell_orientation_node_ordering_invariant():
    """A global material orientation makes anisotropic shell results independent
    of the per-element node ordering.

    The element material frame is built from the global ``orientation`` projected
    onto each element, not from the first edge, so cyclically permuting the nodes
    of every element must not change the solution.
    """
    nodes, elements = square_plate()
    u0 = cantilever_tip_displacement(_anisotropic_shell(nodes, elements))
    u1 = cantilever_tip_displacement(_anisotropic_shell(nodes, elements[:, [1, 2, 0]]))
    assert torch.allclose(u0, u1, atol=1e-10)


def test_shell_orientation_rotates_response():
    """The orientation argument rotates the anisotropic stiffness, and passing
    the stiff axis along x vs y changes the bending response."""
    nodes, elements = square_plate()
    u_x = cantilever_tip_displacement(
        _anisotropic_shell(nodes, elements, orientation=torch.tensor([1.0, 0.0, 0.0]))
    )
    u_y = cantilever_tip_displacement(
        _anisotropic_shell(nodes, elements, orientation=torch.tensor([0.0, 1.0, 0.0]))
    )
    assert not torch.allclose(u_x, u_y, atol=1e-6)
    # A per-element orientation tensor is also accepted and matches the shared one
    per_elem = torch.tensor([1.0, 0.0, 0.0]).expand(len(elements), 3)
    u_pe = cantilever_tip_displacement(
        _anisotropic_shell(nodes, elements, orientation=per_elem)
    )
    assert torch.allclose(u_x, u_pe, atol=1e-12)


def test_laminate_is_section_not_material():
    """A laminate is stored as the shell's section; self.material stays a real
    (pointwise) Material and is None for a layered shell."""
    nodes, elements = square_plate()
    gfrp = OrthotropicElasticityPlaneStress(
        E_1=40000.0, E_2=10000.0, nu_12=0.3, G_12=5000.0, G_13=5000.0, G_23=4000.0
    )
    layered = Shell(nodes, elements, Laminate([gfrp], [1.0], [0.0]))
    assert layered.material is None
    assert isinstance(layered.section, Laminate)
    assert layered.n_state == 0  # provided via the section

    homogeneous = Shell(
        nodes,
        elements,
        IsotropicElasticityPlaneStress(E=70000.0, nu=0.3),
        thickness=1.0,
    )
    assert homogeneous.section is None
    assert homogeneous.material is not None


def _gfrp():
    return OrthotropicElasticityPlaneStress(
        E_1=40000.0, E_2=10000.0, nu_12=0.3, G_12=5000.0, G_13=5000.0, G_23=4000.0
    )


def _uz_under_inplane_stretch(layup):
    """Tip out-of-plane deflection of a clamped plate under in-plane stretch."""
    nodes, elements = square_plate()
    plate = Shell(nodes, elements, layup)
    plate.constraints[[0, 3]] = True  # clamp x = 0 edge (all 6 dofs)
    plate.constraints[[1, 2], 0] = True  # prescribe in-plane u_x on x = 1 edge
    plate.displacements[[1, 2], 0] = 0.01
    u, _, _, _, _ = plate.solve()
    return u[:, 2]


def test_offset_induces_membrane_bending_coupling():
    """An offset shifts the stations, so an in-plane stretch bends the plate.

    A single-layer laminate has no membrane-bending coupling when centered, but
    offsetting the reference surface to the top vs. the bottom produces
    out-of-plane deflections of equal magnitude and opposite sign.
    """
    uz0 = _uz_under_inplane_stretch(Laminate([_gfrp()], [1.0], [0.0], offset=0.0))
    uz_top = _uz_under_inplane_stretch(Laminate([_gfrp()], [1.0], [0.0], offset=0.5))
    uz_bot = _uz_under_inplane_stretch(Laminate([_gfrp()], [1.0], [0.0], offset=-0.5))

    assert uz0.abs().max() < 1e-9
    assert uz_top.abs().max() > 1e-5
    assert torch.allclose(uz_top, -uz_bot, atol=1e-9)


def test_offset_string_aliases_match_floats():
    """The "mid"/"top"/"bottom" aliases match their fractional equivalents."""
    for name, frac in [("mid", 0.0), ("top", 0.5), ("bottom", -0.5)]:
        u_name = _uz_under_inplane_stretch(
            Laminate([_gfrp()], [1.0], [0.0], offset=name)
        )
        u_frac = _uz_under_inplane_stretch(
            Laminate([_gfrp()], [1.0], [0.0], offset=frac)
        )
        assert torch.allclose(u_name, u_frac, atol=1e-12)


def test_symmetric_expands_to_full_stack():
    """`symmetric=True` mirrors the half-stack into the full laminate."""
    gfrp = _gfrp()
    half = Laminate([gfrp, gfrp], [0.25, 0.25], [0.0, torch.pi / 2], symmetric=True)
    full = Laminate([gfrp] * 4, [0.25] * 4, [0.0, torch.pi / 2, torch.pi / 2, 0.0])

    assert half.n_layers == 4
    u_half = cantilever_tip_displacement(Shell(*square_plate(), half))
    u_full = cantilever_tip_displacement(Shell(*square_plate(), full))
    assert torch.allclose(u_half, u_full, atol=1e-12)


def test_symmetric_accepts_tensor_inputs():
    """`symmetric=True` mirrors tensor thicknesses and angles, not just lists."""
    gfrp = _gfrp()
    lam = Laminate(
        [gfrp, gfrp],
        torch.tensor([0.3, 0.125]),
        torch.tensor([0.0, torch.pi / 2]),
        symmetric=True,
    )

    assert lam.n_layers == 4
    assert [round(float(torch.rad2deg(a))) for a in lam.angles] == [0, 90, 90, 0]
    assert [float(t) for t in lam.thicknesses] == [0.3, 0.125, 0.125, 0.3]


def test_symmetric_offset_recouples():
    """A symmetric stack is decoupled; offsetting it restores the coupling."""
    gfrp = _gfrp()
    centered = Laminate([gfrp, gfrp], [0.25, 0.25], [0.0, torch.pi / 2], symmetric=True)
    offset = Laminate(
        [gfrp, gfrp], [0.25, 0.25], [0.0, torch.pi / 2], symmetric=True, offset=0.3
    )

    assert _uz_under_inplane_stretch(centered).abs().max() < 1e-9
    assert _uz_under_inplane_stretch(offset).abs().max() > 1e-5
