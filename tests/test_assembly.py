"""Assemblies of parts coupled by kinematic constraints."""

import pytest
import torch

from torchfem import (
    Assembly,
    Planar,
    ReferencePoint,
    ReferencePointHeat,
    Shell,
    Solid,
    SolidHeat,
)
from torchfem.elements import linear_to_quadratic
from torchfem.materials import (
    IsotropicConductivity3D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStrain,
    IsotropicElasticityPlaneStress,
)
from torchfem.mesh import cube_hexa, rect_quad, rect_tri

E, NU = 1000.0, 0.3
SOLID = IsotropicElasticity3D(E, NU)
PLANE = IsotropicElasticityPlaneStress(E, NU)

# Beam used by the solid-shell examples: slender enough that both formulations
# approach the Euler-Bernoulli tip deflection of 3.125.
L, B, T, P = 10.0, 1.0, 0.4, 0.05


def _block(z0: float, z1: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One hexahedron spanning the unit square between two heights."""
    nodes = torch.tensor(
        [[x, y, z] for z in (z0, z1) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )
    return nodes, torch.tensor([[0, 1, 3, 2, 4, 5, 7, 6]])


def _plate(x0: float, Lx: float, Nx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A flat triangulated plate in the z = 0 plane, starting at x0."""
    nodes, elements = rect_tri(Nx, 3, Lx, B)
    nodes = nodes + torch.tensor([x0, 0.0])
    return torch.cat([nodes, torch.zeros(len(nodes), 1)], dim=1), elements


def _beam_solid(Lx: float, Nx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A quadratic hexahedral beam centered on the z = 0 mid-surface."""
    nodes, elements = linear_to_quadratic(*cube_hexa(Nx, 3, 3, Lx, B, T))
    return nodes - torch.tensor([0.0, 0.0, T / 2]), elements


def test_coupling_reproduces_a_monolithic_solid():
    """Two blocks tied at their interface behave as one two-element bar."""
    nodes = torch.tensor(
        [[x, y, z] for z in (0.0, 1.0, 2.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )
    elements = torch.tensor([[0, 1, 3, 2, 4, 5, 7, 6], [4, 5, 7, 6, 8, 9, 11, 10]])
    reference = Solid(nodes, elements, SOLID)
    reference.constraints[nodes[:, 2] == 0.0] = True
    reference.forces[nodes[:, 2] == 2.0, 2] = 25.0
    u_ref = reference.solve()[0]

    n_a, e_a = _block(0.0, 1.0)
    n_b, e_b = _block(1.0, 2.0)
    a, b = Solid(n_a, e_a, SOLID), Solid(n_b, e_b, SOLID)
    a.constraints[n_a[:, 2] == 0.0] = True
    b.forces[n_b[:, 2] == 2.0, 2] = 25.0
    assembly = Assembly([a, b])
    assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0)
    u, _, _, _, _ = assembly.solve()

    assert torch.allclose(u[0], u_ref[:8])
    assert torch.allclose(u[1], u_ref[4:])
    # The tie is exact, so the interface closes to the last bit
    assert torch.equal(u[1][n_b[:, 2] == 1.0], u[0][n_a[:, 2] == 1.0])


def test_coupling_reproduces_a_monolithic_shell():
    """A shell split in two and tied matches the unsplit shell, rotations included."""
    nodes, elements = _plate(0.0, L, 21)
    reference = Shell(nodes, elements, PLANE, thickness=T)
    reference.constraints[nodes[:, 0] == 0.0] = True
    reference.forces[nodes[:, 0] == L, 2] = P / (nodes[:, 0] == L).sum()
    u_ref = reference.solve()[0]

    n_a, e_a = _plate(0.0, L / 2, 11)
    n_b, e_b = _plate(L / 2, L / 2, 11)
    a = Shell(n_a, e_a, PLANE, thickness=T)
    b = Shell(n_b, e_b, PLANE, thickness=T)
    a.constraints[n_a[:, 0] == 0.0] = True
    b.forces[n_b[:, 0] == L, 2] = P / (n_b[:, 0] == L).sum()
    assembly = Assembly([a, b])
    assembly.coupling(b, n_b[:, 0] == L / 2, a, n_a[:, 0] == L / 2)
    u, _, _, _, _ = assembly.solve()

    assert torch.allclose(u[0], u_ref[nodes[:, 0] <= L / 2])
    assert torch.allclose(u[1], u_ref[nodes[:, 0] >= L / 2])


def test_coupling_couples_a_solid_to_a_shell():
    """A beam built half solid, half shell matches the all-solid beam."""
    nodes, elements = _beam_solid(L, 21)
    reference = Solid(nodes, elements, SOLID)
    reference.constraints[nodes[:, 0] == 0.0] = True
    reference.forces[nodes[:, 0] == L, 2] = P / (nodes[:, 0] == L).sum()
    tip_ref = reference.solve()[0][nodes[:, 0] == L][:, 2].mean()

    n_s, e_s = _beam_solid(L / 2, 11)
    solid = Solid(n_s, e_s, SOLID)
    solid.constraints[n_s[:, 0] == 0.0] = True
    n_h, e_h = _plate(L / 2, L / 2, 11)
    shell = Shell(n_h, e_h, PLANE, thickness=T)
    shell.forces[n_h[:, 0] == L, 2] = P / (n_h[:, 0] == L).sum()

    assembly = Assembly([solid, shell])
    # The solid nodes sit t/2 off the mid-surface, so the shell rotation carries
    # them: they are the secondary side.
    assembly.coupling(solid, n_s[:, 0] == L / 2, shell, n_h[:, 0] == L / 2)
    u, _, _, _, _ = assembly.solve()
    tip = u[1][n_h[:, 0] == L][:, 2].mean()

    assert tip == pytest.approx(tip_ref, rel=0.02)
    assert tip == pytest.approx(P * L**3 / (3 * E * B * T**3 / 12), rel=0.02)


def test_coupling_enforces_the_rigid_relation():
    """Nodes coupled to a reference point follow its rigid-body motion exactly."""
    nodes, elements = cube_hexa(3, 3, 3)
    solid = Solid(nodes, elements, SOLID)
    solid.constraints[nodes[:, 2] == 0.0] = True
    point = ReferencePoint([0.5, 0.5, 2.0])
    point.forces[0, 3] = 50.0

    assembly = Assembly([solid, point])
    top = nodes[:, 2] == nodes[:, 2].max()
    assembly.coupling(solid, top, point)
    u, f, _, _, _ = assembly.solve()

    u_p, theta = u[1][0, :3], u[1][0, 3:]
    expected = u_p + torch.cross(
        theta.expand(int(top.sum()), 3), nodes[top] - point.nodes[0], dim=-1
    )
    assert torch.allclose(u[0][top], expected)
    # The reference point carries no stiffness, so its force is what the
    # coupling transmits, balancing the applied moment.
    assert f[1][0, 3] == pytest.approx(50.0)


def test_coupling_a_whole_part_moves_it_without_stress():
    """A part fully coupled to a translating reference point stays unstrained."""
    nodes, elements = cube_hexa(3, 3, 3)
    solid = Solid(nodes, elements, SOLID)
    point = ReferencePoint([0.5, 0.5, 0.5])
    point.constraints[0, :] = True
    point.displacements[0, :3] = torch.tensor([0.3, -0.2, 0.7])

    assembly = Assembly([solid, point])
    assembly.coupling(solid, torch.ones(len(nodes), dtype=torch.bool), point)
    u, _, flux, _, _ = assembly.solve()

    assert torch.allclose(u[0], torch.tensor([0.3, -0.2, 0.7]))
    assert torch.allclose(flux[0], torch.zeros_like(flux[0]), atol=1e-10)


def test_coupling_a_subset_of_dofs_leaves_the_rest_free():
    """`dofs=[2]` drives u_z only, and leaves the in-plane motion alone."""
    nodes, elements = cube_hexa(3, 3, 3)
    top = nodes[:, 2] == nodes[:, 2].max()
    solid = Solid(nodes, elements, SOLID)
    solid.constraints[nodes[:, 2] == 0.0] = True
    point = ReferencePoint([0.5, 0.5, 2.0])
    point.constraints[0, :] = True
    point.displacements[0, 2] = 0.1

    assembly = Assembly([solid, point])
    assembly.coupling(solid, top, point, dofs=[2])
    u, _, _, _, _ = assembly.solve()

    # Every coupled node follows u_z exactly
    assert torch.allclose(u[0][top][:, 2], torch.full((int(top.sum()),), 0.1))
    # while contracting freely in plane, which a full coupling would forbid
    assert u[0][top][:, 0].max() - u[0][top][:, 0].min() > 1e-3


def test_gradients_match_finite_differences():
    """The adjoint differentiates through the constrained solve."""
    n_a, e_a = _plate(0.0, L / 2, 5)
    n_b, e_b = _plate(L / 2, L / 2, 5)

    def work(thickness: torch.Tensor) -> torch.Tensor:
        a = Shell(n_a, e_a, PLANE, thickness=thickness)
        b = Shell(n_b, e_b, PLANE, thickness=T)
        a.constraints[n_a[:, 0] == 0.0] = True
        b.forces[n_b[:, 0] == L, 2] = P / (n_b[:, 0] == L).sum()
        assembly = Assembly([a, b])
        assembly.coupling(b, n_b[:, 0] == L / 2, a, n_a[:, 0] == L / 2)
        u, _, _, _, _ = assembly.solve(differentiable_parameters=thickness)
        return torch.inner(b.forces.ravel(), u[1].ravel())

    thickness = torch.full((len(e_a),), T, requires_grad=True)
    gradient = torch.autograd.grad(work(thickness), thickness)[0]

    h = 1e-4
    expected = torch.zeros_like(gradient)
    for i in range(len(expected)):
        plus, minus = torch.full_like(expected, T), torch.full_like(expected, T)
        plus[i], minus[i] = T + h, T - h
        expected[i] = (work(plus) - work(minus)).detach() / (2 * h)

    assert torch.allclose(gradient, expected, rtol=1e-5, atol=1e-8)


def test_incremental_loading_matches_a_single_step():
    """A linear assembly reaches the same state however the load is applied."""
    n_a, e_a = _block(0.0, 1.0)
    n_b, e_b = _block(1.0, 2.0)

    def solve(increments):
        a, b = Solid(n_a, e_a, SOLID), Solid(n_b, e_b, SOLID)
        a.constraints[n_a[:, 2] == 0.0] = True
        b.forces[n_b[:, 2] == 2.0, 2] = 25.0
        assembly = Assembly([a, b])
        assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0)
        return assembly.solve(increments=increments)[0]

    single = solve(None)
    stepped = solve(torch.linspace(0.0, 1.0, 5))
    assert torch.allclose(single[0], stepped[0])
    assert torch.allclose(single[1], stepped[1])


def test_an_assembly_without_constraints_solves_its_parts_independently():
    """Parts that are never coupled keep the solution they have on their own."""
    n_a, e_a = _block(0.0, 1.0)
    a = Solid(n_a, e_a, SOLID)
    a.constraints[n_a[:, 2] == 0.0] = True
    a.forces[n_a[:, 2] == 1.0, 2] = 25.0
    alone = a.solve()[0]

    b = Solid(n_a, e_a, SOLID)
    b.constraints[n_a[:, 2] == 0.0] = True
    b.forces[n_a[:, 2] == 1.0, 2] = 25.0
    u, _, _, _, _ = Assembly([b]).solve()
    assert torch.allclose(u[0], alone)


def _two_blocks() -> tuple[Solid, Solid, torch.Tensor, torch.Tensor]:
    n_a, e_a = _block(0.0, 1.0)
    n_b, e_b = _block(1.0, 2.0)
    return Solid(n_a, e_a, SOLID), Solid(n_b, e_b, SOLID), n_a, n_b


def test_return_intermediate_gives_every_increment():
    """Every requested increment is returned, ending at the final state."""
    a, b, n_a, n_b = _two_blocks()
    a.constraints[n_a[:, 2] == 0.0] = True
    b.forces[n_b[:, 2] == 2.0, 2] = 25.0
    assembly = Assembly([a, b])
    assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0)
    increments = torch.linspace(0.0, 1.0, 4)

    every = assembly.solve(increments=increments, return_intermediate=True)[0]
    final = assembly.solve(increments=increments)[0]

    assert [x.shape[0] for x in every] == [len(increments)] * 2
    assert all(torch.allclose(x[-1], y) for x, y in zip(every, final))
    # A linear assembly follows the load factor
    tip = every[1][:, n_b[:, 2] == 2.0, 2].mean(dim=1)
    assert torch.allclose(tip / tip[-1], increments)


def test_coupling_needs_parts_of_this_assembly():
    a, b, n_a, n_b = _two_blocks()
    assembly = Assembly([a])
    with pytest.raises(ValueError, match="must belong to this assembly"):
        assembly.coupling(a, n_a[:, 2] == 1.0, b)


def test_a_dof_cannot_be_eliminated_twice():
    a, b, n_a, n_b = _two_blocks()
    assembly = Assembly([a, b])
    assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0)
    assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0)
    with pytest.raises(ValueError, match="more than one constraint"):
        assembly.solve()


def test_a_primary_dof_cannot_also_be_eliminated():
    a, b, n_a, n_b = _two_blocks()
    point = ReferencePoint([0.5, 0.5, 1.0])
    assembly = Assembly([a, b, point])
    assembly.coupling(a, n_a[:, 2] == 1.0, point)
    assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0)
    with pytest.raises(ValueError, match="both eliminated and used as a primary"):
        assembly.solve()


def test_a_constrained_dof_cannot_be_eliminated():
    a, b, n_a, n_b = _two_blocks()
    b.constraints[n_b[:, 2] == 1.0] = True
    assembly = Assembly([a, b])
    assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0)
    with pytest.raises(ValueError, match="constrained DOF is eliminated"):
        assembly.solve()


def test_dofs_are_checked_against_the_secondary_part():
    a, b, n_a, n_b = _two_blocks()
    assembly = Assembly([a, b])
    with pytest.raises(ValueError, match=r"indices in \[0, 3\)"):
        assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0, dofs=[5])


def test_rotations_need_a_primary_that_has_them():
    nodes, elements = _plate(0.0, L, 5)
    shell = Shell(nodes, elements, PLANE, thickness=T)
    a, _, n_a, _ = _two_blocks()
    assembly = Assembly([a, shell])
    with pytest.raises(ValueError, match="primary part that has none"):
        assembly.coupling(shell, nodes[:, 0] == 0.0, a, n_a[:, 2] == 1.0, dofs=[3])


def test_coupling_pairs_each_node_with_the_nearest_primary():
    """Each secondary node picks up the motion of the primary node it sits on."""
    n_a, e_a = _block(0.0, 1.0)
    n_b, e_b = _block(1.0, 2.0)
    a, b = Solid(n_a, e_a, SOLID), Solid(n_b, e_b, SOLID)

    # Drive A entirely, with a motion that differs from node to node
    a.constraints[:] = True
    a.displacements[:, 2] = 0.1 * n_a[:, 0] + 0.2 * n_a[:, 1]

    assembly = Assembly([a, b])
    assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0)
    u, _, _, _, _ = assembly.solve()

    # The faces are numbered alike, so a correct pairing copies node for node
    interface, top = n_b[:, 2] == 1.0, n_a[:, 2] == 1.0
    assert u[1][interface][:, 2].std() > 0.0  # the four values really do differ
    assert torch.allclose(u[1][interface], u[0][top])


def _bar(z0: float, z1: float) -> SolidHeat:
    """One conducting hexahedron between two heights."""
    nodes, elements = _block(z0, z1)
    return SolidHeat(nodes, elements, IsotropicConductivity3D(1.0))


def test_coupling_ties_two_conducting_meshes():
    """Two thermal meshes tied at an interface behave as one conducting bar."""
    nodes = torch.tensor(
        [[x, y, z] for z in (0.0, 1.0, 2.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )
    elements = torch.tensor([[0, 1, 3, 2, 4, 5, 7, 6], [4, 5, 7, 6, 8, 9, 11, 10]])
    reference = SolidHeat(nodes, elements, IsotropicConductivity3D(1.0))
    reference.constraints[nodes[:, 2] == 0.0] = True
    reference.heat_flux[nodes[:, 2] == 2.0, 0] = 0.25
    T_ref = reference.solve()[0]

    a, b = _bar(0.0, 1.0), _bar(1.0, 2.0)
    n_a, n_b = a.nodes, b.nodes
    a.constraints[n_a[:, 2] == 0.0] = True
    b.heat_flux[n_b[:, 2] == 2.0, 0] = 0.25
    assembly = Assembly([a, b])
    assembly.coupling(b, n_b[:, 2] == 1.0, a, n_a[:, 2] == 1.0)
    T, _, _, _, _ = assembly.solve()

    assert torch.allclose(T[0], T_ref[:8])
    assert torch.allclose(T[1], T_ref[4:])
    # A primary without rotations leaves a pure link, so the interface is smooth
    assert torch.equal(T[1][n_b[:, 2] == 1.0], T[0][n_a[:, 2] == 1.0])


def test_a_thermal_point_holds_a_surface_isothermal():
    """A heat source at the point spreads over the whole coupled surface."""
    nodes, elements = cube_hexa(4, 4, 4)
    solid = SolidHeat(nodes, elements, IsotropicConductivity3D(1.0))
    solid.constraints[nodes[:, 2] == 0.0] = True
    top = nodes[:, 2] == 1.0
    point = ReferencePointHeat([0.5, 0.5, 1.5])
    point.heat_flux[0, 0] = 4.0

    assembly = Assembly([solid, point])
    assembly.coupling(solid, top, point)
    T, q, _, _, _ = assembly.solve()

    # Unit conductivity over a unit cube, so the rise is the heat itself
    assert torch.allclose(T[0][top], torch.full_like(T[0][top], 4.0))
    assert T[1][0, 0] == pytest.approx(4.0)
    assert q[1][0, 0] == pytest.approx(4.0)


def test_an_assembly_solves_one_physics_at_a_time():
    nodes, elements = cube_hexa(2, 2, 2)
    solid = Solid(nodes, elements, SOLID)
    heat = SolidHeat(nodes, elements, IsotropicConductivity3D(1.0))
    with pytest.raises(ValueError, match="mechanical or thermal"):
        Assembly([solid, heat])
    with pytest.raises(ValueError, match="mechanical or thermal"):
        Assembly([heat, ReferencePoint([0.0, 0.0, 1.0])])


def test_parts_share_one_spatial_dimension():
    planar = Planar(*rect_quad(3, 3), IsotropicElasticityPlaneStrain(E, NU))
    solid = Solid(*cube_hexa(2, 2, 2), SOLID)
    with pytest.raises(ValueError, match="one spatial dimension"):
        Assembly([planar, solid])
    # A point of the wrong dimension is caught by the same check
    with pytest.raises(ValueError, match="one spatial dimension"):
        Assembly([planar, ReferencePoint([0.0, 0.0, 0.0])])


def test_coupling_ties_two_planar_meshes():
    """A planar cantilever split in two and tied matches the unsplit one."""
    nodes, elements = rect_quad(5, 3, 2.0, 1.0)
    reference = Planar(nodes, elements, PLANE)
    reference.constraints[nodes[:, 0] == 0.0] = True
    reference.forces[nodes[:, 0] == 2.0, 1] = -1.0
    u_ref = reference.solve()[0]

    n_a, e_a = rect_quad(3, 3, 1.0, 1.0)
    n_b, e_b = rect_quad(3, 3, 1.0, 1.0)
    n_b = n_b + torch.tensor([1.0, 0.0])
    a, b = Planar(n_a, e_a, PLANE), Planar(n_b, e_b, PLANE)
    a.constraints[n_a[:, 0] == 0.0] = True
    b.forces[n_b[:, 0] == 2.0, 1] = -1.0
    assembly = Assembly([a, b])
    assembly.coupling(b, n_b[:, 0] == 1.0, a, n_a[:, 0] == 1.0)
    u, _, _, _, _ = assembly.solve()

    assert torch.allclose(u[0], u_ref[nodes[:, 0] <= 1.0])
    assert torch.allclose(u[1], u_ref[nodes[:, 0] >= 1.0])


def test_a_planar_point_carries_one_rotation():
    """In 2D a reference point has two translations and one rotation about z."""
    nodes, elements = rect_quad(5, 5)
    plate = Planar(nodes, elements, PLANE)
    plate.constraints[nodes[:, 1] == 0.0] = True
    point = ReferencePoint([0.5, 1.5])
    assert point.n_dofs == 3
    point.forces[0, 2] = 20.0  # moment about z

    assembly = Assembly([plate, point])
    top = nodes[:, 1] == 1.0
    assembly.coupling(plate, top, point)
    u, f, _, _, _ = assembly.solve()

    u_p, theta = u[1][0, :2], u[1][0, 2]
    r = nodes[top] - point.nodes[0]
    expected = u_p + theta * torch.stack([-r[:, 1], r[:, 0]], dim=1)
    assert torch.allclose(u[0][top], expected)
    assert f[1][0, 2] == pytest.approx(20.0)


def test_a_part_cannot_appear_twice():
    a, _, _, _ = _two_blocks()
    with pytest.raises(ValueError, match="only once"):
        Assembly([a, a])
