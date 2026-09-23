"""Axisymmetric models, checked against closed-form solutions of revolved bodies."""

import math

import pytest
import torch

from torchfem import Axisymmetric, AxisymmetricHeat
from torchfem.elements import linear_to_quadratic
from torchfem.materials import (
    Hyperelastic3D,
    IsotropicConductivity2D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStress,
)
from torchfem.mesh import rect_quad, rect_tri

# Thick-walled cylinder of inner radius A, outer radius B and height H
E, NU, A, B, H, P = 210000.0, 0.3, 10.0, 20.0, 2.0, 100.0

# Lame constant of the internally pressurized cylinder
C = A**2 * P / (B**2 - A**2)

ETYPES = ["Tria1", "Tria2", "Quad1", "Quad2"]


def _cylinder(etype: str = "Quad1", nr: int = 20) -> Axisymmetric:
    """Pressurized cylinder in plane strain, i.e. with both ends held axially."""
    gen = rect_tri if etype.startswith("Tria") else rect_quad
    nodes, elements = gen(nr + 1, 3, B - A, H)
    if etype.endswith("2"):
        nodes, elements = linear_to_quadratic(nodes, elements)
    nodes[:, 0] += A
    model = Axisymmetric(nodes, elements, IsotropicElasticity3D(E, NU))
    model.constraints[(nodes[:, 1] == 0.0) | (nodes[:, 1] == H), 1] = True
    model.forces = model.integrate_line_load(nodes[:, 0] == A, torch.tensor(-P))
    return model


def _lame_u_r(r: torch.Tensor) -> torch.Tensor:
    return (1 + NU) / E * C * ((1 - 2 * NU) * r + B**2 / r)


class TestLame:
    """A pressurized thick-walled cylinder against the Lame solution."""

    @pytest.mark.parametrize("etype", ETYPES)
    def test_radial_displacement(self, etype):
        model = _cylinder(etype)
        u, _, _, _, _ = model.solve()
        exact = _lame_u_r(model.nodes[:, 0])
        assert torch.allclose(u[:, 0], exact, rtol=1e-3)

    @pytest.mark.parametrize("etype", ETYPES)
    def test_axial_displacement_vanishes(self, etype):
        # Exact to machine precision on a structured mesh. An unstructured
        # quadratic triangulation leaves a residual, which stays below the error
        # of the radial solution and converges away with it.
        model = _cylinder(etype)
        u, _, _, _, _ = model.solve()
        radial_error = (u[:, 0] - _lame_u_r(model.nodes[:, 0])).abs().max()
        assert u[:, 1].abs().max() < radial_error

    def test_radial_and_hoop_stress(self):
        model = _cylinder(nr=40)
        _, _, sigma, _, _ = model.solve()
        r = model.nodes[model.elements, 0].mean(dim=1)
        assert torch.allclose(sigma[:, 0, 0], C * (1 - B**2 / r**2), atol=1e-1)
        assert torch.allclose(sigma[:, 2, 2], C * (1 + B**2 / r**2), atol=1e-1)
        # No shear in a cylinder loaded this way
        assert torch.allclose(sigma[:, 0, 1], torch.zeros_like(r), atol=1e-9)

    def test_convergence_is_second_order(self):
        errors = []
        for nr in (5, 10, 20, 40):
            model = _cylinder(nr=nr)
            u, _, _, _, _ = model.solve()
            errors.append(float((u[:, 0] - _lame_u_r(model.nodes[:, 0])).abs().max()))
        ratios = [a / b for a, b in zip(errors[:-1], errors[1:])]
        assert all(r > 3.5 for r in ratios), ratios

    def test_pressure_carries_no_net_axial_force(self):
        model = _cylinder()
        _, f, _, _, _ = model.solve()
        assert float(f[:, 1].sum()) == pytest.approx(0.0, abs=1e-9)


class TestMeasure:
    """The revolution enters every integrated measure."""

    @pytest.mark.parametrize("etype", ETYPES)
    def test_integrate_field_returns_the_revolved_volume(self, etype):
        model = _cylinder(etype)
        volume = math.pi * (B**2 - A**2) * H
        assert float(model.integrate_field().sum()) == pytest.approx(volume)

    def test_body_load_is_the_revolved_weight(self):
        model = _cylinder()
        f = model.integrate_body_load(torch.tensor([0.0, -1.0]))
        volume = math.pi * (B**2 - A**2) * H
        assert float(f[:, 1].sum()) == pytest.approx(-volume)

    def test_line_load_is_the_revolved_surface(self):
        model = _cylinder()
        # A unit traction along z on the inner wall, whose area is 2 pi A H
        mask = model.nodes[:, 0] == A
        f = model.integrate_line_load(mask, torch.tensor([0.0, 1.0]))
        assert float(f[:, 1].sum()) == pytest.approx(2 * math.pi * A * H)

    def test_char_lengths_measure_the_cross_section(self):
        model = _cylinder(nr=20)
        # A structured mesh of 20 x 2 cells over (B - A) x H
        area = (B - A) / 20 * H / 2
        assert torch.allclose(model.char_lengths, torch.full((40,), area**0.5))


def test_homogeneous_stretch():
    """A stretch the hoop kinematics must reproduce exactly, in finite strain."""
    params = torch.tensor([400.0, 1000.0])
    lam, mu = 1.25, 0.8

    def neo_hooke(F, p):
        C = F.transpose(-1, -2) @ F
        lnJ = 0.5 * torch.logdet(C)
        trC = torch.einsum("...ii->...", C)
        return 0.5 * p[0] * (trC - 3) - p[0] * lnJ + 0.5 * p[1] * lnJ**2

    nodes, elements = rect_quad(4, 3, 2.0, 1.0)
    nodes[:, 0] += 1.0
    model = Axisymmetric(nodes, elements, Hyperelastic3D(neo_hooke, params))
    # The stretch is prescribed on the whole boundary of the cross section
    lo, hi = nodes.aminmax(dim=0)
    edge = ((nodes == lo) | (nodes == hi)).any(dim=1)
    model.constraints[edge] = True
    model.displacements[edge, 0] = (lam - 1.0) * nodes[edge, 0]
    model.displacements[edge, 1] = (mu - 1.0) * nodes[edge, 1]
    _, _, sigma, F, _ = model.solve(increments=torch.linspace(0, 1, 5))

    # The deformation gradient is the imposed stretch in every element
    exact = torch.diag(torch.tensor([lam, mu, lam]))
    assert torch.allclose(F, exact.expand_as(F), atol=1e-12)

    # ...and the stress is the analytic response of the material at it
    Fe = exact.clone().requires_grad_(True)
    (P,) = torch.autograd.grad(neo_hooke(Fe, params), Fe)
    cauchy = P @ Fe.T.detach() / torch.linalg.det(Fe).detach()
    assert torch.allclose(sigma, cauchy.expand_as(sigma), atol=1e-9)


class TestHeat:
    """Steady radial conduction through a cylindrical wall."""

    def _wall(self, t_inner: float = 100.0, k: float = 10.0) -> AxisymmetricHeat:
        nodes, elements = rect_quad(41, 3, B - A, H)
        nodes[:, 0] += A
        model = AxisymmetricHeat(nodes, elements, IsotropicConductivity2D(k))
        model.constraints[(nodes[:, 0] == A) | (nodes[:, 0] == B), 0] = True
        model.temperatures[nodes[:, 0] == A, 0] = t_inner
        return model

    def test_logarithmic_profile(self):
        model = self._wall()
        T, _, _, _, _ = model.solve()
        r = model.nodes[:, 0]
        exact = 100.0 * (1 - torch.log(r / A) / math.log(B / A))
        assert torch.allclose(T[:, 0], exact, atol=1e-2)

    def test_heat_flow_through_the_wall(self):
        model = self._wall()
        _, q, _, _, _ = model.solve()
        exact = 2 * math.pi * 10.0 * H * 100.0 / math.log(B / A)
        assert float(q[model.nodes[:, 0] == A, 0].sum()) == pytest.approx(
            exact, rel=1e-3
        )


class TestInterface:
    def test_rejects_a_plane_material(self):
        nodes, elements = rect_quad(3, 3)
        nodes[:, 0] += 1.0
        with pytest.raises(ValueError, match="needs a 3D MechanicsMaterial"):
            Axisymmetric(nodes, elements, IsotropicElasticityPlaneStress(E, NU))

    def test_takes_no_thickness(self):
        nodes, elements = rect_quad(3, 3)
        nodes[:, 0] += 1.0
        with pytest.raises(TypeError):
            Axisymmetric(nodes, elements, IsotropicElasticity3D(E, NU), 2.0)  # type: ignore[call-arg]

    def test_repr(self):
        assert "axisymmetric" in repr(_cylinder())

    def test_compliance_gradient(self):
        stiffness = torch.tensor(E, requires_grad=True)
        nodes, elements = rect_quad(11, 3, B - A, H)
        nodes[:, 0] += A
        model = Axisymmetric(nodes, elements, IsotropicElasticity3D(stiffness, NU))
        model.constraints[(nodes[:, 1] == 0.0) | (nodes[:, 1] == H), 1] = True
        model.forces = model.integrate_line_load(nodes[:, 0] == A, torch.tensor(-P))
        u, _, _, _, _ = model.solve(differentiable_parameters=stiffness)
        compliance = torch.dot(u.ravel(), model.forces.ravel())
        (grad,) = torch.autograd.grad(compliance, stiffness)
        # Compliance of a linear elastic problem scales as 1 / E
        assert float(grad) == pytest.approx(-float(compliance.detach()) / E, rel=1e-9)

    def test_solve_modes(self):
        model = _cylinder(nr=8)
        model.constraints[model.nodes[:, 0] == A, :] = True
        omega_sq, modes = model.solve_modes(3)
        assert torch.all(omega_sq > 0.0)
        assert modes.shape == (3, model.n_nod, 2)
