import pytest
import torch

from torchfem import Planar, PlanarHeat, Solid, SolidHeat
from torchfem.materials import (
    IsotropicConductivity2D,
    IsotropicConductivity3D,
    IsotropicDamage3D,
    IsotropicElasticityPlaneStress,
)
from torchfem.mesh import cube_hexa, rect_quad


def _planar() -> Planar:
    return Planar(*rect_quad(3, 3), IsotropicElasticityPlaneStress(1000.0, 0.3))


def _planar_heat() -> PlanarHeat:
    return PlanarHeat(*rect_quad(3, 3), IsotropicConductivity2D(kappa=400.0))


class TestMechanicsBoundaryConditions:
    @pytest.mark.parametrize("prop", ["forces", "displacements"])
    def test_assignment_round_trips(self, prop):
        model = _planar()
        value = torch.randn(model.n_nod, model.n_dof_per_node)
        setattr(model, prop, value)
        assert torch.equal(getattr(model, prop), value)

    @pytest.mark.parametrize("prop", ["forces", "displacements"])
    def test_rejects_wrong_shape(self, prop):
        model = _planar()
        with pytest.raises(ValueError, match="same shape as nodes"):
            setattr(model, prop, torch.zeros(model.n_nod, 3))

    @pytest.mark.parametrize("prop", ["forces", "displacements"])
    def test_rejects_non_floating_point(self, prop):
        model = _planar()
        with pytest.raises(TypeError, match="floating-point"):
            setattr(model, prop, torch.zeros(model.n_nod, 2, dtype=torch.int64))

    def test_ext_strain_round_trips(self):
        model = _planar()
        value = torch.randn(model.n_elem, model.n_dof_per_node, model.n_dim)
        model.ext_strain = value
        assert torch.equal(model.ext_strain, value)

    def test_ext_strain_rejects_wrong_shape(self):
        model = _planar()
        with pytest.raises(ValueError, match="same shape as strains"):
            model.ext_strain = torch.zeros(model.n_elem, 3, 3)

    def test_ext_strain_rejects_non_floating_point(self):
        model = _planar()
        with pytest.raises(TypeError, match="floating-point"):
            model.ext_strain = torch.zeros(model.n_elem, 2, 2, dtype=torch.int64)

    def test_constraints_round_trip(self):
        model = _planar()
        value = torch.zeros(model.n_nod, model.n_dof_per_node, dtype=torch.bool)
        value[0] = True
        model.constraints = value
        assert torch.equal(model.constraints, value)

    def test_constraints_reject_wrong_shape(self):
        model = _planar()
        with pytest.raises(ValueError, match="same shape as nodes"):
            model.constraints = torch.zeros(model.n_nod, 3, dtype=torch.bool)

    def test_constraints_reject_non_boolean(self):
        model = _planar()
        with pytest.raises(TypeError, match="boolean"):
            model.constraints = torch.zeros(model.n_nod, model.n_dof_per_node)


class TestShapeFunctions:
    def test_inverted_element_raises(self):
        """Reversing a quad's node order flips the Jacobian sign."""
        nodes, elements = rect_quad(2, 2)
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        model = Planar(nodes, elements.flip(-1), mat)
        with pytest.raises(ValueError, match="Negative Jacobian"):
            model.eval_shape_functions(model.etype.ipoints)


class TestHeatBoundaryConditions:
    @pytest.mark.parametrize("prop", ["heat_flux", "temperatures"])
    def test_assignment_round_trips(self, prop):
        model = _planar_heat()
        value = torch.randn(model.n_nod, 1)
        setattr(model, prop, value)
        assert torch.equal(getattr(model, prop), value)

    @pytest.mark.parametrize("prop", ["heat_flux", "temperatures"])
    def test_rejects_wrong_shape(self, prop):
        model = _planar_heat()
        with pytest.raises(ValueError, match="same shape as nodes"):
            setattr(model, prop, torch.zeros(model.n_nod, 2))

    @pytest.mark.parametrize("prop", ["heat_flux", "temperatures"])
    def test_rejects_non_floating_point(self, prop):
        model = _planar_heat()
        with pytest.raises(TypeError, match="floating-point"):
            setattr(model, prop, torch.zeros(model.n_nod, 1, dtype=torch.int64))


class TestHeatConductivityMatrix:
    def test_k0_is_symmetric_with_zero_row_sums(self):
        """A constant temperature field drives no heat flow, so every element
        conductivity matrix is singular with vanishing row sums."""
        model = _planar_heat()
        k = model.k0()
        assert k.shape == (model.n_elem, 4, 4)
        assert torch.allclose(k, k.transpose(-1, -2))
        assert torch.allclose(k.sum(-1), torch.zeros(model.n_elem, 4), atol=1e-10)

    def test_k0_scales_with_conductivity(self):
        k = _planar_heat().k0()
        doubled = PlanarHeat(*rect_quad(3, 3), IsotropicConductivity2D(800.0)).k0()
        assert torch.allclose(doubled, 2.0 * k)

    def test_solid_heat_k0_is_symmetric_with_zero_row_sums(self):
        model = SolidHeat(*cube_hexa(3, 3, 3), IsotropicConductivity3D(400.0))
        k = model.k0()
        assert k.shape == (model.n_elem, 8, 8)
        assert torch.allclose(k, k.transpose(-1, -2))
        assert torch.allclose(k.sum(-1), torch.zeros(model.n_elem, 8), atol=1e-10)


CYCLE = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0])


class TestLoadCycle:
    """Increments may fall as well as rise, so a solve can unload."""

    @pytest.mark.parametrize("control", ["force", "displacement"])
    def test_elastic_cycle_retraces(self, control):
        model = _planar()
        top = model.nodes[:, 1] > 1.0 - 1e-6
        model.constraints[model.nodes[:, 1] < 1e-6] = True
        if control == "force":
            model.forces[top, 1] = -1.0
        else:
            model.constraints[top, 1] = True
            model.displacements[top, 1] = -0.01

        u, f, _, _, _ = model.solve(increments=CYCLE, return_intermediate=True)
        u_top = u[:, top, 1].mean(dim=1)
        f_top = f[:, top, 1].sum(dim=1)

        # Elastic unloading retraces the loading path back to the origin
        assert torch.allclose(u_top, CYCLE * u_top[2])
        assert torch.allclose(f_top, CYCLE * f_top[2], atol=1e-10)

    def test_damage_unloads_on_a_secant(self):
        eps_0, eps_f = 1.0e-3, 1.0e-2

        def d(kappa, cl):
            evolution = 1.0 - eps_0 / kappa * torch.exp(-(kappa - eps_0) / eps_f)
            evolution[kappa < eps_0] = 0.0
            return evolution

        def d_prime(kappa, cl):
            derivative = (
                eps_0 * torch.exp(-(kappa - eps_0) / eps_f) * (1 / kappa**2 + 1 / eps_f)
            )
            derivative[kappa < eps_0] = 0.0
            return derivative

        material = IsotropicDamage3D(1000.0, 0.3, d, d_prime, "rankine")
        model = Solid(*cube_hexa(3, 3, 3), material)
        top = model.nodes[:, 2] > 1.0 - 1e-6
        model.constraints[model.nodes[:, 2] < 1e-6] = True
        model.constraints[top, 2] = True
        model.displacements[top, 2] = 0.02

        u, f, _, _, state = model.solve(increments=CYCLE, return_intermediate=True)
        f_top = f[:, top, 2].sum(dim=1)

        # Damage is irreversible, so unloading holds the state it reached
        assert state[2, :, 1].max() > 0.0
        assert torch.equal(state[2], state[3]) and torch.equal(state[3], state[4])

        # ... and the reaction returns to the origin along the degraded secant
        assert f_top[3] == pytest.approx(0.5 * f_top[2])
        assert f_top[4] == pytest.approx(0.0, abs=1e-8)
        assert u[4, top, 2].mean() == pytest.approx(0.0, abs=1e-12)
