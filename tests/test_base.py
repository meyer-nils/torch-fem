import pytest
import torch

from torchfem import Planar, PlanarHeat, SolidHeat
from torchfem.materials import (
    IsotropicConductivity2D,
    IsotropicConductivity3D,
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
