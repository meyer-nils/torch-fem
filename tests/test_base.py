import pytest
import torch

from torchfem import (
    Planar,
    PlanarHeat,
    Shell,
    ShellHeat,
    Solid,
    SolidHeat,
    Truss,
    TrussHeat,
)
from torchfem.materials import (
    Hyperelastic3D,
    IsotropicConductivity1D,
    IsotropicConductivity2D,
    IsotropicConductivity3D,
    IsotropicDamage3D,
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStress,
)
from torchfem.mesh import cube_hexa, rect_quad
from torchfem.rotations import axis_rotation


def _planar() -> Planar:
    return Planar(*rect_quad(3, 3), IsotropicElasticityPlaneStress(1000.0, 0.3))


def _planar_heat() -> PlanarHeat:
    return PlanarHeat(*rect_quad(3, 3), IsotropicConductivity2D(kappa=400.0))


def _bar() -> tuple[torch.Tensor, torch.Tensor]:
    """A single bar element, for the truss models."""
    return torch.tensor([[0.0, 0.0], [1.0, 0.0]]), torch.tensor([[0, 1]])


def _flat_quad() -> tuple[torch.Tensor, torch.Tensor]:
    """A single quadrilateral facet in the z = 0 plane, for the shell models."""
    nodes, elements = rect_quad(2, 2)
    return torch.hstack([nodes, torch.zeros(len(nodes), 1)]), elements


def _solid() -> Solid:
    return Solid(*cube_hexa(3, 3, 3), IsotropicElasticity3D(1000.0, 0.3))


def _shell() -> Shell:
    nodes, elements = rect_quad(3, 3)
    nodes = torch.hstack([nodes, torch.zeros(len(nodes), 1)])
    material = IsotropicElasticityPlaneStress(1000.0, 0.3)
    return Shell(nodes, elements, material, thickness=0.1)


def _truss() -> Truss:
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    return Truss(nodes, torch.tensor([[0, 1]]), IsotropicElasticity1D(1000.0))


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
        value = torch.randn(model.n_elem, *model.n_flux)
        model.ext_strain = value
        assert torch.equal(model.ext_strain, value)

    # A shell and a truss carry fewer strain components than they have nodal
    # DOFs, where a planar model and a solid carry the same number.
    @pytest.mark.parametrize("build", [_shell, _truss], ids=["shell", "truss"])
    def test_ext_strain_takes_the_flux_shape(self, build):
        model = build()
        value = torch.randn(model.n_elem, *model.n_flux)
        model.ext_strain = value
        assert torch.equal(model.ext_strain, value)
        with pytest.raises(ValueError, match="same shape as strains"):
            model.ext_strain = torch.zeros(
                model.n_elem, model.n_dof_per_node, model.n_dim
            )

    @pytest.mark.parametrize(
        ("build", "factor"),
        [
            (_planar, 1 / (1 - 0.3)),
            (_shell, 1 / (1 - 0.3)),
            (_solid, 1 / (1 - 2 * 0.3)),
        ],
        ids=["planar", "shell", "solid"],
    )
    def test_ext_strain_drives_a_restrained_thermal_stress(self, build, factor):
        model = build()
        strain = 1.2e-3
        model.ext_strain = strain * torch.eye(model.n_flux[0]).expand(
            model.n_elem, *model.n_flux
        )
        model.constraints[:] = True
        flux = model.solve(method="direct")[2]
        assert float(flux[:, 0, 0].mean()) == pytest.approx(-1000.0 * strain * factor)

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


class TestAssembleMatrix:
    def test_the_matrix_is_compressed_with_int32_indices(self):
        """The index arrays outweigh the values, so their width is the memory."""
        model = _planar()
        K = model.assemble_matrix(model.k0(), torch.tensor([0, 1]))
        assert K.layout == torch.sparse_csr
        assert K.crow_indices().dtype == torch.int32
        assert K.col_indices().dtype == torch.int32

    def test_constrained_rows_and_columns_hold_a_unit_diagonal(self):
        model = _planar()
        con = torch.tensor([0, 1])
        K = model.assemble_matrix(model.k0(), con).to_dense()
        assert torch.allclose(K[con].sum(dim=1), torch.ones(len(con)))
        assert torch.allclose(K[con, con], torch.ones(len(con)))
        assert torch.allclose(K[:, con].sum(dim=0), torch.ones(len(con)))


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


def _single_element(cls, mat, dim):
    nodes, elements = rect_quad(2, 2) if dim == 2 else cube_hexa(2, 2, 2)
    model = cls(nodes, elements, mat)
    model.constraints[nodes[:, 0] < 1e-9] = True
    model._dirichlet[nodes[:, 0] > 1 - 1e-9] = 0.01
    model.constraints[nodes[:, 0] > 1 - 1e-9] = True
    return model


@pytest.mark.parametrize(
    "cls, mat, dim, n_flux",
    [
        (Planar, IsotropicElasticityPlaneStress(1000.0, 0.3), 2, (2, 2)),
        (Solid, IsotropicElasticity3D(1000.0, 0.3), 3, (3, 3)),
        (PlanarHeat, IsotropicConductivity2D(400.0), 2, (2,)),
        (SolidHeat, IsotropicConductivity3D(400.0), 3, (3,)),
    ],
)
def test_solve_keeps_the_element_axis_for_one_element(cls, mat, dim, n_flux):
    model = _single_element(cls, mat, dim)
    assert model.n_elem == 1
    _, _, flux, grad, _ = model.solve()
    assert flux.shape == (1, *n_flux)
    assert grad.shape == (1, *n_flux)


def test_solve_keeps_the_integration_point_axis():
    model = _single_element(Planar, IsotropicElasticityPlaneStress(1000.0, 0.3), 2)
    _, _, flux, _, _ = model.solve(aggregate_integration_points=False)
    assert flux.shape == (model.n_int, 1, 2, 2)


def test_heat_solve_rejects_geometric_nonlinearity():
    """Heat conduction has no kinematics, so `nlgeom` is not silently ignored."""
    with pytest.raises(NotImplementedError, match="not implemented for PlanarHeat"):
        _planar_heat().solve(nlgeom=True)


def _prescribed_gradient(F):
    """A single hexahedron with every node driven to the deformation gradient `F`."""

    def psi(F, params):
        C = F.transpose(-1, -2) @ F
        logJ = 0.5 * torch.logdet(C)
        return params[0] / 2 * (torch.trace(C) - 3.0) - params[0] * logJ + logJ**2

    nodes, elements = cube_hexa(2, 2, 2)
    model = Solid(nodes, elements, Hyperelastic3D(psi, params=[100.0, 150.0]))
    model.constraints[:] = True
    model.displacements = nodes @ F.T - nodes
    return model


def test_nlgeom_reports_an_objective_cauchy_stress():
    """The Cauchy stress is symmetric, and a superposed rotation rotates it.

    `J^-1 P F^T` satisfies both. Its transpose, which a stretch alone cannot
    tell apart, satisfies neither.
    """
    R = axis_rotation(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.7))
    U = torch.tensor([[1.2, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])

    straight = _prescribed_gradient(U).solve(nlgeom=True)[2]
    turned = _prescribed_gradient(R @ U).solve(nlgeom=True)[2]

    assert torch.allclose(straight, straight.transpose(-1, -2), atol=1e-10)
    assert torch.allclose(turned, turned.transpose(-1, -2), atol=1e-10)
    assert torch.allclose(turned, R @ straight @ R.T, atol=1e-8)


class TestMaterialCompatibility:
    """A model takes a material of its own physics and dimension alone."""

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (
                lambda: Solid(*cube_hexa(2, 2, 2), IsotropicConductivity3D(400.0)),
                "Solid needs a 3D MechanicsMaterial, not a 3D IsotropicConductivity3D",
            ),
            (
                lambda: PlanarHeat(
                    *rect_quad(3, 3), IsotropicElasticityPlaneStress(1e3, 0.3)
                ),
                "PlanarHeat needs a 2D HeatMaterial, not a 2D IsotropicElasticity",
            ),
            (
                lambda: Planar(*rect_quad(3, 3), IsotropicElasticity3D(1000.0, 0.3)),
                "Planar needs a 2D MechanicsMaterial, not a 3D IsotropicElasticity3D",
            ),
            (
                lambda: Solid(
                    *cube_hexa(2, 2, 2), IsotropicElasticityPlaneStress(1e3, 0.3)
                ),
                "Solid needs a 3D MechanicsMaterial, not a 2D IsotropicElasticity",
            ),
            (
                lambda: PlanarHeat(*rect_quad(3, 3), IsotropicConductivity3D(400.0)),
                "PlanarHeat needs a 2D HeatMaterial, not a 3D IsotropicConductivity3D",
            ),
            (
                lambda: SolidHeat(*cube_hexa(2, 2, 2), IsotropicConductivity2D(400.0)),
                "SolidHeat needs a 3D HeatMaterial, not a 2D IsotropicConductivity2D",
            ),
            (
                lambda: ShellHeat(*_flat_quad(), IsotropicConductivity3D(400.0)),
                "ShellHeat needs a 2D HeatMaterial, not a 3D IsotropicConductivity3D",
            ),
            (
                lambda: ShellHeat(
                    *_flat_quad(), IsotropicElasticityPlaneStress(1e3, 0.3)
                ),
                "ShellHeat needs a 2D HeatMaterial, not a 2D IsotropicElasticity",
            ),
            (
                lambda: TrussHeat(*_bar(), IsotropicConductivity2D(400.0)),
                "TrussHeat needs a 1D HeatMaterial, not a 2D IsotropicConductivity2D",
            ),
            (
                lambda: Truss(*_bar(), IsotropicConductivity1D(400.0)),
                "Truss needs a 1D MechanicsMaterial, not a 1D IsotropicConductivity1D",
            ),
        ],
    )
    def test_rejects_an_incompatible_material(self, build, message):
        with pytest.raises(ValueError, match=message):
            build()
