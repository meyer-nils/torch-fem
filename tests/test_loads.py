import pytest
import torch

from torchfem import Planar, Shell, Solid, SolidHeat, Truss
from torchfem.elements import (
    Hexa1,
    Hexa2,
    Quad1,
    Quad2,
    Tetra1,
    Tetra2,
    Tria1,
    Tria2,
    linear_to_quadratic,
)
from torchfem.materials import (
    IsotropicConductivity3D,
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStress,
)
from torchfem.mesh import cube_hexa, cube_tetra, rect_quad, rect_tri

SOLID = IsotropicElasticity3D(E=1.0, nu=0.3)
PLANE = IsotropicElasticityPlaneStress(E=1.0, nu=0.3)

# Cuboid of 2 x 1 x 1, so the +x face has unit area
BOX = (2.0, 1.0, 1.0)


def _solid(gen, quadratic: bool) -> Solid:
    nodes, elements = gen(5, 4, 4, *BOX)
    if quadratic:
        nodes, elements = linear_to_quadratic(nodes, elements)
    return Solid(nodes, elements, SOLID)


def _planar(gen, quadratic: bool, nx: int = 5, ny: int = 4) -> Planar:
    nodes, elements = gen(nx, ny, 2.0, 1.0)
    if quadratic:
        nodes, elements = linear_to_quadratic(nodes, elements)
    return Planar(nodes, elements, PLANE)


def _shell(thickness: float = 0.02) -> Shell:
    nodes = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return Shell(nodes, elements, PLANE, thickness=thickness)


class TestFacetTables:
    """The facet tables must describe the reference element they belong to."""

    @pytest.mark.parametrize("etype", [Tetra1, Tetra2, Hexa1, Hexa2])
    def test_faces_are_planar_and_wound_outward(self, etype):
        coords = etype.iso_coords
        center = coords.mean(dim=0)
        for facet in etype.facets:
            points = coords[facet]
            normal = torch.linalg.cross(points[1] - points[0], points[2] - points[0])
            normal = normal / normal.norm()
            # All facet nodes lie in the facet plane
            assert torch.allclose(
                (points - points[0]) @ normal, torch.zeros(len(facet))
            )
            # The normal points away from the element center
            assert torch.dot(normal, points.mean(dim=0) - center) > 0

    @pytest.mark.parametrize("etype", [Tria1, Tria2, Quad1, Quad2])
    def test_edges_close_the_element(self, etype):
        facets = etype.facets
        assert len(facets) == (3 if etype in (Tria1, Tria2) else 4)
        # Consecutive edges share a node, so the edges form a closed loop
        assert torch.equal(facets[:, 1], facets.roll(-1, dims=0)[:, 0])

    @pytest.mark.parametrize("etype", [Tria2, Quad2, Tetra2, Hexa2])
    def test_quadratic_facets_carry_their_midside_nodes(self, etype):
        coords = etype.iso_coords
        assert etype.facets.shape[1] == etype.facet_type.nodes
        for facet in etype.facets:
            # A quadratic facet lists its corners first, then one midside node per edge
            corners = {3: 2, 6: 3, 8: 4}[len(facet)]
            for i in range(len(facet) - corners):
                ends = coords[facet[[i, (i + 1) % corners]]]
                assert torch.allclose(coords[facet[corners + i]], ends.mean(dim=0))


class TestShapeFunctionIntegrals:
    @pytest.mark.parametrize(
        "model",
        [
            _solid(cube_hexa, False),
            _solid(cube_tetra, False),
            _solid(cube_hexa, True),
            _planar(rect_quad, False),
            _shell(),
            Truss(
                torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
                torch.tensor([[0, 1]]),
                IsotropicElasticity1D(E=1.0),
            ),
        ],
    )
    def test_integrate_field_is_a_contraction_of_the_shape_function_integrals(
        self, model
    ):
        torch.manual_seed(0)
        field = torch.rand(model.n_nod)
        w = model.integrate_shape_functions()
        assert torch.allclose(
            model.integrate_field(field), (w * field[model.elements]).sum(dim=1)
        )
        assert torch.allclose(model.integrate_field(), w.sum(dim=1))


class TestSurfaceLoads:
    @pytest.mark.parametrize("gen", [cube_hexa, cube_tetra])
    @pytest.mark.parametrize("quadratic", [False, True])
    def test_pressure_on_a_closed_surface_has_no_net_force(self, gen, quadratic):
        model = _solid(gen, quadratic)
        f = model.integrate_surface_load(
            torch.ones(model.n_nod, dtype=torch.bool), torch.tensor(1.0)
        )
        assert torch.allclose(f.sum(dim=0), torch.zeros(3), atol=1e-12)

    @pytest.mark.parametrize("gen", [cube_hexa, cube_tetra])
    @pytest.mark.parametrize("quadratic", [False, True])
    def test_pressure_acts_along_the_outward_normal(self, gen, quadratic):
        model = _solid(gen, quadratic)
        f = model.integrate_surface_load(model.nodes[:, 0] == BOX[0], torch.tensor(1.0))
        assert torch.allclose(f.sum(dim=0), torch.tensor([1.0, 0.0, 0.0]), atol=1e-12)

    def test_traction_is_applied_in_global_coordinates(self):
        model = _solid(cube_hexa, False)
        traction = torch.tensor([0.0, 3.0, 0.0])
        f = model.integrate_surface_load(model.nodes[:, 0] == BOX[0], traction)
        assert torch.allclose(f.sum(dim=0), traction, atol=1e-12)

    def test_interior_faces_are_not_loaded(self):
        model = _solid(cube_hexa, False)
        # A mid-plane selects only faces shared by two elements
        f = model.integrate_surface_load(model.nodes[:, 0] == 1.0, torch.tensor(1.0))
        assert torch.allclose(f, torch.zeros_like(f))

    def test_shell_element_is_its_own_surface(self):
        model = _shell()
        f = model.integrate_surface_load(
            torch.ones(model.n_nod, dtype=torch.bool), torch.tensor(5.0)
        )
        # A flat unit square with a +z normal
        assert torch.allclose(f.sum(dim=0), torch.tensor([0.0, 0.0, 5.0]), atol=1e-12)

    def test_heat_flux_stays_scalar(self):
        nodes, elements = cube_hexa(5, 4, 2, *BOX)
        model = SolidHeat(nodes, elements, IsotropicConductivity3D(kappa=1.0))
        f = model.integrate_surface_load(nodes[:, 0] == BOX[0], torch.tensor(2.0))
        assert f.shape == (model.n_nod, 1)
        assert torch.isclose(f.sum(), torch.tensor(2.0))


class TestLineLoads:
    @pytest.mark.parametrize("gen", [rect_quad, rect_tri])
    @pytest.mark.parametrize("quadratic", [False, True])
    @pytest.mark.parametrize("size", [(5, 4), (3, 2)])
    def test_pressure_on_a_closed_boundary_has_no_net_force(self, gen, quadratic, size):
        model = _planar(gen, quadratic, *size)
        nodes = model.nodes
        boundary = (
            (nodes[:, 0] == 0.0)
            | (nodes[:, 0] == 2.0)
            | (nodes[:, 1] == 0.0)
            | (nodes[:, 1] == 1.0)
        )
        f = model.integrate_line_load(boundary, torch.tensor(1.0))
        assert torch.allclose(f.sum(dim=0), torch.zeros(2), atol=1e-12)

    @pytest.mark.parametrize("gen", [rect_quad, rect_tri])
    @pytest.mark.parametrize("quadratic", [False, True])
    def test_pressure_acts_along_the_outward_normal(self, gen, quadratic):
        model = _planar(gen, quadratic)
        f = model.integrate_line_load(model.nodes[:, 0] == 2.0, torch.tensor(1.0))
        assert torch.allclose(f.sum(dim=0), torch.tensor([1.0, 0.0]), atol=1e-12)

    def test_quadratic_edge_uses_consistent_weights(self):
        model = _planar(rect_quad, True, 3, 3)
        edge = model.nodes[:, 0] == 2.0
        f = model.integrate_line_load(edge, torch.tensor([1.0, 0.0]))
        weights = f[edge, 0][model.nodes[edge, 1].argsort()]
        # Consistent weights of a quadratic edge chain are 1 : 4 : 2 : 4 : 1, not the
        # 1 : 2 : 2 : 2 : 1 a linear lumping would give
        assert torch.allclose(
            weights / weights[0], torch.tensor([1.0, 4.0, 2.0, 4.0, 1.0])
        )

    def test_shell_edge_load_scales_with_length(self):
        nodes = torch.tensor(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 4.0, 0.0]]
        )
        model = Shell(
            nodes, torch.tensor([[0, 1, 2], [0, 2, 3]]), PLANE, thickness=0.02
        )
        f = model.integrate_line_load(
            nodes[:, 0] == 3.0, torch.tensor([0.0, 0.0, -2.0])
        )
        assert torch.allclose(f.sum(dim=0), torch.tensor([0.0, 0.0, -8.0]), atol=1e-12)


class TestBodyLoads:
    def test_solid_gravity_equals_weight(self):
        model = _solid(cube_hexa, False)
        f = model.integrate_body_load(torch.tensor([0.0, 0.0, -2.0]))
        assert torch.isclose(
            f.sum(dim=0)[2], torch.tensor(-2.0 * BOX[0] * BOX[1] * BOX[2])
        )

    def test_planar_gravity_includes_thickness(self):
        nodes, elements = rect_quad(6, 4, 2.0, 1.0)
        model = Planar(nodes, elements, PLANE, thickness=0.05)
        f = model.integrate_body_load(torch.tensor([0.0, -2.0]))
        assert torch.isclose(f.sum(dim=0)[1], torch.tensor(-2.0 * 2.0 * 1.0 * 0.05))

    def test_shell_gravity_includes_thickness_and_is_not_a_pressure(self):
        model = _shell(thickness=0.02)
        f = model.integrate_body_load(torch.tensor([0.0, 0.0, -2.0]))
        # Weight of a unit square, not a pressure over its area
        assert torch.isclose(f.sum(dim=0)[2], torch.tensor(-2.0 * 1.0 * 0.02))

    def test_truss_gravity_includes_cross_section(self):
        model = Truss(
            torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
            torch.tensor([[0, 1]]),
            IsotropicElasticity1D(E=1.0),
        )
        model.areas = torch.tensor([0.01])
        f = model.integrate_body_load(torch.tensor([0.0, -2.0]))
        assert torch.isclose(f.sum(dim=0)[1], torch.tensor(-2.0 * 5.0 * 0.01))

    def test_heat_source_stays_scalar(self):
        nodes, elements = cube_hexa(5, 4, 2, *BOX)
        model = SolidHeat(nodes, elements, IsotropicConductivity3D(kappa=1.0))
        f = model.integrate_body_load(torch.tensor([3.0]))
        assert f.shape == (model.n_nod, 1)
        assert torch.isclose(f.sum(), torch.tensor(3.0 * BOX[0] * BOX[1] * BOX[2]))

    def test_int_source_does_not_break_the_float_dtype(self):
        nodes, elements = cube_hexa(5, 4, 2, *BOX)
        model = SolidHeat(nodes, elements, IsotropicConductivity3D(kappa=1.0))
        f = model.integrate_body_load(3)
        assert f.dtype == nodes.dtype
        assert torch.isclose(f.sum(), torch.tensor(3.0 * BOX[0] * BOX[1] * BOX[2]))

    def test_per_element_load_matches_a_uniform_one(self):
        model = _solid(cube_hexa, False)
        load = torch.tensor([0.0, 0.0, -1.0])
        uniform = model.integrate_body_load(load)
        per_element = model.integrate_body_load(load.expand(model.n_elem, 3))
        assert torch.allclose(uniform, per_element)


class TestScalarLoadsAcceptFloats:
    """A scalar pressure or source needs no tensor wrapping at the call site."""

    def test_solid_pressure(self):
        model = _solid(cube_hexa, False)
        mask = model.nodes[:, 0] == BOX[0]
        assert torch.allclose(
            model.integrate_surface_load(mask, 2.5),
            model.integrate_surface_load(mask, torch.tensor(2.5)),
        )

    def test_shell_pressure(self):
        model = _shell()
        mask = torch.ones(model.n_nod, dtype=torch.bool)
        assert torch.allclose(
            model.integrate_surface_load(mask, 2.5),
            model.integrate_surface_load(mask, torch.tensor(2.5)),
        )

    def test_planar_edge_pressure(self):
        model = _planar(rect_quad, False)
        mask = model.nodes[:, 0] == 2.0
        assert torch.allclose(
            model.integrate_line_load(mask, 2.5),
            model.integrate_line_load(mask, torch.tensor(2.5)),
        )

    def test_heat_source(self):
        nodes, elements = cube_hexa(5, 4, 2, *BOX)
        model = SolidHeat(nodes, elements, IsotropicConductivity3D(kappa=1.0))
        assert torch.allclose(
            model.integrate_body_load(2.5),
            model.integrate_body_load(torch.tensor([2.5])),
        )

    def test_a_float_line_load_in_3d_is_still_ambiguous(self):
        model = _shell()
        mask = torch.ones(model.n_nod, dtype=torch.bool)
        with pytest.raises(ValueError, match="no unique normal"):
            model.integrate_line_load(mask, 1.0)


class TestUnsupportedLoads:
    def test_solid_has_no_line_loads(self):
        model = _solid(cube_hexa, False)
        mask = torch.ones(model.n_nod, dtype=torch.bool)
        with pytest.raises(NotImplementedError, match="no edges to load"):
            model.integrate_line_load(mask, torch.tensor([0.0, 0.0, 1.0]))

    def test_planar_has_no_surface_loads(self):
        model = _planar(rect_quad, False)
        mask = torch.ones(model.n_nod, dtype=torch.bool)
        with pytest.raises(NotImplementedError, match="no surfaces to load"):
            model.integrate_surface_load(mask, torch.tensor(1.0))

    @pytest.mark.parametrize(
        "method", ["integrate_surface_load", "integrate_line_load"]
    )
    def test_truss_has_neither(self, method):
        model = Truss(
            torch.tensor([[0.0, 0.0], [3.0, 4.0]]),
            torch.tensor([[0, 1]]),
            IsotropicElasticity1D(E=1.0),
        )
        mask = torch.ones(model.n_nod, dtype=torch.bool)
        with pytest.raises(NotImplementedError, match="to load"):
            getattr(model, method)(mask, torch.tensor([0.0, 1.0]))

    def test_scalar_line_load_in_3d_is_ambiguous(self):
        model = _shell()
        mask = torch.ones(model.n_nod, dtype=torch.bool)
        with pytest.raises(ValueError, match="no unique normal"):
            model.integrate_line_load(mask, torch.tensor(1.0))
