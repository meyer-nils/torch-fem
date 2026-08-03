"""Model-level tests exercising every supported element type."""

import pytest
import torch

from torchfem import Planar, Shell, Solid, Truss
from torchfem.elements import linear_to_quadratic
from torchfem.materials import (
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStress,
    OrthotropicElasticityPlaneStress,
)
from torchfem.mesh import cube_hexa, cube_tetra, rect_quad, rect_tri

ETYPES = ["Tria1", "Tria2", "Quad1", "Quad2", "Tetra1", "Tetra2", "Hexa1", "Hexa2"]

# Gradients of a linear displacement field, which any conforming element
# reproduces exactly.
GRADIENT_2D = torch.tensor([[1.0e-3, 2.0e-4], [3.0e-4, -1.0e-3]])
GRADIENT_3D = torch.tensor(
    [[1.0e-3, 2.0e-4, -1.0e-4], [3.0e-4, -1.0e-3, 5.0e-5], [1.0e-4, 2.0e-4, 7.0e-4]]
)


def _build(etype: str) -> Planar | Solid:
    """Build the model whose connectivity selects `etype`."""
    planar = IsotropicElasticityPlaneStress(1000.0, 0.3)
    solid = IsotropicElasticity3D(1000.0, 0.3)
    cases = {
        "Tria1": (rect_tri(4, 4), Planar, planar, False),
        "Tria2": (rect_tri(4, 4), Planar, planar, True),
        "Quad1": (rect_quad(4, 4), Planar, planar, False),
        "Quad2": (rect_quad(4, 4), Planar, planar, True),
        "Tetra1": (cube_tetra(3, 3, 3), Solid, solid, False),
        "Tetra2": (cube_tetra(3, 3, 3), Solid, solid, True),
        "Hexa1": (cube_hexa(3, 3, 3), Solid, solid, False),
        "Hexa2": (cube_hexa(3, 3, 3), Solid, solid, True),
    }
    mesh, model, material, quadratic = cases[etype]
    nodes, elements = linear_to_quadratic(*mesh) if quadratic else mesh
    return model(nodes, elements, material)


def _on_boundary(nodes: torch.Tensor) -> torch.Tensor:
    """Mask of nodes on any face of the bounding box."""
    mask = torch.zeros(len(nodes), dtype=torch.bool)
    for dim in range(nodes.shape[1]):
        coord = nodes[:, dim]
        mask |= torch.isclose(coord, coord.min()) | torch.isclose(coord, coord.max())
    return mask


class TestPatch:
    @pytest.mark.parametrize("etype", ETYPES)
    def test_reproduces_a_linear_displacement_field(self, etype):
        """First-order patch test: prescribing a linear field on the boundary must
        reproduce it exactly in the interior and give the same stress everywhere."""
        model = _build(etype)
        assert model.etype.__name__ == etype

        gradient = GRADIENT_2D if model.n_dim == 2 else GRADIENT_3D
        u_exact = model.nodes @ gradient.T
        model.constraints = _on_boundary(model.nodes)[:, None].repeat(
            1, model.n_dof_per_node
        )
        model.displacements = u_exact

        u, _, sigma, _, _ = model.solve()
        assert torch.allclose(u, u_exact, atol=1e-12)
        assert torch.allclose(sigma, sigma[0].expand_as(sigma), atol=1e-12)

    @pytest.mark.parametrize("model", [Planar, Solid])
    def test_rejects_unsupported_connectivity(self, model):
        if model is Planar:
            nodes, material = (
                rect_quad(3, 3)[0],
                IsotropicElasticityPlaneStress(1e3, 0.3),
            )
        else:
            nodes, material = cube_hexa(2, 2, 2)[0], IsotropicElasticity3D(1e3, 0.3)
        with pytest.raises(ValueError, match="not supported"):
            _ = model(nodes, torch.tensor([[0, 1, 2, 3, 4]]), material).etype


class TestThickness:
    def test_planar_accepts_one_thickness_per_element(self):
        nodes, elements = rect_quad(3, 3)
        thickness = torch.linspace(0.5, 1.5, len(elements))
        material = IsotropicElasticityPlaneStress(1000.0, 0.3)
        model = Planar(nodes, elements, material, thickness=thickness)
        assert torch.equal(model.thickness, thickness)

    def test_shell_accepts_one_thickness_per_element(self):
        nodes = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        elements = torch.tensor([[0, 1, 2]])
        material = IsotropicElasticityPlaneStress(1000.0, 0.3)
        thickness = torch.tensor([0.2])
        shell = Shell(nodes, elements, material, thickness=thickness)
        assert torch.equal(shell.thickness, thickness)


class TestShellValidation:
    nodes = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements = torch.tensor([[0, 1, 2]])

    def test_rejects_even_n_simpson(self):
        material = IsotropicElasticityPlaneStress(1000.0, 0.3)
        with pytest.raises(ValueError, match="odd integer"):
            Shell(self.nodes, self.elements, material, n_simpson=4)

    def test_requires_a_shear_modulus(self):
        """An orthotropic material without transverse moduli defines no
        transverse shear stiffness, and neither does `G_12`."""
        material = OrthotropicElasticityPlaneStress(
            E_1=100e3, E_2=10e3, nu_12=0.3, G_12=5e3
        )
        with pytest.raises(ValueError, match="shear modulus"):
            Shell(self.nodes, self.elements, material)

    def test_integrates_the_transverse_moduli_of_an_orthotropic_material(self):
        material = OrthotropicElasticityPlaneStress(
            E_1=100e3, E_2=10e3, nu_12=0.3, G_12=5e3, G_13=4.8e3, G_23=3e3
        )
        shell = Shell(self.nodes, self.elements, material, thickness=0.2)
        expected = 0.2 * torch.tensor([[4.8e3, 0.0], [0.0, 3e3]])
        assert torch.allclose(shell.As[0], expected)


class TestRepr:
    """`__repr__` names the element type, not the metaclass of its class."""

    @pytest.mark.parametrize("etype", ETYPES)
    def test_planar_and_solid_name_the_element_type(self, etype):
        assert etype in repr(_build(etype))

    def test_truss_names_the_element_type(self):
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        truss = Truss(nodes, torch.tensor([[0, 1]]), IsotropicElasticity1D(1000.0))
        assert "Bar1" in repr(truss)

    def test_shell_names_the_element_type(self):
        nodes = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        material = IsotropicElasticityPlaneStress(1000.0, 0.3)
        shell = Shell(nodes, torch.tensor([[0, 1, 2]]), material, thickness=0.1)
        assert "Tria1" in repr(shell)
