"""Model-level tests exercising every supported element type."""

import math

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

    def test_solve_rejects_geometric_nonlinearity(self):
        material = IsotropicElasticityPlaneStress(1000.0, 0.3)
        shell = Shell(self.nodes, self.elements, material)
        shell.constraints[0] = True
        shell.forces[2, 2] = 1.0
        with pytest.raises(NotImplementedError, match="not implemented for Shell"):
            shell.solve(nlgeom=True)

    def test_integrates_the_transverse_moduli_of_an_orthotropic_material(self):
        material = OrthotropicElasticityPlaneStress(
            E_1=100e3, E_2=10e3, nu_12=0.3, G_12=5e3, G_13=4.8e3, G_23=3e3
        )
        shell = Shell(self.nodes, self.elements, material, thickness=0.2)
        expected = 0.2 * torch.tensor([[4.8e3, 0.0], [0.0, 3e3]])
        assert torch.allclose(shell.As[0], expected)


class TestShellTransverseShear:
    """A narrow strip with `nu=0` is exactly a Timoshenko beam, so its tip
    deflection checks the transverse shear stiffness away from the thin limit.
    The shear stiffness used to carry a spurious factor of the element area,
    which cancels only for thin shells and grows under mesh refinement."""

    E, L, b, P, kappa = 1000.0, 10.0, 1.0, 1.0, 5.0 / 6.0

    @pytest.mark.parametrize("t", [2.0, 0.5])
    @pytest.mark.parametrize("etype", ["Tria1", "Quad1"])
    def test_tip_deflection_matches_timoshenko(self, t, etype):
        if etype == "Tria1":
            nodes, elements = rect_tri(41, 3, self.L, self.b, variant="center")
        else:
            nodes, elements = rect_quad(41, 3, self.L, self.b)
        nodes = torch.hstack([nodes, torch.zeros((len(nodes), 1))])
        material = IsotropicElasticityPlaneStress(E=self.E, nu=0.0)
        beam = Shell(nodes, elements, material, thickness=t)
        tip = nodes[:, 0] > self.L - 1e-9
        beam.forces[tip, 2] = self.P / int(tip.sum())
        beam.constraints[nodes[:, 0] < 1e-9, :] = True
        u, _, _, _, _ = beam.solve(method="spsolve")

        bending = self.P * self.L**3 / (3 * self.E * self.b * t**3 / 12)
        shear = self.P * self.L / (self.kappa * (self.E / 2) * self.b * t)
        assert u[:, 2].abs().max() == pytest.approx(bending + shear, rel=0.01)


def _clamped_plate(n: int, t: float, E: float = 1.0e6, nu: float = 0.3, q: float = 1.0):
    """Uniformly loaded square plate, clamped all round, on an `n` by `n` quad mesh."""
    nodes, elements = rect_quad(n + 1, n + 1)
    nodes = torch.hstack([nodes, torch.zeros((len(nodes), 1))])
    material = IsotropicElasticityPlaneStress(E=E, nu=nu)
    plate = Shell(nodes, elements, material, thickness=t)
    surface = torch.ones(plate.n_nod, dtype=torch.bool)
    load = torch.tensor([0.0, 0.0, q])
    plate.forces[:, 0:3] = plate.integrate_surface_load(surface, load)
    edge = ((nodes[:, :2] < 1e-9) | (nodes[:, :2] > 1.0 - 1e-9)).any(dim=1)
    plate.constraints[edge, :] = True
    plate.constraints[:, [0, 1, 5]] = True
    u, _, _, _, _ = plate.solve(method="spsolve")
    # Normalized on the thin-plate deflection 0.00126 q L^4 / D of Timoshenko
    D = E * t**3 / (12 * (1 - nu**2))
    return float(u[:, 2].max()) / (0.00126 * q / D)


class TestShellQuadrilateral:
    """The MITC4 quadrilateral ties its transverse shear strains to the element
    edges, which keeps a thin plate from locking. A plain bilinear quadrilateral
    returns a deflection near zero in the same test."""

    # Distorted patch, so the tests below do not sit on a regular mesh
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.31, 0.24, 0.0],
            [0.72, 0.33, 0.0],
            [0.66, 0.71, 0.0],
            [0.28, 0.63, 0.0],
        ]
    )
    elements = torch.tensor([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    material = IsotropicElasticityPlaneStress(1000.0, 0.3)

    @pytest.mark.parametrize("t", [1.0e-2, 1.0e-3, 1.0e-4])
    def test_thin_plate_does_not_lock(self, t):
        """The deflection stays at the thin-plate limit however thin the plate is."""
        assert _clamped_plate(16, t) == pytest.approx(1.0, rel=0.01)

    def test_plate_converges_under_refinement(self):
        coarse, fine = _clamped_plate(4, 1.0e-3), _clamped_plate(16, 1.0e-3)
        assert abs(fine - 1.0) < abs(coarse - 1.0)

    def test_reproduces_a_linear_displacement_field(self):
        """Membrane patch test: a uniform strain is integrated exactly."""
        eps = 1.0e-3
        patch = Shell(self.nodes, self.elements, self.material, thickness=0.1)
        patch.constraints[:] = True
        patch.displacements[:, 0] = eps * self.nodes[:, 0]
        patch.displacements[:, 1] = -0.3 * eps * self.nodes[:, 1]
        _, _, sigma, _, _ = patch.solve(method="spsolve")
        assert sigma[..., 0, 0] == pytest.approx(1000.0 * eps, abs=1e-12)
        assert sigma[..., 1, 1] == pytest.approx(0.0, abs=1e-12)
        assert sigma[..., 0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_reproduces_a_constant_curvature(self):
        """Bending patch test: a uniform curvature is integrated exactly."""
        t, c, nu = 0.1, 1.0e-4, 0.3
        patch = Shell(self.nodes, self.elements, self.material, thickness=t)
        patch.constraints[:] = True
        patch.displacements[:, 2] = (
            c / 2 * (self.nodes[:, 0] ** 2 + self.nodes[:, 1] ** 2)
        )
        patch.displacements[:, 3] = -c * self.nodes[:, 1]
        patch.displacements[:, 4] = c * self.nodes[:, 0]
        _, _, sigma, _, _ = patch.solve(
            method="spsolve", aggregate_integration_points=False
        )
        # Integration points run over the in-plane points and the Simpson stations
        outer = sigma.reshape(-1, patch.n_z, patch.n_elem, 2, 2)[:, -1]
        expected = 1000.0 / (1 - nu**2) * (1 + nu) * c * t / 2
        assert outer[..., 0, 0] == pytest.approx(expected, abs=1e-12)
        assert outer[..., 0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_has_no_spurious_zero_energy_modes(self):
        """Six rigid body modes and one drilling mode per node, and nothing else."""
        corners = torch.tensor(
            [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.9, 1.3, 0.0], [-0.1, 0.8, 0.0]]
        )
        element = Shell(
            corners, torch.tensor([[0, 1, 2, 3]]), self.material, drill_penalty=0.0
        )
        eigenvalues = torch.linalg.eigvalsh(element.k0()[0])
        zero = eigenvalues < 1e-9 * eigenvalues.max()
        assert int(zero.sum()) == 6 + 4

    def test_rejects_an_unsupported_element(self):
        nodes = torch.zeros(5, 3)
        with pytest.raises(ValueError, match="Element type not supported"):
            Shell(nodes, torch.tensor([[0, 1, 2, 3, 4]]), self.material)


class TestShellDrilling:
    """The drilling rotation only enters the response where element normals differ,
    so a curved shell is the only place it shows. Every other shell test here is
    flat, where it decouples and the penalty has no effect at all."""

    @pytest.mark.parametrize("tri", [False, True])
    def test_pinched_hemisphere(self, tri):
        """Hemisphere with an 18° hole pulled apart by two pairs of opposed radial
        loads (MacNeal and Harder), meshed by wrapping a unit square onto the sphere.
        The deformation is nearly inextensional, so a penalty resisting a rigid
        rotation locks it well below the reference deflection of 0.0940."""
        n, R, t, E, nu = 8, 10.0, 0.04, 6.825e7, 0.3
        grid, elements = rect_tri(n + 1, n + 1) if tri else rect_quad(n + 1, n + 1)
        phi, theta = math.radians(72.0) * grid[:, 0], math.pi / 2 * grid[:, 1]
        nodes = R * torch.stack(
            [phi.cos() * theta.cos(), phi.cos() * theta.sin(), phi.sin()], dim=1
        )
        material = IsotropicElasticityPlaneStress(E=E, nu=nu)
        model = Shell(nodes, elements, material, thickness=t)
        x, y, z = nodes.T

        # Symmetry on the two cut planes, one node pinned against rigid translation
        sym_x, sym_y = x.abs() < 1e-9, y.abs() < 1e-9
        model.constraints[sym_y, 1] = model.constraints[sym_y, 3] = True
        model.constraints[sym_x, 0] = model.constraints[sym_x, 4] = True
        model.constraints[sym_x | sym_y, 5] = True

        # Opposed radial loads at the equator, outward on x and inward on y
        equator = z.abs() < 1e-9
        outward = int(torch.argmax((equator & sym_y).double()))
        inward = int(torch.argmax((equator & sym_x).double()))
        model.constraints[outward, 2] = True
        model.forces[outward, 0] = 1.0
        model.forces[inward, 1] = -1.0

        u, *_ = model.solve(method="spsolve")
        assert u[outward, 0].item() / 0.0940 == pytest.approx(1.0, rel=0.03)

    @pytest.mark.parametrize("n", [4, 3])
    def test_a_rigid_rotation_carries_no_energy(self, n):
        """The penalty ties the drilling rotation to the in-plane rotation of the
        membrane field rather than to zero, so a tilted element does not resist the
        drilling a rigid rotation leaves on it."""
        # Planar but tilted, so a rotation about z drills the element
        nodes = torch.tensor(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.36], [1.1, 1.3, 0.33], [0.0, 0.8, 0.0]]
        )[:n]
        material = IsotropicElasticityPlaneStress(1000.0, 0.3)
        element = Shell(nodes, torch.arange(n)[None], material, thickness=0.1)
        axis = torch.tensor([0.2, -0.4, 1.0])
        v = torch.zeros(n, 6)
        v[:, 0:3] = torch.linalg.cross(axis.expand(n, 3), nodes)
        v[:, 3:6] = axis
        k, v = element.k0()[0], v.reshape(-1)
        assert v @ k @ v / (v @ v) < 1e-9 * torch.linalg.eigvalsh(k).max()


class TestRepr:
    """`__repr__` names the element type, not the metaclass of its class."""

    @pytest.mark.parametrize("etype", ETYPES)
    def test_planar_and_solid_name_the_element_type(self, etype):
        assert etype in repr(_build(etype))

    def test_truss_names_the_element_type(self):
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        truss = Truss(nodes, torch.tensor([[0, 1]]), IsotropicElasticity1D(1000.0))
        assert "Bar1" in repr(truss)

    @pytest.mark.parametrize(
        "etype, elements", [("Tria1", [[0, 1, 2]]), ("Quad1", [[0, 1, 2, 3]])]
    )
    def test_shell_names_the_element_type(self, etype, elements):
        nodes = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        material = IsotropicElasticityPlaneStress(1000.0, 0.3)
        shell = Shell(nodes, torch.tensor(elements), material, thickness=0.1)
        assert etype in repr(shell)
