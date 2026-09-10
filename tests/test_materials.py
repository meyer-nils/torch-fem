import pytest
import torch

import torchfem.materials
from torchfem.materials import (
    HeatMaterial,
    Hyperelastic3D,
    HyperelasticPlaneStress,
    IsotropicConductivity2D,
    IsotropicConductivity3D,
    IsotropicDamage1D,
    IsotropicDamage3D,
    IsotropicDamagePlaneStrain,
    IsotropicDamagePlaneStress,
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStrain,
    IsotropicElasticityPlaneStress,
    IsotropicPlasticity1D,
    IsotropicPlasticity3D,
    IsotropicPlasticityPlaneStrain,
    IsotropicPlasticityPlaneStress,
    MechanicsMaterial,
    OrthotropicConductivity2D,
    OrthotropicConductivity3D,
    OrthotropicElasticity3D,
    OrthotropicElasticityPlaneStrain,
    OrthotropicElasticityPlaneStress,
    TransverseIsotropicElasticity3D,
    TransverseIsotropicElasticityPlaneStrain,
    TransverseIsotropicElasticityPlaneStress,
)
from torchfem.rotations import axis_rotation, planar_rotation
from torchfem.utils import stiffness2voigt, stress2voigt

N_ELEM = 10


def _make_step_args_3d(n_elem=1, n_state=0):
    """Create minimal tensors for calling step() on a 3D material."""
    H_base = torch.tensor(
        [[1.0e-3, 2.0e-4, -1.0e-4], [0.0, -6.0e-4, 3.0e-4], [5.0e-5, 1.0e-4, 4.0e-4]]
    )
    H_inc = H_base.unsqueeze(0).repeat(n_elem, 1, 1)
    F = torch.eye(3).unsqueeze(0).expand(n_elem, -1, -1).clone()
    stress = torch.zeros(n_elem, 3, 3)
    state = torch.zeros(n_elem, n_state)
    de0 = torch.zeros(n_elem, 3, 3)
    cl = torch.ones(n_elem)
    return H_inc, F, stress, state, de0, cl


def _make_step_args_2d(n_elem=1, n_state=0):
    """Create minimal tensors for calling step() on a 2D material."""
    H_base = torch.tensor([[1.0e-3, 2.0e-4], [-1.0e-4, -6.0e-4]])
    H_inc = H_base.unsqueeze(0).repeat(n_elem, 1, 1)
    F = torch.eye(2).unsqueeze(0).expand(n_elem, -1, -1).clone()
    stress = torch.zeros(n_elem, 2, 2)
    state = torch.zeros(n_elem, n_state)
    de0 = torch.zeros(n_elem, 2, 2)
    cl = torch.ones(n_elem)
    return H_inc, F, stress, state, de0, cl


def _make_step_args_1d(n_elem=1, n_state=0):
    """Create minimal tensors for calling step() on a 1D material."""
    H_inc = 1.0e-3 * torch.ones(n_elem, 1, 1)
    F = torch.ones(n_elem, 1, 1)
    stress = torch.zeros(n_elem, 1, 1)
    state = torch.zeros(n_elem, n_state)
    de0 = torch.zeros(n_elem, 1, 1)
    cl = torch.ones(n_elem)
    return H_inc, F, stress, state, de0, cl


def _make_thermal_step_args(dim, n_elem=1):
    """Create minimal tensors for calling step() on a conductivity material."""
    grad_inc = torch.linspace(1.0, float(dim), dim).expand(n_elem, 1, dim).clone()
    grad = torch.zeros(n_elem, 1, dim)
    heat_flux = torch.zeros(n_elem, 1, dim)
    state = torch.zeros(n_elem, 0)
    cl = torch.ones(n_elem)
    return grad_inc, grad, heat_flux, state, cl


# Common yield function for plasticity tests
def sigma_f(ep):
    return 200.0 + 50.0 * ep


def sigma_f_prime(ep):
    return torch.full_like(ep, 50.0)


def _make_step_args(dim, n_elem=1, n_state=0):
    """Dispatch to the step arguments of a `dim`-dimensional mechanics material."""
    builder = {1: _make_step_args_1d, 2: _make_step_args_2d, 3: _make_step_args_3d}
    return builder[dim](n_elem, n_state)


# Every material whose step is a plain contraction of a constant stiffness. They
# share `step`, `vectorize` and `rotate` through their bases, so the properties
# below are checked once over the table rather than once per class.
LINEAR = [
    pytest.param(lambda: IsotropicElasticity3D(1000.0, 0.3), 3, id="isotropic-3d"),
    pytest.param(lambda: IsotropicElasticityPlaneStress(1000.0, 0.3), 2, id="iso-ps"),
    pytest.param(lambda: IsotropicElasticityPlaneStrain(1000.0, 0.3), 2, id="iso-pe"),
    pytest.param(lambda: IsotropicElasticity1D(1000.0), 1, id="isotropic-1d"),
    pytest.param(lambda: _orthotropic_3d(), 3, id="orthotropic-3d"),
    pytest.param(lambda: _plane_stress(), 2, id="orthotropic-ps"),
    pytest.param(lambda: _plane_strain(), 2, id="orthotropic-pe"),
    pytest.param(lambda: TransverseIsotropicElasticity3D(**TI), 3, id="transverse-3d"),
]

# Conductivities, with the matrix they build and the flux they return for the
# gradient `[1, ..., dim]` that `_make_thermal_step_args` applies.
CONDUCTIVITIES = [
    pytest.param(
        lambda: IsotropicConductivity3D(400.0), [400.0] * 3, id="isotropic-3d"
    ),
    pytest.param(
        lambda: IsotropicConductivity2D(400.0), [400.0] * 2, id="isotropic-2d"
    ),
    pytest.param(
        lambda: OrthotropicConductivity3D(1.0, 2.0, 3.0),
        [1.0, 2.0, 3.0],
        id="orthotropic-3d",
    ),
    pytest.param(
        lambda: OrthotropicConductivity2D(1.0, 2.0), [1.0, 2.0], id="orthotropic-2d"
    ),
]


@pytest.mark.parametrize("build, dim", LINEAR)
class TestLinearMechanics:
    def test_stiffness_has_the_dimension_of_the_material(self, build, dim):
        assert build().C.shape == (dim,) * 4

    def test_stiffness_keeps_its_symmetries(self, build, dim):
        C = build().C
        # Major symmetry: C_ijkl = C_klij, minor symmetry: C_ijkl = C_jikl
        assert torch.allclose(C, C.permute(2, 3, 0, 1), atol=1e-6)
        assert torch.allclose(C, C.permute(1, 0, 2, 3), atol=1e-6)

    def test_step_contracts_the_stiffness_with_the_strain_increment(self, build, dim):
        mat = build().vectorize(N_ELEM)
        H_inc, F, stress, state, de0, cl = _make_step_args(dim, N_ELEM)
        s_new, st_new, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, de)
        assert s_new.shape == (N_ELEM, dim, dim)
        assert ddsdde.shape == (N_ELEM, *(dim,) * 4)
        assert torch.allclose(s_new, expected, atol=1e-10, rtol=1e-10)
        # A linear material carries no state, so the step leaves it alone
        assert torch.equal(st_new, state)
        assert torch.isfinite(ddsdde).all()

    def test_vectorize_batches_the_stiffness(self, build, dim):
        assert build().vectorize(N_ELEM).C.shape == (N_ELEM, *(dim,) * 4)


@pytest.mark.parametrize("build, kappa", CONDUCTIVITIES)
class TestConductivities:
    def test_conductivity_is_diagonal_in_the_principal_axes(self, build, kappa):
        assert torch.allclose(build().KAPPA, torch.diag(torch.tensor(kappa)))

    def test_step_applies_the_conductivity_per_direction(self, build, kappa):
        dim = len(kappa)
        mat = build().vectorize(N_ELEM)
        grad_inc, grad, q, state, cl = _make_thermal_step_args(dim, N_ELEM)
        q_new, state_new, tangent = mat.step(grad_inc, grad, q, state, cl, 0)
        # The gradient runs [1, ..., dim] against the conductivities
        expected = torch.tensor(kappa) * torch.arange(1, dim + 1)
        assert torch.allclose(q_new, expected.expand(N_ELEM, 1, dim))
        assert tangent.shape == (N_ELEM, dim, dim)
        assert torch.equal(state_new, state)

    def test_vectorize_batches_the_conductivity(self, build, kappa):
        mat = build().vectorize(N_ELEM)
        assert mat.KAPPA.shape == (N_ELEM, len(kappa), len(kappa))


@pytest.mark.parametrize(
    "build", [p.values[0] for p in LINEAR + CONDUCTIVITIES], ids=lambda b: ""
)
def test_vectorize_is_idempotent(build):
    """`vectorize` is shared by every material, and returns `self` when batched."""
    mat = build().vectorize(N_ELEM)
    assert mat.vectorize(N_ELEM) is mat


class TestIsotropicElasticity3D:
    def test_lame_parameters(self):
        E, nu = 210e3, 0.3
        mat = IsotropicElasticity3D(E, nu)
        lbd_expected = E * nu / ((1 + nu) * (1 - 2 * nu))
        G_expected = E / (2 * (1 + nu))
        assert torch.allclose(mat.lbd, torch.tensor(lbd_expected))
        assert torch.allclose(mat.G, torch.tensor(G_expected))


class TestHyperelastic3D:
    @staticmethod
    def neo_hookean(F, params):
        mu, lam = params[0], params[1]
        J = torch.linalg.det(F)
        C = F.T @ F
        return (
            0.5 * mu * (torch.trace(C) - 3)
            - mu * torch.log(J)
            + 0.5 * lam * torch.log(J) ** 2
        )

    def test_step(self):
        mat = Hyperelastic3D(self.neo_hookean, torch.tensor([80.0, 120.0]))
        n = N_ELEM
        mat_v = mat.vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_3d(n)
        H_inc = 10.0 * H_inc
        s_new, _, ddsdde = mat_v.step(H_inc, F, stress, state, de0, cl, 0)
        assert s_new.shape == (n, 3, 3)
        assert ddsdde.shape == (n, 3, 3, 3, 3)
        assert torch.isfinite(s_new).all()

    def test_zero_increment_at_identity_gives_zero_stress(self):
        n = N_ELEM
        mat = Hyperelastic3D(self.neo_hookean, torch.tensor([80.0, 120.0])).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_3d(n)
        H_inc.zero_()
        s_new, st_new, _ = mat.step(H_inc, F, stress, state, de0, cl, 0)
        assert torch.allclose(s_new, torch.zeros_like(s_new), atol=1e-8)
        assert torch.allclose(st_new, state)

    def test_vectorize(self):
        mat = Hyperelastic3D(self.neo_hookean, torch.tensor([80.0, 120.0]))
        mat_v = mat.vectorize(N_ELEM)
        assert mat_v.params.shape == (N_ELEM, 2)


class TestIsotropicPlasticity3D:
    def test_elastic_step(self):
        """Very small strain should stay elastic."""
        n = N_ELEM
        mat = IsotropicPlasticity3D(210e3, 0.3, sigma_f, sigma_f_prime).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_3d(n, n_state=1)
        H_inc.zero_()
        H_inc[:, 0, 0] = 1e-6
        H_inc[:, 1, 1] = 1e-6
        H_inc[:, 2, 2] = 1e-6
        s_new, st_new, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, de)
        assert s_new.shape == (n, 3, 3)
        assert st_new.shape == (n, 1)
        assert torch.allclose(s_new, expected, atol=1e-6, rtol=1e-6)
        # Equivalent plastic strain should remain zero for elastic step
        assert torch.allclose(st_new, torch.zeros_like(st_new), atol=1e-8)
        assert torch.isfinite(ddsdde).all()

    def test_plastic_step(self):
        """Large shear strain should trigger plasticity."""
        n = N_ELEM
        mat = IsotropicPlasticity3D(210e3, 0.3, sigma_f, sigma_f_prime).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_3d(n, n_state=1)
        H_inc.zero_()
        H_inc[:, 0, 0] = 0.01
        H_inc[:, 1, 1] = -0.005
        H_inc[:, 2, 2] = -0.005
        s_new, st_new, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        assert s_new.shape == (n, 3, 3)
        # Plastic strain should be positive for all elements
        assert (st_new[:, 0] > 0).all()
        assert torch.isfinite(s_new).all()
        assert torch.isfinite(ddsdde).all()


class TestIsotropicPlasticity1D:
    def test_elastic_step(self):
        n = N_ELEM
        mat = IsotropicPlasticity1D(1000.0, sigma_f, sigma_f_prime).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_1d(n, n_state=1)
        H_inc[:] = 1e-6  # very small
        s_new, st_new, _ = mat.step(H_inc, F, stress, state, de0, cl, 0)
        assert s_new.shape == (n, 1, 1)
        assert torch.allclose(st_new, torch.zeros_like(st_new), atol=1e-8)
        assert torch.isfinite(s_new).all()

    @pytest.mark.parametrize("n_yielding", [1, 2, N_ELEM])
    def test_plastic_step_yields_a_per_element_tangent(self, n_yielding):
        """The tangent must broadcast per element, not across the yielding ones."""
        n, E, H = N_ELEM, 1000.0, 50.0
        mat = IsotropicPlasticity1D(E, sigma_f, sigma_f_prime).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_1d(n, n_state=1)
        H_inc[:] = 0.0
        H_inc[:n_yielding] = 1.0  # well past yield
        s_new, st_new, tangent = mat.step(H_inc, F, stress, state, de0, cl, 1)
        assert tangent.shape == (n, 1, 1, 1, 1)
        assert torch.allclose(
            tangent[:n_yielding], torch.full((n_yielding, 1, 1, 1, 1), E * H / (E + H))
        )
        assert torch.allclose(
            tangent[n_yielding:], torch.full_like(tangent[n_yielding:], E)
        )
        assert (st_new[:n_yielding, 0] > 0).all()
        assert (st_new[n_yielding:, 0] == 0).all()


class TestIsotropicPlasticityPlaneStress:
    def test_elastic_step(self):
        n = N_ELEM
        mat = IsotropicPlasticityPlaneStress(
            210e3, 0.3, sigma_f, sigma_f_prime
        ).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_2d(n, n_state=1)
        H_inc[:] = 1e-7
        s_new, st_new, _ = mat.step(H_inc, F, stress, state, de0, cl, 0)
        assert s_new.shape == (n, 2, 2)
        assert torch.allclose(st_new, torch.zeros_like(st_new), atol=1e-8)
        assert torch.isfinite(s_new).all()

    def test_plastic_step_batched(self):
        """All points in a batch yield together and the consistent tangent
        matches a finite difference.

        This exercises the plastic return-mapping branch with more than one
        yielding point, using a nonlinear hardening law whose slope
        ``sigma_f_prime(q)`` returns one value per point. The algorithmic
        tangent must broadcast that per-point slope correctly over the batch.
        """
        n = N_ELEM

        # Nonlinear (saturating) hardening: sigma_f_prime varies per point
        def sigma_f_nl(q):
            return 200.0 + 50.0 * q + 100.0 * (1.0 - torch.exp(-10.0 * q))

        def sigma_f_prime_nl(q):
            return 50.0 + 1000.0 * torch.exp(-10.0 * q)

        mat = IsotropicPlasticityPlaneStress(
            210e3, 0.3, sigma_f_nl, sigma_f_prime_nl
        ).vectorize(n)
        _, F, stress, state, de0, cl = _make_step_args_2d(n, n_state=1)
        # Uniform strain increment that yields every point
        H_inc = torch.zeros(n, 2, 2)
        H_inc[:, 0, 0] = 0.01
        H_inc[:, 1, 1] = 0.002

        s_new, st_new, C = mat.step(H_inc, F, stress, state, de0, cl, 1)
        assert torch.isfinite(s_new).all() and torch.isfinite(C).all()
        # Plastic strain has accumulated at every point
        assert (st_new[:, 0] > 0.0).all()

        # Consistent tangent vs. central finite difference (Voigt)
        Cv = stiffness2voigt(C)
        fd = torch.zeros(n, 3, 3)
        eps = 1e-8
        for j, (a, b) in enumerate([(0, 0), (1, 1), (0, 1)]):
            dp, dm = H_inc.clone(), H_inc.clone()
            dp[:, a, b] += eps
            dm[:, a, b] -= eps
            if a != b:
                dp[:, b, a] += eps
                dm[:, b, a] -= eps
            sp, _, _ = mat.step(dp, F, stress, state, de0, cl, 1)
            sm, _, _ = mat.step(dm, F, stress, state, de0, cl, 1)
            col = (stress2voigt(sp) - stress2voigt(sm)) / (2 * eps)
            fd[:, :, j] = col if a == b else col / 2.0
        assert torch.allclose(Cv, fd, rtol=1e-4, atol=1e-3)


class TestIsotropicPlasticityPlaneStrain:
    def test_elastic_step(self):
        n = N_ELEM
        mat = IsotropicPlasticityPlaneStrain(
            210e3, 0.3, sigma_f, sigma_f_prime
        ).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_2d(n, n_state=2)
        H_inc[:] = 1e-7
        s_new, st_new, _ = mat.step(H_inc, F, stress, state, de0, cl, 0)
        assert s_new.shape == (n, 2, 2)
        assert torch.allclose(st_new[:, 0], torch.zeros_like(st_new[:, 0]), atol=1e-8)
        assert torch.isfinite(s_new).all()


class TestIsotropicDamage3D:
    def test_declares_an_unsymmetric_tangent(self):
        """Its rank-one term has two different factors, so CG cannot solve it."""

        def d(kappa, cl):
            return torch.clamp(1 - 0.01 / kappa, min=0.0)

        mat = IsotropicDamage3D(210e3, 0.3, d, lambda k, cl: 0.01 / k**2, "rankine")
        assert mat.symmetric_tangent is False
        # A class attribute, so batching the tensor properties leaves it alone
        assert mat.vectorize(N_ELEM).symmetric_tangent is False
        assert IsotropicElasticity3D(210e3, 0.3).symmetric_tangent is True

    def test_elastic_step(self):
        def d(kappa, cl):
            return torch.clamp(1 - 0.01 / kappa, min=0.0)

        def d_prime(kappa, cl):
            return 0.01 / kappa**2

        n = N_ELEM
        mat = IsotropicDamage3D(210e3, 0.3, d, d_prime, "rankine").vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_3d(n, n_state=2)
        H_inc.zero_()
        H_inc[:, 0, 0] = 1e-3
        H_inc[:, 1, 1] = 1e-3
        H_inc[:, 2, 2] = 1e-3
        s_new, st_new, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        assert s_new.shape == (n, 3, 3)
        assert st_new.shape == (n, 2)
        assert torch.isfinite(s_new).all()
        assert torch.isfinite(ddsdde).all()
        assert st_new[0, 1] >= state[0, 1]


def _orthotropic_3d() -> OrthotropicElasticity3D:
    return OrthotropicElasticity3D(
        E_1=100e3,
        E_2=10e3,
        E_3=10e3,
        nu_12=0.3,
        nu_13=0.3,
        nu_23=0.3,
        G_12=5e3,
        G_13=5e3,
        G_23=3e3,
    )


class TestTransverseIsotropicElasticity3D:
    def test_transverse_plane_is_isotropic(self):
        mat = TransverseIsotropicElasticity3D(100e3, 10e3, 0.3, 0.3, 5e3)
        assert torch.allclose(mat.C[1, 1, 1, 1], mat.C[2, 2, 2, 2])

    def test_vectorize_over_a_batch_of_constants(self):
        mat = TransverseIsotropicElasticity3D(
            torch.full((3,), 100e3),
            torch.full((3,), 10e3),
            torch.full((3,), 0.3),
            torch.full((3,), 0.3),
            torch.full((3,), 5e3),
        )
        assert mat.C.shape == (3, 3, 3, 3, 3)

    def test_rejects_an_inadmissible_longitudinal_shear(self):
        with pytest.raises(ValueError, match="G_L must be less than"):
            TransverseIsotropicElasticity3D(100e3, 10e3, 0.3, 0.3, 1e9)


TI = {"E_L": 100e3, "E_T": 10e3, "nu_L": 0.3, "nu_T": 0.25, "G_L": 5e3}
G_T = TI["E_T"] / (2 * (1 + TI["nu_T"]))


class TestTransverseIsotropicElasticityPlaneStress:
    def test_matches_the_orthotropic_equivalent(self):
        mat = TransverseIsotropicElasticityPlaneStress(**TI)
        ref = OrthotropicElasticityPlaneStress(
            TI["E_L"], TI["E_T"], TI["nu_L"], TI["G_L"], TI["G_L"], G_T
        )
        assert mat.C.shape == (2, 2, 2, 2)
        assert torch.allclose(mat.C, ref.C)

    def test_carries_the_transverse_shear_moduli_a_shell_needs(self):
        mat = TransverseIsotropicElasticityPlaneStress(**TI)
        assert torch.allclose(mat.G_13, torch.tensor(TI["G_L"]))
        assert torch.allclose(mat.G_23, torch.tensor(G_T))


class TestTransverseIsotropicElasticityPlaneStrain:
    def test_matches_the_orthotropic_equivalent(self):
        mat = TransverseIsotropicElasticityPlaneStrain(**TI)
        ref = OrthotropicElasticityPlaneStrain(
            TI["E_L"],
            TI["E_T"],
            TI["E_T"],
            TI["nu_L"],
            TI["nu_L"],
            TI["nu_T"],
            TI["G_L"],
            TI["G_L"],
            G_T,
        )
        assert mat.C.shape == (2, 2, 2, 2)
        assert torch.allclose(mat.C, ref.C)

    def test_is_the_in_plane_block_of_the_3d_material(self):
        mat = TransverseIsotropicElasticityPlaneStrain(**TI)
        full = TransverseIsotropicElasticity3D(**TI)
        assert torch.allclose(mat.C, full.C[:2, :2, :2, :2])


def _plane_stress():
    return OrthotropicElasticityPlaneStress(E_1=100e3, E_2=10e3, nu_12=0.3, G_12=5e3)


def _plane_strain():
    return OrthotropicElasticityPlaneStrain(
        E_1=100e3, E_2=10e3, E_3=10e3, nu_12=0.3, nu_13=0.3, nu_23=0.3, G_12=5e3
    )


class TestOrthotropicElasticityPlaneStress:
    def test_identity_rotation_recovers_input_constants(self):
        """The engineering constants are re-extracted from the compliance, so
        rotating by the identity must return exactly what was passed in."""
        mat = _plane_stress().rotate(torch.eye(2))
        assert torch.allclose(mat.E_1, torch.tensor(100e3), rtol=1e-5)
        assert torch.allclose(mat.E_2, torch.tensor(10e3), rtol=1e-5)
        assert torch.allclose(mat.nu_12, torch.tensor(0.3), rtol=1e-5)
        assert torch.allclose(mat.G_12, torch.tensor(5e3), rtol=1e-5)

    def test_rotation_by_90_deg_swaps_axes(self):
        mat = _plane_stress().rotate(planar_rotation(torch.pi / 2))
        assert torch.allclose(mat.E_1, torch.tensor(10e3), rtol=1e-5)
        assert torch.allclose(mat.E_2, torch.tensor(100e3), rtol=1e-5)
        assert torch.allclose(mat.G_12, torch.tensor(5e3), rtol=1e-5)


class TestOrthotropicElasticityPlaneStrain:
    def test_rotation_by_90_deg_swaps_axes(self):
        """Plane strain constrains eps_33, so the extracted constants differ from
        the input; compare against the unrotated material instead."""
        ref = _plane_strain().rotate(torch.eye(2))
        rot = _plane_strain().rotate(planar_rotation(torch.pi / 2))
        assert torch.allclose(rot.E_1, ref.E_2, rtol=1e-5)
        assert torch.allclose(rot.E_2, ref.E_1, rtol=1e-5)
        assert torch.allclose(rot.G_12, ref.G_12, rtol=1e-5)


class TestHyperelasticPlaneStress:
    @staticmethod
    def neo_hookean_3d(F, params):
        mu, lam = params[0], params[1]
        J = torch.linalg.det(F)
        C = F.T @ F
        return (
            0.5 * mu * (torch.trace(C) - 3)
            - mu * torch.log(J)
            + 0.5 * lam * torch.log(J) ** 2
        )

    def test_step(self):
        n = N_ELEM
        mat = HyperelasticPlaneStress(
            self.neo_hookean_3d, torch.tensor([80.0, 120.0])
        ).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_2d(n, n_state=1)
        H_inc.zero_()
        s_new, _, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        assert s_new.shape == (n, 2, 2)
        assert ddsdde.shape == (n, 2, 2, 2, 2)
        assert torch.allclose(s_new, torch.zeros_like(s_new), atol=1e-8)


class TestOrthotropicConductivity3D:
    def test_rotation_about_z_swaps_in_plane_axes(self):
        mat = OrthotropicConductivity3D(1.0, 2.0, 3.0)
        rot = mat.rotate(axis_rotation(torch.tensor([0.0, 0.0, 1.0]), torch.pi / 2))
        expected = torch.diag(torch.tensor([2.0, 1.0, 3.0]))
        assert torch.allclose(rot.KAPPA, expected, atol=1e-6)


class TestOrthotropicConductivity2D:
    def test_rotation_by_90_deg_swaps_axes(self):
        mat = OrthotropicConductivity2D(1.0, 2.0)
        rot = mat.rotate(planar_rotation(torch.pi / 2))
        assert torch.allclose(
            rot.KAPPA, torch.diag(torch.tensor([2.0, 1.0])), atol=1e-6
        )


ANISOTROPIC = [
    lambda: OrthotropicElasticity3D(100e3, 10e3, 5e3, 0.3, 0.2, 0.1, 4e3, 3e3, 2e3),
    lambda: OrthotropicElasticityPlaneStress(100e3, 10e3, 0.3, 5e3),
    lambda: OrthotropicElasticityPlaneStrain(100e3, 10e3, 5e3, 0.3, 0.2, 0.1, 4e3),
    lambda: OrthotropicConductivity3D(1.0, 2.0, 3.0),
    lambda: OrthotropicConductivity2D(1.0, 2.0),
]


def _anisotropy(mat):
    """The tensor a rotation acts on."""
    return mat.C if hasattr(mat, "C") else mat.KAPPA


def _rotation_for(mat):
    if _anisotropy(mat).shape[-1] == 3:
        return axis_rotation(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.3))
    return planar_rotation(torch.tensor(0.3))


@pytest.mark.parametrize("build", ANISOTROPIC)
def test_rotate_rejects_a_matrix_of_the_wrong_size(build):
    mat = build()
    dim = _anisotropy(mat).shape[-1]
    with pytest.raises(ValueError, match=f"{dim}x{dim}"):
        mat.rotate(torch.eye(5 - dim))


@pytest.mark.parametrize("build", ANISOTROPIC)
def test_rotate_leaves_the_material_unchanged(build):
    mat = build()
    R = _rotation_for(mat)
    before = {k: v.clone() for k, v in vars(mat).items() if isinstance(v, torch.Tensor)}

    assert mat.rotate(R) is not mat
    for key, value in before.items():
        assert torch.equal(getattr(mat, key), value), f"rotate() modified {key}"


@pytest.mark.parametrize("build", ANISOTROPIC)
def test_rotation_does_not_accumulate(build):
    mat = build().vectorize(N_ELEM)
    R = _rotation_for(mat)
    once = _anisotropy(mat.rotate(R))
    for _ in range(3):
        assert torch.allclose(_anisotropy(mat.rotate(R)), once)


@pytest.mark.parametrize("build", ANISOTROPIC)
def test_vectorize_keeps_a_rotation(build):
    # vectorize() batches the tensors a material holds, so it carries a rotated
    # one over rather than rebuilding it from the engineering constants.
    mat = build()
    R = _rotation_for(mat)
    rotated = mat.rotate(R)
    assert torch.allclose(
        _anisotropy(rotated.vectorize(N_ELEM))[0], _anisotropy(rotated)
    )


@pytest.mark.parametrize("build", ANISOTROPIC)
def test_rotation_commutes_with_vectorization(build):
    mat = build()
    R = _rotation_for(mat)
    assert torch.allclose(
        _anisotropy(mat.rotate(R).vectorize(N_ELEM)),
        _anisotropy(mat.vectorize(N_ELEM).rotate(R)),
    )


class TestMaterialBases:
    """Every material derives from the base of the balance law it closes."""

    def test_every_exported_material_picks_a_physics(self):
        m = torchfem.materials
        undecided = [
            name
            for name in m.__all__
            if isinstance(getattr(m, name), type)
            and issubclass(getattr(m, name), m.Material)
            and name not in {"Material", "MechanicsMaterial", "HeatMaterial"}
            and not issubclass(getattr(m, name), (m.MechanicsMaterial, m.HeatMaterial))
        ]
        assert undecided == []

    @pytest.mark.parametrize(
        "material",
        [
            IsotropicElasticity3D(1000.0, 0.3),
            IsotropicElasticity1D(1000.0),
            OrthotropicElasticity3D(1e3, 5e2, 5e2, 0.3, 0.3, 0.3, 3e2, 3e2, 3e2),
            Hyperelastic3D(lambda F, p: (F * F).sum(), [1.0]),
            IsotropicPlasticity3D(1000.0, 0.3, sigma_f, sigma_f_prime),
            IsotropicDamage3D(1000.0, 0.3, lambda k, cl: k, lambda k, cl: k, "rankine"),
        ],
    )
    def test_mechanics_materials(self, material):
        assert isinstance(material, MechanicsMaterial)
        assert not isinstance(material, HeatMaterial)

    @pytest.mark.parametrize(
        "material",
        [
            IsotropicConductivity3D(400.0),
            IsotropicConductivity2D(400.0),
            OrthotropicConductivity3D(1.0, 2.0, 3.0),
            OrthotropicConductivity2D(1.0, 2.0),
        ],
    )
    def test_heat_materials(self, material):
        assert isinstance(material, HeatMaterial)
        assert not isinstance(material, MechanicsMaterial)


def _damage_law(eps_0=8.0e-4, d_max=0.3):
    """A bounded, Lipschitz damage law, free of the characteristic length."""

    def d(kappa, cl):
        return d_max * torch.clamp((kappa - eps_0) / eps_0, 0.0, 1.0)

    def d_prime(kappa, cl):
        inside = (kappa > eps_0) & (kappa < 2 * eps_0)
        return torch.where(inside, torch.full_like(kappa, d_max / eps_0), 0.0 * kappa)

    return d, d_prime


class TestIsotropicDamagePlaneStrain:
    """Plane strain damage is the 3D model with a vanishing out-of-plane strain."""

    def _step(self, mat, eps, dim, n=3):
        F = torch.eye(dim).expand(n, dim, dim).contiguous()
        return mat.step(
            eps,
            F,
            torch.zeros(n, dim, dim),
            torch.zeros(n, 2),
            torch.zeros(n, dim, dim),
            torch.full((n,), 5.0),
            1,
        )

    @pytest.mark.parametrize("nu", [0.05, 0.3, 0.49])
    def test_matches_the_3d_model_over_random_strains(self, nu):
        """Unconditionally, for any strain state: the vanishing out-of-plane
        principal never wins the magnitude comparison and adds no tangent term."""
        d, d_prime = _damage_law()
        n = 200
        m2 = IsotropicDamagePlaneStrain(6000.0, nu, d, d_prime, "rankine").vectorize(n)
        m3 = IsotropicDamage3D(6000.0, nu, d, d_prime, "rankine").vectorize(n)
        torch.manual_seed(0)
        e = (torch.rand(n, 2, 2) - 0.5) * 8.0e-3  # tension, compression and shear
        e2 = 0.5 * (e + e.transpose(-1, -2))
        e3 = torch.zeros(n, 3, 3)
        e3[:, :2, :2] = e2
        s2, st2, t2 = self._step(m2, e2, 2, n=n)
        s3, st3, t3 = self._step(m3, e3, 3, n=n)
        driving = torch.linalg.eigvalsh(e2).abs().argmax(-1)
        assert (driving == 0).any() and (driving == 1).any()  # both signs represented
        assert (st2[:, 1] > 0).any()
        assert torch.equal(st2[:, 0], st3[:, 0])  # equivalent strain, bit for bit
        assert torch.allclose(s2, s3[:, :2, :2])
        assert torch.allclose(t2, t3[:, :2, :2, :2, :2])

    def test_matches_the_3d_model_restricted_to_the_plane(self):
        d, d_prime = _damage_law()
        n = 3
        m2 = IsotropicDamagePlaneStrain(6000.0, 0.3, d, d_prime, "rankine").vectorize(n)
        m3 = IsotropicDamage3D(6000.0, 0.3, d, d_prime, "rankine").vectorize(n)
        e2 = torch.zeros(n, 2, 2)
        e2[:, 0, 0], e2[:, 1, 1] = 2.0e-3, 6.0e-4
        e2[:, 0, 1] = e2[:, 1, 0] = 4.0e-4
        e3 = torch.zeros(n, 3, 3)
        e3[:, :2, :2] = e2  # eps_zz = 0
        s2, st2, t2 = self._step(m2, e2, 2)
        s3, st3, t3 = self._step(m3, e3, 3)
        assert (st2[:, 1] > 0).all()  # damage is actually active
        assert torch.allclose(st2, st3)
        assert torch.allclose(s2, s3[:, :2, :2])
        assert torch.allclose(t2, t3[:, :2, :2, :2, :2])

    def test_tangent_matches_a_numerical_jacobian(self):
        """The rank-one softening term is what this pins down."""
        d, d_prime = _damage_law()
        mat = IsotropicDamagePlaneStrain(6000.0, 0.3, d, d_prime, "rankine").vectorize(
            1
        )
        eps = torch.zeros(1, 2, 2)
        eps[:, 0, 0], eps[:, 1, 1] = 2.0e-3, 6.0e-4
        eps[:, 0, 1] = eps[:, 1, 0] = 4.0e-4
        _, state, tangent = self._step(mat, eps, 2, n=1)
        assert state[0, 1] > 0

        def stress_of(e):
            return self._step(mat, e, 2, n=1)[0]

        numerical = torch.autograd.functional.jacobian(stress_of, eps).reshape(
            2, 2, 2, 2
        )
        assert torch.allclose(tangent[0], numerical, rtol=1e-6, atol=1e-8)


class TestIsotropicDamagePlaneStress:
    """The out-of-plane strain follows the in-plane one, so it both drives the
    damage and contributes to the tangent."""

    def _step(self, mat, eps, n=1):
        return mat.step(
            eps,
            torch.eye(2).expand(n, 2, 2).contiguous(),
            torch.zeros(n, 2, 2),
            torch.zeros(n, 2),
            torch.zeros(n, 2, 2),
            torch.full((n,), 5.0),
            1,
        )

    def _material(self, nu):
        d, d_prime = _damage_law()
        return IsotropicDamagePlaneStress(6000.0, nu, d, d_prime, "rankine").vectorize(
            1
        )

    @staticmethod
    def _strain(e11, e22, e12=0.0):
        eps = torch.zeros(1, 2, 2)
        eps[:, 0, 0], eps[:, 1, 1] = e11, e22
        eps[:, 0, 1] = eps[:, 1, 0] = e12
        return eps

    @pytest.mark.parametrize(
        ("nu", "e11", "e22", "out_of_plane"),
        [
            (0.30, 1.2e-3, 0.0, False),  # in-plane drives, n_3 = 0
            (0.30, 1.0e-3, 4.0e-4, False),
            (0.40, -1.0e-3, -1.0e-3, True),  # out-of-plane drives, n_3 = 1
            (0.45, -8.0e-4, -8.0e-4, True),
        ],
    )
    def test_tangent_matches_a_numerical_jacobian(self, nu, e11, e22, out_of_plane):
        mat = self._material(nu)
        eps = self._strain(e11, e22)
        _, state, tangent = self._step(mat, eps)
        kappa = state[0, 0]
        assert mat.d_prime(state[:, 0], None)[0] > 0  # softening term is active
        eps_33 = -nu / (1 - nu) * (e11 + e22)
        assert torch.isclose(kappa, torch.tensor(eps_33)) == out_of_plane

        numerical = torch.autograd.functional.jacobian(
            lambda x: self._step(mat, x)[0], eps
        ).reshape(2, 2, 2, 2)
        assert torch.allclose(tangent[0], numerical, rtol=1e-7, atol=1e-10)

    def test_equivalent_strain_and_stress_match_the_3d_model(self):
        """Fed the same out-of-plane strain, the 3D model must agree."""
        d, d_prime = _damage_law()
        nu = 0.3
        m2 = IsotropicDamagePlaneStress(6000.0, nu, d, d_prime, "rankine").vectorize(1)
        m3 = IsotropicDamage3D(6000.0, nu, d, d_prime, "rankine").vectorize(1)
        e2 = self._strain(2.0e-3, 5.0e-4)
        e3 = torch.zeros(1, 3, 3)
        e3[:, :2, :2] = e2
        e3[:, 2, 2] = -nu / (1 - nu) * (e2[0, 0, 0] + e2[0, 1, 1])
        s2, st2, _ = self._step(m2, e2)
        s3, st3, _ = m3.step(
            e3,
            torch.eye(3).expand(1, 3, 3).contiguous(),
            torch.zeros(1, 3, 3),
            torch.zeros(1, 2),
            torch.zeros(1, 3, 3),
            torch.full((1,), 5.0),
            1,
        )
        assert st2[0, 1] > 0
        assert torch.equal(st2[:, 0], st3[:, 0])
        assert torch.allclose(s2, s3[:, :2, :2])

    def test_out_of_plane_strain_can_drive_the_damage(self):
        """Inheriting the 3D step alone would miss this and silently mis-drive."""
        mat = self._material(0.45)
        eps = self._strain(-8.0e-4, -8.0e-4)
        _, state, _ = self._step(mat, eps)
        in_plane_max = torch.linalg.eigvalsh(eps[0]).abs().max()
        assert state[0, 0] > in_plane_max


class TestIsotropicDamage1D:
    """The single strain is the only principal strain, so it always drives."""

    def _step(self, mat, eps, n):
        return mat.step(
            eps,
            torch.eye(1).expand(n, 1, 1).contiguous(),
            torch.zeros(n, 1, 1),
            torch.zeros(n, 2),
            torch.zeros(n, 1, 1),
            torch.full((n,), 5.0),
            1,
        )

    def test_builds_the_1d_stiffness_not_the_3d_one(self):
        """`IsotropicElasticity1D` is a sibling of the 3D class, so the
        constructor cannot go through `super()`."""
        d, d_prime = _damage_law()
        mat = IsotropicDamage1D(1000.0, d, d_prime, "rankine").vectorize(3)
        assert mat.dim == 1
        assert mat.C.shape == (3, 1, 1, 1, 1)
        assert torch.allclose(mat.C, torch.full((3, 1, 1, 1, 1), 1000.0))

    def test_degrades_the_stress_and_matches_a_numerical_jacobian(self):
        d, d_prime = _damage_law()
        E, n = 1000.0, 4
        mat = IsotropicDamage1D(E, d, d_prime, "rankine").vectorize(n)
        eps = torch.zeros(n, 1, 1)
        eps[:, 0, 0] = torch.tensor([5.0e-4, 1.2e-3, -1.2e-3, 2.0e-3])
        stress, state, tangent = self._step(mat, eps, n)
        assert torch.allclose(stress[:, 0, 0], (1 - state[:, 1]) * E * eps[:, 0, 0])
        assert state[1, 1] > 0 and state[3, 1] > 0  # damage is active
        assert state[2, 1] == 0  # a bar in compression never damages
        numerical = torch.autograd.functional.jacobian(
            lambda x: self._step(mat, x, n)[0], eps
        )
        diagonal = torch.stack([numerical[i, 0, 0, i, 0, 0] for i in range(n)])
        assert torch.allclose(tangent[:, 0, 0, 0, 0], diagonal)
