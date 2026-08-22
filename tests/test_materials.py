import pytest
import torch

from torchfem.materials import (
    Hyperelastic3D,
    HyperelasticPlaneStress,
    IsotropicConductivity1D,
    IsotropicConductivity2D,
    IsotropicConductivity3D,
    IsotropicDamage3D,
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStrain,
    IsotropicElasticityPlaneStress,
    IsotropicPlasticity1D,
    IsotropicPlasticity3D,
    IsotropicPlasticityPlaneStrain,
    IsotropicPlasticityPlaneStress,
    OrthotropicConductivity2D,
    OrthotropicConductivity3D,
    OrthotropicElasticity3D,
    OrthotropicElasticityPlaneStrain,
    OrthotropicElasticityPlaneStress,
    TransverseIsotropicElasticity3D,
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
    F = torch.zeros(n_elem, 1, dim)
    heat_flux = torch.zeros(n_elem, 1, dim)
    state = torch.zeros(n_elem, 0)
    de0 = torch.zeros(n_elem, 1, dim)
    cl = torch.ones(n_elem)
    return grad_inc, F, heat_flux, state, de0, cl


# Common yield function for plasticity tests
def sigma_f(ep):
    return 200.0 + 50.0 * ep


def sigma_f_prime(ep):
    return torch.tensor(50.0)


class TestIsotropicElasticity3D:
    def test_stiffness_symmetry(self):
        mat = IsotropicElasticity3D(1000.0, 0.3)
        C = mat.C
        # Major symmetry: C_ijkl = C_klij
        assert torch.allclose(C, C.permute(2, 3, 0, 1), atol=1e-10)
        # Minor symmetry: C_ijkl = C_jikl
        assert torch.allclose(C, C.permute(1, 0, 2, 3), atol=1e-10)

    def test_lame_parameters(self):
        E, nu = 210e3, 0.3
        mat = IsotropicElasticity3D(E, nu)
        lbd_expected = E * nu / ((1 + nu) * (1 - 2 * nu))
        G_expected = E / (2 * (1 + nu))
        assert torch.allclose(mat.lbd, torch.tensor(lbd_expected))
        assert torch.allclose(mat.G, torch.tensor(G_expected))

    def test_step_linear(self):
        n = N_ELEM
        mat = IsotropicElasticity3D(1000.0, 0.3).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_3d(n)
        s_new, st_new, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, de)
        assert s_new.shape == (n, 3, 3)
        assert ddsdde.shape == (n, 3, 3, 3, 3)
        assert torch.allclose(s_new, expected, atol=1e-12, rtol=1e-10)
        assert torch.allclose(st_new, state)
        assert torch.isfinite(ddsdde).all()

    def test_vectorize(self):
        mat = IsotropicElasticity3D(1000.0, 0.3)
        mat_v = mat.vectorize(N_ELEM)
        assert mat_v.E.shape == (N_ELEM,)
        assert mat_v.nu.shape == (N_ELEM,)
        assert mat_v.C.shape == (N_ELEM, 3, 3, 3, 3)

    def test_vectorize_idempotent(self):
        mat = IsotropicElasticity3D(1000.0, 0.3).vectorize(N_ELEM)
        mat2 = mat.vectorize(N_ELEM)
        assert mat2 is mat


class TestIsotropicElasticityPlaneStress:
    def test_stiffness_shape(self):
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3)
        assert mat.C.shape == (2, 2, 2, 2)

    def test_step(self):
        n = N_ELEM
        mat = IsotropicElasticityPlaneStress(1000.0, 0.3).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_2d(n)
        s_new, _, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, de)
        assert s_new.shape == (n, 2, 2)
        assert ddsdde.shape == (n, 2, 2, 2, 2)
        assert torch.allclose(s_new, expected, atol=1e-12, rtol=1e-10)
        assert torch.isfinite(ddsdde).all()


class TestIsotropicElasticityPlaneStrain:
    def test_stiffness_shape(self):
        mat = IsotropicElasticityPlaneStrain(1000.0, 0.3)
        assert mat.C.shape == (2, 2, 2, 2)

    def test_step(self):
        n = N_ELEM
        mat = IsotropicElasticityPlaneStrain(1000.0, 0.3).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_2d(n)
        s_new, _, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, de)
        assert s_new.shape == (n, 2, 2)
        assert ddsdde.shape == (n, 2, 2, 2, 2)
        assert torch.allclose(s_new, expected, atol=1e-12, rtol=1e-10)
        assert torch.isfinite(ddsdde).all()


class TestIsotropicElasticity1D:
    def test_stiffness_shape(self):
        mat = IsotropicElasticity1D(1000.0)
        assert mat.C.shape == (1, 1, 1, 1)

    def test_step(self):
        n = N_ELEM
        mat = IsotropicElasticity1D(1000.0).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_1d(n)
        s_new, _, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, H_inc)
        assert s_new.shape == (n, 1, 1)
        assert ddsdde.shape == (n, 1, 1, 1, 1)
        assert torch.allclose(s_new, expected, atol=1e-12, rtol=1e-10)
        assert torch.isfinite(ddsdde).all()

    def test_vectorize(self):
        mat = IsotropicElasticity1D(500.0)
        mat_v = mat.vectorize(N_ELEM)
        assert mat_v.E.shape == (N_ELEM,)


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


class TestOrthotropicElasticity3D:
    def test_stiffness_shape(self):
        mat = OrthotropicElasticity3D(
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
        assert mat.C.shape == (3, 3, 3, 3)

    def test_stiffness_symmetry(self):
        mat = OrthotropicElasticity3D(
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
        C = mat.C
        assert torch.allclose(C, C.permute(2, 3, 0, 1), atol=1e-6)

    def test_step(self):
        n = N_ELEM
        mat = OrthotropicElasticity3D(
            E_1=100e3,
            E_2=10e3,
            E_3=10e3,
            nu_12=0.3,
            nu_13=0.3,
            nu_23=0.3,
            G_12=5e3,
            G_13=5e3,
            G_23=3e3,
        ).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_3d(n)
        s_new, _, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, de)
        assert s_new.shape == (n, 3, 3)
        assert ddsdde.shape == (n, 3, 3, 3, 3)
        assert torch.allclose(s_new, expected, atol=1e-10, rtol=1e-10)
        assert torch.isfinite(ddsdde).all()

    def test_vectorize(self):
        mat = OrthotropicElasticity3D(
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
        mat_v = mat.vectorize(N_ELEM)
        assert mat_v.C.shape == (N_ELEM, 3, 3, 3, 3)


class TestTransverseIsotropicElasticity3D:
    def test_stiffness_shape(self):
        mat = TransverseIsotropicElasticity3D(
            E_L=100e3,
            E_T=10e3,
            nu_L=0.3,
            nu_T=0.3,
            G_L=5e3,
        )
        assert mat.C.shape == (3, 3, 3, 3)


def _plane_stress():
    return OrthotropicElasticityPlaneStress(E_1=100e3, E_2=10e3, nu_12=0.3, G_12=5e3)


def _plane_strain():
    return OrthotropicElasticityPlaneStrain(
        E_1=100e3, E_2=10e3, E_3=10e3, nu_12=0.3, nu_13=0.3, nu_23=0.3, G_12=5e3
    )


class TestOrthotropicElasticityPlaneStress:
    def test_stiffness_shape(self):
        assert _plane_stress().C.shape == (2, 2, 2, 2)

    def test_step(self):
        n = N_ELEM
        mat = _plane_stress().vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_2d(n)
        s_new, _, _ = mat.step(H_inc, F, stress, state, de0, cl, 0)
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, de)
        assert s_new.shape == (n, 2, 2)
        assert torch.allclose(s_new, expected, atol=1e-10, rtol=1e-10)

    def test_vectorize(self):
        mat = _plane_stress().vectorize(N_ELEM)
        assert mat.C.shape == (N_ELEM, 2, 2, 2, 2)
        assert mat.E_1.shape == (N_ELEM,)

    def test_vectorize_idempotent(self):
        mat = _plane_stress().vectorize(N_ELEM)
        assert mat.vectorize(N_ELEM) is mat

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

    def test_rotate_rejects_non_2x2_matrix(self):
        with pytest.raises(ValueError, match="2x2"):
            _plane_stress().rotate(torch.eye(3))


class TestOrthotropicElasticityPlaneStrain:
    def test_stiffness_shape(self):
        assert _plane_strain().C.shape == (2, 2, 2, 2)

    def test_step(self):
        n = N_ELEM
        mat = _plane_strain().vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_2d(n)
        s_new, _, ddsdde = mat.step(H_inc, F, stress, state, de0, cl, 0)
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, de)
        assert s_new.shape == (n, 2, 2)
        assert ddsdde.shape == (n, 2, 2, 2, 2)
        assert torch.allclose(s_new, expected, atol=1e-10, rtol=1e-10)

    def test_vectorize(self):
        mat = _plane_strain().vectorize(N_ELEM)
        assert mat.C.shape == (N_ELEM, 2, 2, 2, 2)
        assert mat.E_1.shape == (N_ELEM,)

    def test_vectorize_idempotent(self):
        mat = _plane_strain().vectorize(N_ELEM)
        assert mat.vectorize(N_ELEM) is mat

    def test_rotation_by_90_deg_swaps_axes(self):
        """Plane strain constrains eps_33, so the extracted constants differ from
        the input; compare against the unrotated material instead."""
        ref = _plane_strain().rotate(torch.eye(2))
        rot = _plane_strain().rotate(planar_rotation(torch.pi / 2))
        assert torch.allclose(rot.E_1, ref.E_2, rtol=1e-5)
        assert torch.allclose(rot.E_2, ref.E_1, rtol=1e-5)
        assert torch.allclose(rot.G_12, ref.G_12, rtol=1e-5)

    def test_rotate_rejects_non_2x2_matrix(self):
        with pytest.raises(ValueError, match="2x2"):
            _plane_strain().rotate(torch.eye(3))


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


class TestIsotropicConductivity3D:
    def test_conductivity_is_kappa_times_identity(self):
        assert torch.allclose(
            IsotropicConductivity3D(400.0).KAPPA, 400.0 * torch.eye(3)
        )

    def test_step_scales_temperature_gradient_by_kappa(self):
        n = N_ELEM
        mat = IsotropicConductivity3D(400.0).vectorize(n)
        grad_inc, F, q, state, de0, cl = _make_thermal_step_args(3, n)
        q_new, state_new, tangent = mat.step(grad_inc, F, q, state, de0, cl, 0)
        assert q_new.shape == (n, 1, 3)
        assert tangent.shape == (n, 3, 3)
        assert torch.allclose(q_new, 400.0 * grad_inc)
        assert torch.allclose(state_new, state)

    def test_vectorize(self):
        mat = IsotropicConductivity3D(400.0).vectorize(N_ELEM)
        assert mat.kappa.shape == (N_ELEM,)
        assert mat.KAPPA.shape == (N_ELEM, 3, 3)

    def test_vectorize_idempotent(self):
        mat = IsotropicConductivity3D(400.0).vectorize(N_ELEM)
        assert mat.vectorize(N_ELEM) is mat


class TestIsotropicConductivity2D:
    def test_conductivity_is_in_plane_block(self):
        assert torch.allclose(
            IsotropicConductivity2D(400.0).KAPPA, 400.0 * torch.eye(2)
        )

    def test_step_scales_temperature_gradient_by_kappa(self):
        n = N_ELEM
        mat = IsotropicConductivity2D(400.0).vectorize(n)
        grad_inc, F, q, state, de0, cl = _make_thermal_step_args(2, n)
        q_new, _, tangent = mat.step(grad_inc, F, q, state, de0, cl, 0)
        assert q_new.shape == (n, 1, 2)
        assert tangent.shape == (n, 2, 2)
        assert torch.allclose(q_new, 400.0 * grad_inc)

    def test_vectorize(self):
        mat = IsotropicConductivity2D(400.0).vectorize(N_ELEM)
        assert mat.KAPPA.shape == (N_ELEM, 2, 2)

    def test_vectorize_idempotent(self):
        mat = IsotropicConductivity2D(400.0).vectorize(N_ELEM)
        assert mat.vectorize(N_ELEM) is mat


class TestIsotropicConductivity1D:
    def test_conductivity_is_scalar_block(self):
        assert torch.allclose(
            IsotropicConductivity1D(400.0).KAPPA, 400.0 * torch.eye(1)
        )

    def test_step_scales_temperature_gradient_by_kappa(self):
        n = N_ELEM
        mat = IsotropicConductivity1D(400.0).vectorize(n)
        grad_inc, F, q, state, de0, cl = _make_thermal_step_args(1, n)
        q_new, _, tangent = mat.step(grad_inc, F, q, state, de0, cl, 0)
        assert q_new.shape == (n, 1, 1)
        assert tangent.shape == (n, 1, 1)
        assert torch.allclose(q_new, 400.0 * grad_inc)

    def test_vectorize(self):
        mat = IsotropicConductivity1D(400.0).vectorize(N_ELEM)
        assert mat.KAPPA.shape == (N_ELEM, 1, 1)

    def test_vectorize_idempotent(self):
        mat = IsotropicConductivity1D(400.0).vectorize(N_ELEM)
        assert mat.vectorize(N_ELEM) is mat


class TestOrthotropicConductivity3D:
    def test_conductivity_is_diagonal_in_principal_axes(self):
        mat = OrthotropicConductivity3D(1.0, 2.0, 3.0)
        assert torch.allclose(mat.KAPPA, torch.diag(torch.tensor([1.0, 2.0, 3.0])))

    def test_step_applies_conductivity_per_direction(self):
        n = N_ELEM
        mat = OrthotropicConductivity3D(1.0, 2.0, 3.0).vectorize(n)
        grad_inc, F, q, state, de0, cl = _make_thermal_step_args(3, n)
        q_new, _, _ = mat.step(grad_inc, F, q, state, de0, cl, 0)
        # Gradient [1, 2, 3] against conductivities [1, 2, 3].
        assert torch.allclose(q_new, torch.tensor([1.0, 4.0, 9.0]).expand(n, 1, 3))

    def test_rotation_about_z_swaps_in_plane_axes(self):
        mat = OrthotropicConductivity3D(1.0, 2.0, 3.0)
        rot = mat.rotate(axis_rotation(torch.tensor([0.0, 0.0, 1.0]), torch.pi / 2))
        expected = torch.diag(torch.tensor([2.0, 1.0, 3.0]))
        assert torch.allclose(rot.KAPPA, expected, atol=1e-6)

    def test_rotate_rejects_non_3x3_matrix(self):
        with pytest.raises(ValueError, match="3x3"):
            OrthotropicConductivity3D(1.0, 2.0, 3.0).rotate(torch.eye(2))

    def test_vectorize(self):
        mat = OrthotropicConductivity3D(1.0, 2.0, 3.0).vectorize(N_ELEM)
        assert mat.KAPPA.shape == (N_ELEM, 3, 3)

    def test_vectorize_idempotent(self):
        mat = OrthotropicConductivity3D(1.0, 2.0, 3.0).vectorize(N_ELEM)
        assert mat.vectorize(N_ELEM) is mat


class TestOrthotropicConductivity2D:
    def test_conductivity_is_diagonal_in_principal_axes(self):
        mat = OrthotropicConductivity2D(1.0, 2.0)
        assert torch.allclose(mat.KAPPA, torch.diag(torch.tensor([1.0, 2.0])))

    def test_step_applies_conductivity_per_direction(self):
        n = N_ELEM
        mat = OrthotropicConductivity2D(1.0, 2.0).vectorize(n)
        grad_inc, F, q, state, de0, cl = _make_thermal_step_args(2, n)
        q_new, _, _ = mat.step(grad_inc, F, q, state, de0, cl, 0)
        assert torch.allclose(q_new, torch.tensor([1.0, 4.0]).expand(n, 1, 2))

    def test_rotation_by_90_deg_swaps_axes(self):
        mat = OrthotropicConductivity2D(1.0, 2.0)
        rot = mat.rotate(planar_rotation(torch.pi / 2))
        assert torch.allclose(
            rot.KAPPA, torch.diag(torch.tensor([2.0, 1.0])), atol=1e-6
        )

    def test_rotate_rejects_non_2x2_matrix(self):
        with pytest.raises(ValueError, match="2x2"):
            OrthotropicConductivity2D(1.0, 2.0).rotate(torch.eye(3))

    def test_vectorize(self):
        mat = OrthotropicConductivity2D(1.0, 2.0).vectorize(N_ELEM)
        assert mat.KAPPA.shape == (N_ELEM, 2, 2)

    def test_vectorize_idempotent(self):
        mat = OrthotropicConductivity2D(1.0, 2.0).vectorize(N_ELEM)
        assert mat.vectorize(N_ELEM) is mat


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
