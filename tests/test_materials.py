import torch

from torchfem.materials import (
    Hyperelastic3D,
    HyperelasticPlaneStress,
    IsotropicDamage3D,
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStrain,
    IsotropicElasticityPlaneStress,
    IsotropicPlasticity1D,
    IsotropicPlasticity3D,
    IsotropicPlasticityPlaneStrain,
    IsotropicPlasticityPlaneStress,
    OrthotropicElasticity3D,
    OrthotropicElasticityPlaneStress,
    TransverseIsotropicElasticity3D,
)

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

    def test_is_orthotropic_subclass(self):
        mat = TransverseIsotropicElasticity3D(
            E_L=100e3,
            E_T=10e3,
            nu_L=0.3,
            nu_T=0.3,
            G_L=5e3,
        )
        assert isinstance(mat, OrthotropicElasticity3D)


class TestOrthotropicElasticityPlaneStress:
    def test_stiffness_shape(self):
        mat = OrthotropicElasticityPlaneStress(
            E_1=100e3,
            E_2=10e3,
            nu_12=0.3,
            G_12=5e3,
        )
        assert mat.C.shape == (2, 2, 2, 2)

    def test_step(self):
        n = N_ELEM
        mat = OrthotropicElasticityPlaneStress(
            E_1=100e3,
            E_2=10e3,
            nu_12=0.3,
            G_12=5e3,
        ).vectorize(n)
        H_inc, F, stress, state, de0, cl = _make_step_args_2d(n)
        s_new, _, _ = mat.step(H_inc, F, stress, state, de0, cl, 0)
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        expected = torch.einsum("...ijkl,...kl->...ij", mat.C, de)
        assert s_new.shape == (n, 2, 2)
        assert torch.allclose(s_new, expected, atol=1e-10, rtol=1e-10)


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
