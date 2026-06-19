import torch

from torchfem.failure import PlyStrength, hashin, rotate_stress_to_ply, tsai_hill

STRENGTH = PlyStrength(Xt=2200.0, Xc=1400.0, Yt=70.0, Yc=240.0, S12=90.0, S23=60.0)


def t(x):
    return torch.tensor(x)


def test_tsai_hill_uniaxial_fiber():
    # s11 = Xt alone -> FI = 1
    fi = tsai_hill(t(STRENGTH.Xt), t(0.0), t(0.0), STRENGTH)
    assert torch.isclose(fi, t(1.0))


def test_tsai_hill_pure_shear():
    fi = tsai_hill(t(0.0), t(0.0), t(STRENGTH.S12), STRENGTH)
    assert torch.isclose(fi, t(1.0))


def test_tsai_hill_compression_uses_Xc():
    # s11 = -Xc alone -> FI = 1 (compressive strength selected)
    fi = tsai_hill(t(-STRENGTH.Xc), t(0.0), t(0.0), STRENGTH)
    assert torch.isclose(fi, t(1.0))


def test_tsai_hill_safe_below_one():
    fi = tsai_hill(t(0.5 * STRENGTH.Xt), t(0.0), t(0.0), STRENGTH)
    assert fi < 1.0


def test_hashin_fiber_tension():
    h = hashin(t(STRENGTH.Xt), t(0.0), t(0.0), STRENGTH)
    assert torch.isclose(h["fiber_tension"], t(1.0))
    assert torch.isclose(h["fiber"], t(1.0))


def test_hashin_fiber_compression():
    h = hashin(t(-STRENGTH.Xc), t(0.0), t(0.0), STRENGTH)
    assert torch.isclose(h["fiber_compression"], t(1.0))
    assert torch.isclose(h["fiber"], t(1.0))


def test_hashin_matrix_tension():
    h = hashin(t(0.0), t(STRENGTH.Yt), t(0.0), STRENGTH)
    assert torch.isclose(h["matrix_tension"], t(1.0))
    assert torch.isclose(h["matrix"], t(1.0))


def test_hashin_matrix_compression():
    # At s22 = -Yc the matrix-compression index must be >= 1 (failure)
    h = hashin(t(0.0), t(-STRENGTH.Yc), t(0.0), STRENGTH)
    assert h["matrix_compression"] >= 1.0
    assert torch.isclose(h["matrix"], h["matrix_compression"])


def test_hashin_envelope_is_max():
    s11, s22, t12 = t(500.0), t(-50.0), t(30.0)
    h = hashin(s11, s22, t12, STRENGTH)
    assert torch.isclose(h["max"], torch.maximum(h["fiber"], h["matrix"]))


def test_rotate_stress_to_ply_90deg():
    # Element uniaxial sigma_xx -> 90 deg ply sees it as transverse (s22)
    sig = torch.tensor([[100.0, 0.0], [0.0, 0.0]])
    sig_mat = rotate_stress_to_ply(sig, torch.pi / 2)
    assert torch.isclose(sig_mat[0, 0], t(0.0), atol=1e-9)
    assert torch.isclose(sig_mat[1, 1], t(100.0))


def test_batched_inputs():
    s11 = torch.tensor([2200.0, 1100.0, 0.0])
    s22 = torch.zeros(3)
    t12 = torch.zeros(3)
    fi = tsai_hill(s11, s22, t12, STRENGTH)
    assert fi.shape == (3,)
    assert torch.isclose(fi[0], t(1.0))
