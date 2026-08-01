import pytest
import torch

from torchfem import Planar, Solid, base
from torchfem.materials import (
    HyperelasticPlaneStrain,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStress,
)
from torchfem.mesh import cube_hexa, rect_quad

# Uniform increments: a linear model assembles K once and reuses it, so the
# stabilization coefficient alpha/dt must stay constant over the load path.
INCREMENTS = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])


def _build_cantilever() -> Planar:
    """Planar cantilever, clamped at the west edge and loaded at the east edge."""
    material = IsotropicElasticityPlaneStress(E=1000.0, nu=0.3)
    model = Planar(*rect_quad(5, 3, 4.0, 2.0), material)
    west = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].min())
    east = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].max())
    model.constraints[west] = True
    model.forces[east, 1] = 1.0
    return model


def _dense(model: Planar, k: torch.Tensor) -> torch.Tensor:
    """Assemble element matrices into a dense global matrix."""
    K = torch.zeros(model.n_dofs, model.n_dofs)
    idx = model.idx.long()
    for e in range(model.n_elem):
        i = idx[e]
        K[i[:, None], i[None, :]] += k[e]
    return K


def test_stabilization_defaults_to_off():
    """The default solve must be unchanged by the stabilization feature."""
    reference = _build_cantilever().solve(increments=INCREMENTS)
    explicit = _build_cantilever().solve(increments=INCREMENTS, alpha=0.0)

    for a, b in zip(reference, explicit):
        assert torch.equal(a, b)


def test_stabilization_energy_is_zero_without_damping():
    model = _build_cantilever()
    model.solve(increments=INCREMENTS)

    assert torch.all(model.stabilization_energy == 0.0)


def test_stabilization_matches_abaqus_formulation():
    """Abaqus adds a viscous force alpha M du/dt and dissipates its work.

    This is a linear model, so `integrate_material` returns `k = None` once
    `self.K` is cached. The viscous tangent must not be added again in that
    case, since the cached matrix already carries it.
    """
    alpha = 2.0
    model = _build_cantilever()
    u, _, _, _, _ = model.solve(
        increments=INCREMENTS, alpha=alpha, return_intermediate=True
    )

    # Reference: incremental solves of (K + alpha/dt M) du = F_ext - K u_prev
    K = _dense(model, model.k0())
    M = _dense(model, model.integrate_mass())
    free = torch.nonzero(~model.constraints.ravel()).ravel()

    u_ref = torch.zeros(len(INCREMENTS), model.n_dofs)
    energy_ref = torch.zeros(len(INCREMENTS))
    for n in range(1, len(INCREMENTS)):
        c = alpha / float(INCREMENTS[n] - INCREMENTS[n - 1])
        A = (K + c * M)[free[:, None], free[None, :]]
        b = (INCREMENTS[n] * model.forces.ravel() - K @ u_ref[n - 1])[free]
        du = torch.zeros(model.n_dofs)
        du[free] = torch.linalg.solve(A, b)
        u_ref[n] = u_ref[n - 1] + du
        energy_ref[n] = energy_ref[n - 1] + c * du @ (M @ du)

    assert torch.allclose(u.reshape(len(INCREMENTS), -1), u_ref)
    assert torch.allclose(model.stabilization_energy, energy_ref)


def test_stabilization_forces_equilibrate_external_loads():
    """Returned nodal forces include the viscous part, so free DOFs stay in balance."""
    model = _build_cantilever()
    _, f, _, _, _ = model.solve(
        increments=INCREMENTS, alpha=2.0, return_intermediate=True
    )
    free = torch.nonzero(~model.constraints.ravel()).ravel()

    for n in range(1, len(INCREMENTS)):
        residual = f[n].ravel() - INCREMENTS[n] * model.forces.ravel()
        assert torch.allclose(residual[free], torch.zeros_like(residual[free]))


def test_stabilization_vanishes_for_small_damping():
    """Stabilization is a perturbation that disappears as alpha goes to zero."""
    u_ref, _, _, _, _ = _build_cantilever().solve(increments=INCREMENTS)

    errors = []
    for alpha in (1e-2, 1e-3, 1e-4):
        u, _, _, _, _ = _build_cantilever().solve(increments=INCREMENTS, alpha=alpha)
        errors.append(float((u - u_ref).abs().max()))

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-4 * float(u_ref.abs().max())


def test_stabilization_gradients_match_finite_differences():
    """The adjoint must differentiate through the viscous term."""
    nodes, elements = rect_quad(4, 3, 3.0, 2.0)
    material = IsotropicElasticityPlaneStress(E=1000.0, nu=0.3)

    def compliance(thickness: torch.Tensor) -> torch.Tensor:
        model = Planar(nodes, elements, material)
        model.thickness = thickness
        west = torch.isclose(nodes[:, 0], nodes[:, 0].min())
        east = torch.isclose(nodes[:, 0], nodes[:, 0].max())
        model.constraints[west] = True
        model.forces[east, 1] = 1.0
        u, _, _, _, _ = model.solve(
            increments=INCREMENTS,
            alpha=0.5,
            differentiable_parameters=thickness,
        )
        return torch.inner(model.forces.ravel(), u.ravel())

    thickness = torch.ones(len(elements), requires_grad=True)
    (grad,) = torch.autograd.grad(compliance(thickness), thickness)

    eps = 1e-6
    for e in (0, len(elements) // 2):
        perturbed = []
        for sign in (1.0, -1.0):
            t = torch.ones(len(elements))
            t[e] += sign * eps
            perturbed.append(compliance(t))
        fd = (perturbed[0] - perturbed[1]) / (2 * eps)
        assert torch.allclose(grad[e], fd, rtol=1e-5)


def _neo_hookean(F: torch.Tensor, params) -> torch.Tensor:
    G, D = params[0], params[1]
    C = F.transpose(-1, -2) @ F
    J = torch.exp(0.5 * torch.logdet(C))
    return G * (torch.trace(C * J ** (-2.0 / 3.0)) - 3.0) + 1 / D * (J - 1) ** 2


def _build_bent_cantilever() -> Planar:
    """Hyperelastic cantilever, loaded far past what one Newton pass can take."""
    material = HyperelasticPlaneStrain(_neo_hookean, params=[200.0, 1e-3])
    model = Planar(*rect_quad(5, 3, 4.0, 2.0), material)
    model.thickness = torch.ones(model.n_elem)
    west = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].min())
    east = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].max())
    model.constraints[west] = True
    model.forces[east, 1] = 100.0
    return model


def _count_newton_solves(monkeypatch) -> list:
    """Record one entry per substep actually solved."""
    calls = []
    original = base.newton_solve

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(base, "newton_solve", counting)
    return calls


def test_cutback_recovers_a_diverging_increment(monkeypatch):
    """A single increment Newton cannot take must be subdivided, not abandoned."""
    # Reference along a path fine enough that no increment is ever in trouble.
    reference = _build_bent_cantilever().solve(
        increments=torch.linspace(0, 1, 21), nlgeom=True
    )

    calls = _count_newton_solves(monkeypatch)
    cut_back = _build_bent_cantilever().solve(
        increments=torch.tensor([0.0, 1.0]), nlgeom=True
    )

    # One requested increment, so any solve beyond the first is a cutback.
    assert len(calls) > 1
    # The material is path independent, so both must reach the same state.
    assert torch.allclose(reference[0], cut_back[0], rtol=1e-6)


def test_a_recovered_substep_spans_the_next_increment(monkeypatch, capsys):
    """After a cutback the substep must grow back to one per increment.

    Growth applies to the size the solver asked for, not to the substep clipped
    to land on the requested increment. Growing from that clipped remainder
    would shrink the substep for good and subdivide every later increment.
    """
    original = base.newton_solve
    calls = []

    def failing_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("forced cutback")
        return original(*args, **kwargs)

    monkeypatch.setattr(base, "newton_solve", failing_once)
    _build_cantilever().solve(increments=torch.linspace(0, 1, 11), verbose=True)

    # Report rows, whose third column counts the substeps of each increment. The
    # substep ramps back up at `growth_factor` and then spans a whole increment;
    # growing from the clipped remainder instead never gets there.
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[7:-2]]
    assert rows[0][2] != "1", "the forced failure must subdivide increment 1"
    assert [row[2] for row in rows[-5:]] == ["1"] * 5


def test_max_cutbacks_bounds_the_retries(monkeypatch):
    """Forbidding cutbacks must surface the underlying convergence failure."""
    increments = torch.tensor([0.0, 1.0])

    with pytest.raises(RuntimeError, match="after 0 cutbacks"):
        _build_bent_cantilever().solve(
            increments=increments, nlgeom=True, max_cutbacks=0
        )

    # A gentler cutback needs more substeps than the default to get there.
    calls = _count_newton_solves(monkeypatch)
    _build_bent_cantilever().solve(
        increments=increments, nlgeom=True, cutback_factor=0.8, growth_factor=1.05
    )
    assert len(calls) > 1


def test_growing_increments_are_each_attempted_whole(capsys):
    """A load path with growing increments takes one substep per increment."""
    increments = torch.tensor([0.0, 0.1, 0.25, 0.45, 0.7, 1.0])
    _build_cantilever().solve(increments=increments, verbose=True)

    # Third column of the report is the substep count of each increment.
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[7:-2]]
    assert [row[2] for row in rows] == ["1"] * (len(increments) - 1)


def test_cutback_returns_results_at_the_requested_increments(monkeypatch):
    """Substepping must not change which load factors are reported."""
    requested = torch.tensor([0.0, 0.3, 0.55, 1.0])

    # Force a cutback, so the solve substeps by construction
    original = base.newton_solve
    calls = []

    def failing_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("forced cutback")
        return original(*args, **kwargs)

    monkeypatch.setattr(base, "newton_solve", failing_once)
    u, f, _, _, _ = _build_bent_cantilever().solve(
        increments=requested, nlgeom=True, return_intermediate=True
    )

    assert len(calls) > len(requested) - 1
    assert u.shape[0] == len(requested)

    # Each reported state must be the equilibrium of its own load factor.
    for n, factor in enumerate(requested):
        model = _build_bent_cantilever()
        model.forces *= float(factor)
        u_n, _, _, _, _ = model.solve(increments=torch.linspace(0, 1, 21), nlgeom=True)
        assert torch.allclose(u[n], u_n, rtol=1e-6, atol=1e-9)


def test_stabilization_works_for_solids():
    model = Solid(*cube_hexa(3, 3, 3), IsotropicElasticity3D(E=1000.0, nu=0.3))
    bottom = torch.isclose(model.nodes[:, 2], model.nodes[:, 2].min())
    top = torch.isclose(model.nodes[:, 2], model.nodes[:, 2].max())
    model.constraints[bottom | top] = True
    model.displacements[top, 0] = 0.1

    u, _, _, _, _ = model.solve(increments=INCREMENTS, alpha=1.0)

    assert torch.allclose(u[top, 0], torch.full((int(top.sum()),), 0.1))
    assert model.stabilization_energy[-1] > 0.0
    # Cumulative dissipation increases monotonically.
    diff = model.stabilization_energy[1:] - model.stabilization_energy[:-1]
    assert torch.all(diff >= 0.0)
