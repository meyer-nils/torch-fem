import pytest
import torch

from torchfem import PlanarHeat
from torchfem.materials import IsotropicConductivity2D
from torchfem.mesh import rect_quad


def _build_heated_plate() -> PlanarHeat:
    material = IsotropicConductivity2D(kappa=400.0, rho=1.0e5)
    model = PlanarHeat(*rect_quad(5, 5, 1.0, 1.0), material)
    west = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].min())
    east = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].max())
    model.constraints[west | east] = True
    model.displacements[west, 0] = 5.0
    model.displacements[east, 0] = 20.0
    return model


def test_time_integration_returns_one_result_per_output_time():
    model = _build_heated_plate()
    t_output = torch.tensor([0.0, 6.0, 12.0, 18.0, 24.0])

    temp, rfl, flux, grad, state = model.time_integration(t_output, delta_t=1.0)

    for result in (temp, rfl, flux, grad, state):
        assert result.shape[0] == len(t_output)


def test_time_integration_single_output_time_keeps_time_axis():
    model = _build_heated_plate()

    temp, _, flux, grad, _ = model.time_integration(torch.tensor([10.0]), delta_t=1.0)

    # A bare squeeze() would collapse the length-1 time axis here.
    assert temp.shape[0] == 1
    assert flux.shape[0] == 1
    assert grad.shape[0] == 1


def test_time_integration_sparse_output_matches_dense_output():
    """Requesting fewer output times must not change the internal time stepping."""
    model = _build_heated_plate()
    delta_t = 0.1

    dense_times = torch.arange(0.0, 10.0 + delta_t, delta_t)
    dense, *_ = model.time_integration(dense_times, delta_t)

    sparse_times = torch.tensor([0.0, 2.5, 5.0, 10.0])
    sparse, *_ = model.time_integration(sparse_times, delta_t)

    rows = torch.searchsorted(dense_times.contiguous(), sparse_times)
    assert torch.allclose(sparse, dense[rows], atol=1e-9)


def test_time_integration_delta_t_controls_accuracy_independently_of_t_output():
    """delta_t must drive the internal stepping even for identical output times."""
    model = _build_heated_plate()
    t_output = torch.tensor([0.0, 1.0])

    reference, *_ = model.time_integration(t_output, delta_t=0.005)
    coarse, *_ = model.time_integration(t_output, delta_t=99.0)  # clamped to one step
    fine, *_ = model.time_integration(t_output, delta_t=0.05)

    error_coarse = (coarse[-1] - reference[-1]).abs().max()
    error_fine = (fine[-1] - reference[-1]).abs().max()

    assert error_fine < error_coarse


def test_time_integration_starts_from_equilibrium_when_output_starts_late():
    model = _build_heated_plate()
    delta_t = 0.5

    full, *_ = model.time_integration(
        torch.arange(0.0, 10.0 + delta_t, delta_t), delta_t
    )
    late, *_ = model.time_integration(torch.tensor([5.0, 10.0]), delta_t)

    # Integration always starts at t=0, even if t=0 is not requested as output.
    assert torch.allclose(late, full[torch.tensor([10, 20])], atol=1e-9)


@pytest.mark.parametrize(
    "t_output",
    [
        torch.tensor([]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([1.0, 0.5]),
        torch.tensor([1.0, 1.0]),
    ],
)
def test_time_integration_rejects_invalid_output_times(t_output):
    model = _build_heated_plate()

    with pytest.raises(ValueError):
        model.time_integration(t_output, delta_t=0.5)
