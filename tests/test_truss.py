import math

import pytest
import torch

from torchfem import Truss
from torchfem.elements import Bar1, Bar2, linear_to_quadratic
from torchfem.materials import IsotropicElasticity1D

E = 1000.0
LENGTH = 2.0


def _axial_bar() -> Truss:
    """Single bar of length 2 along x, fixed at node 0, pulled at node 1."""
    nodes = torch.tensor([[0.0, 0.0], [LENGTH, 0.0]])
    elements = torch.tensor([[0, 1]])
    bar = Truss(nodes, elements, IsotropicElasticity1D(E))
    bar.constraints[0, :] = True
    bar.constraints[1, 1] = True
    bar.forces[1, 0] = 10.0
    return bar


class TestTruss:
    def test_etype_is_bar1_for_two_node_elements(self):
        assert _axial_bar().etype is Bar1

    def test_etype_is_bar2_for_three_node_elements(self):
        nodes, elements = linear_to_quadratic(
            torch.tensor([[0.0, 0.0], [LENGTH, 0.0]]), torch.tensor([[0, 1]])
        )
        assert Truss(nodes, elements, IsotropicElasticity1D(E)).etype is Bar2

    def test_etype_rejects_unsupported_connectivity(self):
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        elements = torch.tensor([[0, 1, 2, 3]])
        with pytest.raises(ValueError, match="not supported"):
            _ = Truss(nodes, elements, IsotropicElasticity1D(E)).etype

    def test_char_lengths_are_element_lengths(self):
        nodes = torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
        elements = torch.tensor([[0, 1], [1, 2], [0, 2]])
        truss = Truss(nodes, elements, IsotropicElasticity1D(E))
        assert torch.allclose(truss.char_lengths, torch.tensor([3.0, 4.0, 5.0]))

    def test_axial_bar_matches_analytical_solution(self):
        bar = _axial_bar()
        u, f, sigma, _, _ = bar.solve()
        # u = F L / (E A) and sigma = F / A for a bar in uniaxial tension.
        assert torch.allclose(u[1, 0], torch.tensor(10.0 * LENGTH / E))
        assert torch.allclose(sigma.ravel(), torch.tensor(10.0))
        # Internal forces balance the applied load.
        assert torch.allclose(f[1], torch.tensor([10.0, 0.0]))
        assert torch.allclose(f.sum(dim=0), torch.zeros(2))

    def test_area_scales_stiffness(self):
        bar = _axial_bar()
        bar.areas = 2.0 * torch.ones(1)
        u, _, sigma, _, _ = bar.solve()
        assert torch.allclose(u[1, 0], torch.tensor(10.0 * LENGTH / (E * 2.0)))
        assert torch.allclose(sigma.ravel(), torch.tensor(5.0))

    def test_eval_shape_functions_returns_expected_shapes(self):
        bar = _axial_bar()
        xi = bar.etype.ipoints
        N, B, detJ = bar.eval_shape_functions(xi)
        assert N.shape == (xi.shape[0], 2)
        # B maps the 4 nodal DOFs of the element onto one axial strain.
        assert B.shape == (xi.shape[0], 1, 1, 4)
        assert torch.allclose(detJ, torch.tensor(LENGTH / 2.0))

    def test_zero_length_element_raises(self):
        nodes = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        bar = Truss(nodes, torch.tensor([[0, 1]]), IsotropicElasticity1D(E))
        with pytest.raises(ValueError, match="Negative Jacobian"):
            bar.eval_shape_functions(bar.etype.ipoints)

    def test_modal_frequencies_match_analytical_bar(self):
        """First axial mode of a fixed-free bar: omega = pi / (2 L) sqrt(E / rho)."""
        n_elem, rho = 20, 2.0
        x = torch.linspace(0.0, 1.0, n_elem + 1)
        nodes = torch.stack([x, torch.zeros_like(x)], dim=1)
        elements = torch.stack([torch.arange(n_elem), torch.arange(1, n_elem + 1)], 1)
        bar = Truss(nodes, elements, IsotropicElasticity1D(E, rho))
        bar.constraints[0, :] = True
        bar.constraints[:, 1] = True

        omega_sq, modes = bar.solve_modes(n_modes=3)
        omega = torch.sqrt(omega_sq)
        assert modes.shape == (3, n_elem + 1, 2)
        assert torch.all(omega[1:] > omega[:-1])
        expected = math.pi / 2.0 * math.sqrt(E / rho)
        assert torch.allclose(omega[0], torch.tensor(expected), rtol=1e-2)
