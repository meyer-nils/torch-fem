import re

import torch

from torchfem import Planar, PlanarHeat
from torchfem.materials import IsotropicConductivity2D, IsotropicElasticityPlaneStress
from torchfem.mesh import rect_quad
from torchfem.report import WIDTH, SolveReport, machine
from torchfem.sparse import describe_method


def _build_cantilever() -> Planar:
    material = IsotropicElasticityPlaneStress(E=1000.0, nu=0.3)
    model = Planar(*rect_quad(5, 3, 1.0, 1.0), material)
    model.constraints[torch.isclose(model.nodes[:, 0], model.nodes[:, 0].min())] = True
    model.forces[torch.isclose(model.nodes[:, 0], model.nodes[:, 0].max()), 1] = 1.0
    return model


class TestSolveReport:
    def test_a_row_is_streamed_per_increment(self, capsys):
        report = SolveReport("title", {"model": "a model"})
        for n in range(1, 4):
            report.begin(n, 0.25 * n)
            report.iteration(0, 1.0)
            report.iteration(1, 1e-12)
            report.end()
        report.close()

        out = capsys.readouterr().out
        assert " model    a model" in out
        for title in ("Increment", "Load factor", "Steps", "Iterations"):
            assert title in out
        # A header line, the column titles, three rows, a summary and 3 rules.
        # No row is ever elided, so the count also pins the table height.
        assert len(out.splitlines()) == 9
        # One linear solve each: the residual opening a substep is not counted.
        assert "3 increments | 3 iterations" in out
        # No substep flag in the right margin, since none was cut back or grown.
        assert not re.search(r"[v^]\d", out)

    def test_cutbacks_and_growths_are_counted(self, capsys):
        report = SolveReport("title", {})
        report.begin(1, 1.0)
        # Three failed attempts, then a substep that converges in one solve
        # and lets the solver grow the substep twice.
        for _ in range(3):
            report.iteration(0, 1.0)
            report.cutback()
        report.iteration(0, 1.0)
        report.iteration(1, 1e-12)
        report.growth()
        report.growth()
        report.end()
        report.close()

        out = capsys.readouterr().out
        # Increment 1 at λ = 1, solved in 4 substeps and one linear solve.
        row = out.splitlines()[-3]
        assert row.split()[:5] == ["1", "1", "4", "1", "1.00e-12"]
        # The flags count both events and live in the right margin, so the row
        # still fits the rule.
        assert row.endswith("v3 ^2")
        assert len(row) <= WIDTH
        assert "1 increment | 1 iteration" in out


class TestVerboseSolve:
    def test_solve_reports_the_backend_and_one_row_per_increment(self, capsys):
        model = _build_cantilever()

        model.solve(increments=torch.linspace(0, 1, 4), verbose=True)

        out = capsys.readouterr().out
        assert describe_method(model.n_dofs, "cpu", None) in out
        assert f"{model.n_elem:,} elem" in out
        assert f" machine  {machine()}" in out
        assert "3 increments" in out

    def test_solve_is_silent_by_default(self, capsys):
        _build_cantilever().solve()

        assert capsys.readouterr().out == ""

    def test_time_integration_reports_its_time_steps(self, capsys):
        material = IsotropicConductivity2D(kappa=400.0, rho=1.0e5)
        model = PlanarHeat(*rect_quad(3, 3, 1.0, 1.0), material)
        model.displacements[:, 0] = 5.0
        model.constraints[:] = True

        model.time_integration(torch.tensor([2.0]), delta_t=1.0, verbose=True)

        out = capsys.readouterr().out
        assert "torch-fem | time integration" in out
        assert "2 time steps" in out


class TestMachine:
    def test_the_thread_count_is_read_per_call(self):
        """Only the CPU name and memory are cached, so a later change shows up."""
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            assert "1 thread |" in machine()
        finally:
            torch.set_num_threads(threads)
