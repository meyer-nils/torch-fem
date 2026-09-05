import pytest
import torch
from scipy.linalg import eigh as scipy_eigh

from torchfem.sparse import (
    available_backends,
    describe_method,
    differentiable_modal_eigsolve,
    differentiable_sparse_solve,
    modal_eigsolve,
    resolve_library,
    resolve_method,
    resolve_preconditioner,
    sparse_solve,
)

N = 6
N_MODES = 3


def _spd(n: int, seed: int) -> torch.Tensor:
    """Dense symmetric positive definite matrix."""
    torch.manual_seed(seed)
    A = torch.randn(n, n)
    return A @ A.T + n * torch.eye(n)


def _nonsymmetric(n: int, seed: int) -> torch.Tensor:
    """Dense diagonally dominant, non-symmetric (hence invertible) matrix.

    The adjoint transposes the system matrix, which is invisible on a symmetric
    matrix, so the gradient checks need a non-symmetric case as well.
    """
    torch.manual_seed(seed)
    return torch.randn(n, n) + n * torch.eye(n)


def _to_sparse(dense: torch.Tensor, values: torch.Tensor | None = None):
    """Sparse COO copy of `dense`, optionally with substituted values."""
    idx = torch.cartesian_prod(
        torch.arange(dense.shape[0]), torch.arange(dense.shape[1])
    ).T
    vals = dense.flatten() if values is None else values
    return torch.sparse_coo_tensor(idx, vals, dense.shape).coalesce(), idx


def _to_csr(dense: torch.Tensor, values: torch.Tensor | None = None) -> torch.Tensor:
    """Sparse CSR copy of `dense`, the layout the solvers take."""
    return _to_sparse(dense, values)[0].to_sparse_csr()


class TestSparseSolve:
    @pytest.mark.parametrize("method", [None, "direct", "cg", "bicgstab"])
    def test_matches_dense_solution(self, method):
        A_dense = _spd(N, 0)
        b = torch.randn(N)
        x, _, _ = sparse_solve(_to_csr(A_dense), b, method=method)
        assert torch.allclose(x, torch.linalg.solve(A_dense, b), atol=1e-8)

    def test_direct_method_returns_no_preconditioner(self):
        _, M, _ = sparse_solve(_to_csr(_spd(N, 0)), torch.randn(N), method="direct")
        assert M is None

    @pytest.mark.parametrize("method", ["cg", "bicgstab"])
    def test_iterative_method_builds_a_preconditioner(self, method):
        """The preconditioner is returned so the adjoint solve can reuse it."""
        _, M, _ = sparse_solve(_to_csr(_spd(N, 0)), torch.randn(N), method=method)
        assert M is not None

    def test_bicgstab_solves_a_non_symmetric_system(self):
        """CG assumes a symmetric matrix; a damage tangent is not one."""
        A_dense = _nonsymmetric(40, 0)
        b = torch.randn(40)
        x, _, _ = sparse_solve(_to_csr(A_dense), b, method="bicgstab")
        assert torch.allclose(x, torch.linalg.solve(A_dense, b), atol=1e-8)

    def test_rejects_unknown_preconditioner(self):
        with pytest.raises(ValueError, match="is not supported"):
            sparse_solve(_to_csr(_spd(N, 0)), torch.randn(N), preconditioner="ilu")

    @pytest.mark.parametrize("preconditioner", ["amg", "jacobi", "none"])
    def test_every_preconditioner_reaches_the_same_solution(self, preconditioner):
        A_dense = _spd(N, 0)
        b = torch.randn(N)
        x, _, _ = sparse_solve(
            _to_csr(A_dense), b, method="cg", preconditioner=preconditioner
        )
        assert torch.allclose(x, torch.linalg.solve(A_dense, b), atol=1e-8)

    def test_rejects_non_square_matrix(self):
        """The message is matched on purpose: without this guard the call still
        fails, but deep inside SciPy with a different message."""
        with pytest.raises(ValueError, match="square 2D matrix"):
            sparse_solve(_to_csr(torch.ones(2, 3)), torch.ones(2))

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="is not supported"):
            sparse_solve(_to_csr(_spd(N, 0)), torch.randn(N), method="not-a-solver")


class TestDifferentiableSparseSolve:
    """The adjoint backward pass of `Solve`, checked against dense autograd."""

    @staticmethod
    def _sparse_grads(A_dense, b, w, compressed=True):
        values = A_dense.flatten().clone().requires_grad_(True)
        A, idx = _to_sparse(A_dense, values)
        b_grad = b.clone().requires_grad_(True)
        x = differentiable_sparse_solve(A.to_sparse_csr() if compressed else A, b_grad)
        grad_values, grad_b = torch.autograd.grad((x * w).sum(), [values, b_grad])
        grad_A = torch.sparse_coo_tensor(idx, grad_values, A_dense.shape).to_dense()
        return x, grad_A, grad_b

    @staticmethod
    def _dense_grads(A_dense, b, w):
        A = A_dense.clone().requires_grad_(True)
        b_grad = b.clone().requires_grad_(True)
        x = torch.linalg.solve(A, b_grad)
        grad_A, grad_b = torch.autograd.grad((x * w).sum(), [A, b_grad])
        return x, grad_A, grad_b

    @pytest.mark.parametrize("build", [_spd, _nonsymmetric])
    def test_forward_matches_dense(self, build):
        A_dense, b, w = build(N, 0), torch.randn(N), torch.randn(N)
        x, _, _ = self._sparse_grads(A_dense, b, w)
        assert torch.allclose(x, torch.linalg.solve(A_dense, b), atol=1e-10)

    @pytest.mark.parametrize("build", [_spd, _nonsymmetric])
    def test_gradient_wrt_rhs_matches_dense(self, build):
        A_dense, b, w = build(N, 0), torch.randn(N), torch.randn(N)
        _, _, grad_b = self._sparse_grads(A_dense, b, w)
        _, _, grad_b_dense = self._dense_grads(A_dense, b, w)
        assert torch.allclose(grad_b, grad_b_dense, atol=1e-10)

    @pytest.mark.parametrize("build", [_spd, _nonsymmetric])
    def test_gradient_wrt_matrix_matches_dense(self, build):
        A_dense, b, w = build(N, 0), torch.randn(N), torch.randn(N)
        _, grad_A, _ = self._sparse_grads(A_dense, b, w)
        _, grad_A_dense, _ = self._dense_grads(A_dense, b, w)
        assert torch.allclose(grad_A, grad_A_dense, atol=1e-10)

    @pytest.mark.parametrize("build", [_spd, _nonsymmetric])
    def test_a_coo_matrix_gives_the_same_gradients(self, build):
        """A COO input is compressed on the way in, and its gradient chains back.

        The non-symmetric case pins the adjoint, whose `A.t()` is a CSC view.
        """
        A_dense, b, w = build(N, 0), torch.randn(N), torch.randn(N)
        for coo, csr in zip(
            self._sparse_grads(A_dense, b, w, compressed=False),
            self._sparse_grads(A_dense, b, w),
        ):
            assert torch.allclose(coo, csr, atol=1e-10)

    @pytest.mark.parametrize("build", [_spd, _nonsymmetric])
    def test_gradient_wrt_rhs_is_the_adjoint_solution(self, build):
        """dL/db solves the adjoint system A^T lambda = dL/dx.

        On a non-symmetric matrix this also pins the transpose: solving with A
        instead of A^T would give a different vector.
        """
        A_dense, b, w = build(N, 0), torch.randn(N), torch.randn(N)
        _, _, grad_b = self._sparse_grads(A_dense, b, w)
        assert torch.allclose(grad_b, torch.linalg.solve(A_dense.T, w), atol=1e-10)

    def test_matrix_gradient_keeps_the_sparsity_pattern(self):
        """The outer product is only evaluated on the stored entries, so no
        dense n-by-n matrix is ever formed."""
        A_dense = _spd(N, 0)
        # Drop the corner entries so the pattern is genuinely sparse.
        A_dense[0, -1] = A_dense[-1, 0] = 0.0
        idx = A_dense.nonzero().T
        values = A_dense[idx[0], idx[1]].clone().requires_grad_(True)
        A = torch.sparse_coo_tensor(idx, values, A_dense.shape).coalesce()
        b = torch.randn(N).requires_grad_(True)
        x = differentiable_sparse_solve(A, b)
        (grad_values,) = torch.autograd.grad(x.sum(), [values])
        assert grad_values.shape == values.shape

    @pytest.mark.parametrize("method", ["cg", "bicgstab"])
    @pytest.mark.parametrize("preconditioner", ["jacobi", "none"])
    def test_an_iterative_adjoint_matches_dense(self, method, preconditioner):
        """The adjoint hands the solver `A.t()`, compressed by column. `cg`
        rereads those arrays as rows rather than transposing them, which is only
        the same matrix because `cg` already assumes symmetry."""
        A_dense, b, w = _spd(N, 0), torch.randn(N), torch.randn(N)
        values = A_dense.flatten().clone().requires_grad_(True)
        A, _ = _to_sparse(A_dense, values)
        b_grad = b.clone().requires_grad_(True)
        x = differentiable_sparse_solve(
            A.to_sparse_csr(), b_grad, method=method, preconditioner=preconditioner
        )
        grad_b = torch.autograd.grad((x * w).sum(), [b_grad])[0]
        assert torch.allclose(x, torch.linalg.solve(A_dense, b), atol=1e-8)
        assert torch.allclose(grad_b, torch.linalg.solve(A_dense.T, w), atol=1e-8)


class TestModalEigsolve:
    @staticmethod
    def _problem(n=8):
        K_dense, M_dense = _spd(n, 0), _spd(n, 1)
        free = torch.arange(2, n)
        K, M = _to_csr(K_dense), _to_csr(M_dense)
        return K_dense, M_dense, K, M, free

    def test_matches_dense_generalized_eigenproblem(self):
        K_dense, M_dense, K, M, free = self._problem()
        values, _ = modal_eigsolve(K, M, N_MODES, free)
        reference = scipy_eigh(
            K_dense[free][:, free].numpy(),
            M_dense[free][:, free].numpy(),
            eigvals_only=True,
        )[:N_MODES]
        assert torch.allclose(values, torch.tensor(reference), rtol=1e-8)

    def test_eigenvalues_are_ascending(self):
        _, _, K, M, free = self._problem()
        values, _ = modal_eigsolve(K, M, N_MODES, free)
        assert (values[1:] >= values[:-1]).all()

    def test_constrained_dofs_vanish_in_the_modes(self):
        """The eigenproblem is solved in the free-DOF subspace, so constrained
        rows must come back as exact zeros rather than penalty artefacts."""
        _, _, K, M, free = self._problem()
        _, vectors = modal_eigsolve(K, M, N_MODES, free)
        constrained = torch.tensor([i for i in range(8) if i not in free])
        assert vectors.shape == (8, N_MODES)
        assert torch.allclose(
            vectors[constrained], torch.zeros(len(constrained), N_MODES)
        )


class TestEigensolveGradients:
    """Rayleigh-quotient sensitivities, checked along symmetric perturbations.

    A single off-diagonal entry cannot be perturbed on its own, because the
    backend treats the matrices as symmetric; the derivative is only well posed
    along a symmetric direction.
    """

    @staticmethod
    def _direction(n, seed=7):
        torch.manual_seed(seed)
        D = torch.randn(n, n)
        return 0.5 * (D + D.T)

    def test_eigenvalue_gradient_wrt_stiffness_matches_finite_difference(self):
        n, eps = 8, 1e-6
        K_dense, M_dense = _spd(n, 0), _spd(n, 1)
        free = torch.arange(2, n)
        M = _to_csr(M_dense)
        D = self._direction(n)

        values = K_dense.flatten().clone().requires_grad_(True)
        K = _to_csr(K_dense, values)
        lambdas, _ = differentiable_modal_eigsolve(K, M, N_MODES, free)
        (grad,) = torch.autograd.grad(lambdas.sum(), [values])

        plus, _ = modal_eigsolve(_to_csr(K_dense + eps * D), M, N_MODES, free)
        minus, _ = modal_eigsolve(_to_csr(K_dense - eps * D), M, N_MODES, free)
        finite_difference = (plus.sum() - minus.sum()) / (2 * eps)
        assert torch.allclose(
            torch.dot(grad, D.flatten()), finite_difference, rtol=1e-5
        )

    def test_eigenvalue_gradient_wrt_mass_matches_finite_difference(self):
        n, eps = 8, 1e-6
        K_dense, M_dense = _spd(n, 0), _spd(n, 1)
        free = torch.arange(2, n)
        K = _to_csr(K_dense)
        D = self._direction(n)

        values = M_dense.flatten().clone().requires_grad_(True)
        M = _to_csr(M_dense, values)
        lambdas, _ = differentiable_modal_eigsolve(K, M, N_MODES, free)
        (grad,) = torch.autograd.grad(lambdas.sum(), [values])

        plus, _ = modal_eigsolve(K, _to_csr(M_dense + eps * D), N_MODES, free)
        minus, _ = modal_eigsolve(K, _to_csr(M_dense - eps * D), N_MODES, free)
        finite_difference = (plus.sum() - minus.sum()) / (2 * eps)
        assert torch.allclose(
            torch.dot(grad, D.flatten()), finite_difference, rtol=1e-5
        )

    def test_mode_shapes_are_detached(self):
        """Only eigenvalue gradients are supported, so the mode shapes must come
        back detached rather than carrying a silently wrong graph."""
        n = 8
        K_dense, M_dense = _spd(n, 0), _spd(n, 1)
        values = K_dense.flatten().clone().requires_grad_(True)
        K = _to_csr(K_dense, values)
        M = _to_csr(M_dense)
        lambdas, phis = differentiable_modal_eigsolve(K, M, N_MODES, torch.arange(2, n))
        assert lambdas.requires_grad
        assert not phis.requires_grad


class TestResolveMethod:
    def test_an_explicit_method_is_kept(self):
        assert resolve_method(10, "cg") == "cg"
        assert resolve_method(10**6, "direct") == "direct"

    def test_small_systems_use_a_direct_solver(self):
        assert resolve_method(9999, None) == "direct"
        assert resolve_method(10000, None) != "direct"

    def test_an_unsymmetric_tangent_needs_bicgstab(self):
        assert resolve_method(10**6, None, symmetric=True) == "cg"
        assert resolve_method(10**6, None, symmetric=False) == "bicgstab"

    def test_the_description_names_preconditioner_library_and_device(self):
        """Every preconditioner is named on purpose: what `None` resolves to on
        CUDA depends on whether AmgX is installed, which the machine decides."""
        direct = "pypardiso" if "pypardiso" in available_backends else "scipy"
        assert (
            describe_method("cg", "cpu", None) == "cg | iterative | amg | scipy | cpu"
        )
        assert (
            describe_method("cg", "cuda", "jacobi")
            == "cg | iterative | jacobi | torch | cuda"
        )
        assert (
            describe_method("direct", "cuda", None)
            == f"direct | direct | {direct} | cuda"
        )


class TestResolvePreconditioner:
    def test_a_direct_solve_takes_none(self):
        assert resolve_preconditioner("direct", "cpu", None) == "none"

    def test_a_direct_solve_rejects_one(self):
        with pytest.raises(ValueError, match="takes no preconditioner"):
            resolve_preconditioner("direct", "cpu", "amg")

    def test_an_iterative_solve_defaults_to_amg_on_the_cpu(self):
        assert resolve_preconditioner("cg", "cpu", None) == "amg"

    def test_an_explicit_preconditioner_is_kept(self):
        assert resolve_preconditioner("cg", "cpu", "jacobi") == "jacobi"

    def test_amg_on_cuda_needs_amgx(self):
        if "amgx" in available_backends:
            assert resolve_preconditioner("cg", "cuda", "amg") == "amg"
        else:
            assert resolve_preconditioner("cg", "cuda", None) == "jacobi"
            with pytest.raises(RuntimeError, match="AmgX is not available"):
                resolve_preconditioner("cg", "cuda", "amg")


class TestResolveLibrary:
    def test_the_cpu_uses_pardiso_for_a_direct_solve_where_it_is_installed(self):
        direct = "pypardiso" if "pypardiso" in available_backends else "scipy"
        assert resolve_library("direct", "cpu", "none") == direct

    def test_amg_on_cuda_is_amgx(self):
        assert resolve_library("cg", "cuda", "amg") == "amgx"

    def test_the_cpu_uses_scipy_for_an_iterative_solve(self):
        assert resolve_library("cg", "cpu", "amg") == "scipy"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("preconditioner", ["jacobi", "none"])
    def test_a_diagonal_preconditioner_keeps_the_solve_in_torch(
        self, device, preconditioner
    ):
        """Only a hierarchy or a factorisation is worth another library."""
        assert resolve_library("cg", device, preconditioner) == "torch"
