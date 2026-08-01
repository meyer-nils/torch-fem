import inspect

import pytest
import torch
from scipy.linalg import eigh as scipy_eigh

from torchfem.sparse import (
    CachedSolve,
    available_backends,
    describe_method,
    differentiable_modal_eigsolve,
    differentiable_sparse_solve,
    modal_eigsolve,
    newton_solve,
    resolve_method,
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


class TestSparseSolve:
    @pytest.mark.parametrize("method", [None, "spsolve", "cg", "minres"])
    def test_matches_dense_solution(self, method):
        A_dense = _spd(N, 0)
        b = torch.randn(N)
        A, _ = _to_sparse(A_dense)
        x, _ = sparse_solve(A, b, method=method)
        assert torch.allclose(x, torch.linalg.solve(A_dense, b), atol=1e-8)

    def test_direct_method_returns_no_preconditioner(self):
        A, _ = _to_sparse(_spd(N, 0))
        _, M = sparse_solve(A, torch.randn(N), method="spsolve")
        assert M is None

    @pytest.mark.parametrize("method", ["cg", "minres"])
    def test_iterative_method_builds_a_preconditioner(self, method):
        """The preconditioner is returned so the adjoint solve can reuse it."""
        A, _ = _to_sparse(_spd(N, 0))
        _, M = sparse_solve(A, torch.randn(N), method=method)
        assert M is not None

    def test_rejects_non_square_matrix(self):
        """The message is matched on purpose: without this guard the call still
        fails, but deep inside SciPy with a different message."""
        A = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]]), torch.ones(2), (2, 3)
        ).coalesce()
        with pytest.raises(ValueError, match="square 2D matrix"):
            sparse_solve(A, torch.ones(2))

    def test_rejects_unknown_method(self):
        A, _ = _to_sparse(_spd(N, 0))
        with pytest.raises(ValueError, match="is not supported"):
            sparse_solve(A, torch.randn(N), method="not-a-solver")

    def test_initial_guess_does_not_change_the_solution(self):
        """A warm start may only change the iteration count, never the answer."""
        A_dense = _spd(N, 0)
        b = torch.randn(N)
        A, _ = _to_sparse(A_dense)
        reference = torch.linalg.solve(A_dense, b)
        x, _ = sparse_solve(A, b, method="cg", x0=reference + 0.1 * torch.randn(N))
        assert torch.allclose(x, reference, atol=1e-8)


class TestDifferentiableSparseSolve:
    """The adjoint backward pass of `Solve`, checked against dense autograd."""

    @staticmethod
    def _sparse_grads(A_dense, b, w):
        values = A_dense.flatten().clone().requires_grad_(True)
        A, idx = _to_sparse(A_dense, values)
        b_grad = b.clone().requires_grad_(True)
        x = differentiable_sparse_solve(A, b_grad)
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


class TestCachedSolve:
    def test_update_x_stores_a_detached_copy(self):
        cache = CachedSolve()
        x = torch.ones(3, requires_grad=True)
        cache.update_x(x)
        assert cache.previous_x is not None
        assert not cache.previous_x.requires_grad
        assert cache.previous_x is not x

    def test_update_grad_stores_a_detached_copy(self):
        cache = CachedSolve()
        g = torch.ones(3, requires_grad=True)
        cache.update_grad(g)
        assert cache.previous_grad is not None
        assert not cache.previous_grad.requires_grad

    def test_update_with_none_clears_the_entry(self):
        cache = CachedSolve(previous_x=torch.ones(3), previous_grad=torch.ones(3))
        cache.update_x(None)
        cache.update_grad(None)
        assert cache.previous_x is None
        assert cache.previous_grad is None

    @pytest.mark.parametrize("func", [differentiable_sparse_solve, newton_solve])
    def test_solvers_do_not_share_a_default_cache(self, func):
        """Regression guard: a `CachedSolve()` default would be constructed once
        at import and shared by every caller, so one solve could warm-start a
        later, unrelated solve from a stale (possibly wrong-sized) vector."""
        assert inspect.signature(func).parameters["cached_solve"].default is None

    def test_warm_start_reaches_the_same_solution(self):
        A_dense = _spd(N, 0)
        b = torch.randn(N)
        A, _ = _to_sparse(A_dense)
        reference = torch.linalg.solve(A_dense, b)

        cache = CachedSolve()
        first = differentiable_sparse_solve(
            A, b, method="cg", cached_solve=cache, update_cache=True
        )
        assert cache.previous_x is not None
        second = differentiable_sparse_solve(
            A, b, method="cg", cached_solve=cache, update_cache=True
        )
        assert torch.allclose(first, reference, atol=1e-8)
        assert torch.allclose(second, reference, atol=1e-8)


class TestModalEigsolve:
    @staticmethod
    def _problem(n=8):
        K_dense, M_dense = _spd(n, 0), _spd(n, 1)
        free = torch.arange(2, n)
        K, _ = _to_sparse(K_dense)
        M, _ = _to_sparse(M_dense)
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
        M, _ = _to_sparse(M_dense)
        D = self._direction(n)

        values = K_dense.flatten().clone().requires_grad_(True)
        K, _ = _to_sparse(K_dense, values)
        lambdas, _ = differentiable_modal_eigsolve(K, M, N_MODES, free)
        (grad,) = torch.autograd.grad(lambdas.sum(), [values])

        plus, _ = modal_eigsolve(_to_sparse(K_dense + eps * D)[0], M, N_MODES, free)
        minus, _ = modal_eigsolve(_to_sparse(K_dense - eps * D)[0], M, N_MODES, free)
        finite_difference = (plus.sum() - minus.sum()) / (2 * eps)
        assert torch.allclose(
            torch.dot(grad, D.flatten()), finite_difference, rtol=1e-5
        )

    def test_eigenvalue_gradient_wrt_mass_matches_finite_difference(self):
        n, eps = 8, 1e-6
        K_dense, M_dense = _spd(n, 0), _spd(n, 1)
        free = torch.arange(2, n)
        K, _ = _to_sparse(K_dense)
        D = self._direction(n)

        values = M_dense.flatten().clone().requires_grad_(True)
        M, _ = _to_sparse(M_dense, values)
        lambdas, _ = differentiable_modal_eigsolve(K, M, N_MODES, free)
        (grad,) = torch.autograd.grad(lambdas.sum(), [values])

        plus, _ = modal_eigsolve(K, _to_sparse(M_dense + eps * D)[0], N_MODES, free)
        minus, _ = modal_eigsolve(K, _to_sparse(M_dense - eps * D)[0], N_MODES, free)
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
        K, _ = _to_sparse(K_dense, values)
        M, _ = _to_sparse(M_dense)
        lambdas, phis = differentiable_modal_eigsolve(K, M, N_MODES, torch.arange(2, n))
        assert lambdas.requires_grad
        assert not phis.requires_grad


class TestResolveMethod:
    def test_an_explicit_method_is_kept(self):
        assert resolve_method(10, "cpu", "cg") == "cg"
        assert resolve_method(10**6, "cuda", "spsolve") == "spsolve"

    def test_large_systems_switch_to_an_iterative_solver(self):
        assert resolve_method(9999, "cpu", None) != "minres"
        assert resolve_method(10000, "cpu", None) == "minres"

    def test_small_systems_use_a_direct_solver(self):
        direct = "pardiso" if "pypardiso" in available_backends else "spsolve"
        assert resolve_method(10, "cpu", None) == direct
        # Pardiso is CPU only, so the GPU falls back to the CuPy direct solver.
        assert resolve_method(10, "cuda", None) == "spsolve"

    def test_the_description_names_preconditioner_library_and_device(self):
        assert describe_method(10, "cpu", "cg") == "cg · iterative · AMG · scipy · cpu"
        assert "jacobi" in describe_method(10, "cuda", "cg")
        assert "cupy" in describe_method(10, "cuda", "spsolve")
