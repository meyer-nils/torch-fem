from __future__ import annotations

from math import sqrt
from typing import Callable

import torch
from torch import Tensor

from ..utils import (
    stiffness2voigt,
    strain2voigt,
    stress2voigt,
    voigt2stiffness,
    voigt2stress,
)
from .elasticity import (
    IsotropicElasticity1D,
    IsotropicElasticity3D,
    IsotropicElasticityPlaneStrain,
    IsotropicElasticityPlaneStress,
)


class IsotropicPlasticity3D(IsotropicElasticity3D):
    """Isotropic elastoplastic material model in 3D.

    This class extends `IsotropicElasticity3D` to incorporate isotropic
    plasticity with a von Mises yield criterion and associative flow rule.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor | float): Poisson's ratio.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        sigma_f (Callable): Yield stress function $\\sigma_f(q)$ of the
            equivalent plastic strain $q$.
        sigma_f_prime (Callable): Derivative $\\sigma_f'(q)$.
        tolerance (float): Convergence tolerance for the local Newton solver.
            Default is `1e-5`.
        max_iter (int): Maximum number of Newton iterations. Default is `10`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption.
        - One internal state variable (``n_state = 1``): equivalent plastic
          strain $q$.
        - Supports batched/vectorized material parameters.
    """

    def __init__(
        self,
        E: float | Tensor,
        nu: float | Tensor,
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, nu, rho)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 1
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int) -> IsotropicPlasticity3D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicPlasticity3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicPlasticity3D(
                E,
                nu,
                self.sigma_f,
                self.sigma_f_prime,
                self.tolerance,
                self.max_iter,
                rho,
            )

    def step(
        self,
        H_inc: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        de0: Tensor,
        cl: Tensor,
        iter: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Performs a strain increment with the radial return-mapping algorithm.

        In each increment, a trial stress is computed as

        $$
            \\pmb{\\sigma}_{\\text{trial}} = \\pmb{\\sigma}_n
                + \\mathbb{C} : \\Delta\\pmb{\\varepsilon}
        $$

        and the yield condition is checked via the flow potential

        $$
            f = \\|\\pmb{\\sigma}'_{\\text{trial}}\\|
                - \\sqrt{\\tfrac{2}{3}} \\, \\sigma_f(q)
        $$

        where $\\pmb{\\sigma}'$ denotes the deviatoric stress.

        **Elastic step** ($f \\le 0$): The trial stress is accepted.

        **Plastic step** ($f > 0$): The flow direction is
        $\\mathbf{n} = \\pmb{\\sigma}'_{\\text{trial}}
        / \\|\\pmb{\\sigma}'_{\\text{trial}}\\|$
        and the updates are

        $$
            \\pmb{\\sigma}_{n+1} = \\pmb{\\sigma}_{\\text{trial}}
                - 2 G \\, \\Delta\\gamma \\, \\mathbf{n}, \\quad
            q_{n+1} = q_n + \\sqrt{\\tfrac{2}{3}} \\, \\Delta\\gamma.
        $$

        The consistent (algorithmic) tangent is

        $$
            \\mathbb{C}^{\\text{alg}} = \\mathbb{C}
                - \\frac{2 G}{1 + \\frac{\\sigma_f'}{3 G}}
                  \\, \\mathbf{n} \\otimes \\mathbf{n}
                - \\frac{4 G^2 \\Delta\\gamma}
                  {\\|\\pmb{\\sigma}'_{\\text{trial}}\\|}
                  \\left( \\mathbb{I}^{\\text{dev}}
                  - \\mathbf{n} \\otimes \\mathbf{n} \\right)
        $$

        with $\\mathbb{I}^{\\text{dev}}_{ijkl} = \\frac{1}{2}(
        \\delta_{ik}\\delta_{jl} + \\delta_{il}\\delta_{jk})
        - \\frac{1}{3}\\delta_{ij}\\delta_{kl}$.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                - Shape: `(..., 3, 3)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                - Shape: `(..., 3, 3)`, same as `H_inc`.
            stress (Tensor): Current Cauchy stress tensor.
                - Shape: `(..., 3, 3)`.
            state (Tensor): Internal state variables, here: equivalent plastic strain.
                - Shape: `(..., 1)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                - Shape: `(..., 3, 3)`.
            cl (Tensor): Characteristic lengths.
                - Shape: `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated Cauchy stress tensor after plastic
                    update. Shape: `(..., 3, 3)`.
                - **state_new (Tensor)**: Updated internal state with updated plastic
                    strain. Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                    Shape: `(..., 3, 3, 3, 3)`.
        """
        # Second order identity tensor
        I2 = torch.eye(H_inc.shape[-1])
        # Compute small strain tensor
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)

        # Initialize solution variables
        stress_new = stress.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ddsdde = self.C.clone()

        # Compute trial stress
        s_trial = stress + torch.einsum("...ijkl,...kl->...ij", self.C, de - de0)

        # Compute the deviatoric trial stress
        s_trial_trace = s_trial[..., 0, 0] + s_trial[..., 1, 1] + s_trial[..., 2, 2]
        dev = s_trial.clone()
        dev[..., 0, 0] -= s_trial_trace / 3
        dev[..., 1, 1] -= s_trial_trace / 3
        dev[..., 2, 2] -= s_trial_trace / 3
        dev_norm = torch.linalg.norm(dev, dim=(-1, -2))

        # Flow potential
        f = dev_norm - sqrt(2.0 / 3.0) * self.sigma_f(q)
        fm = f > 0

        # Direction of flow
        n = dev[fm] / dev_norm[fm][..., None, None]

        # Local Newton solver to find plastic strain increment
        dGamma = torch.zeros_like(f[fm])
        G = self.G[fm]
        for _ in range(self.max_iter):
            res = (
                dev_norm[fm] - 2.0 * G * dGamma - sqrt(2.0 / 3.0) * self.sigma_f(q[fm])
            )
            ddGamma = res / (2.0 * G + 2.0 / 3.0 * self.sigma_f_prime(q[fm]))
            dGamma += ddGamma
            q[fm] += sqrt(2.0 / 3.0) * ddGamma

            # Check convergence for early stopping
            if (torch.abs(res) < self.tolerance).all():
                break

        # Check if the local Newton iteration converged
        if (torch.abs(res) > self.tolerance).any():
            print("Local Newton iteration did not converge")

        # Update stress
        stress_new[~fm] = s_trial[~fm]
        stress_new[fm] = s_trial[fm] - (2.0 * G * dGamma)[:, None, None] * n

        # Update state
        state_new[..., 0] = q

        # Update algorithmic tangent
        A = 2.0 * G / (1.0 + self.sigma_f_prime(q[fm]) / (3.0 * G))
        B = 4.0 * G**2 * dGamma / dev_norm[fm]
        I2 = torch.eye(3)
        I4 = torch.einsum("ij,kl->ijkl", I2, I2)
        I4S = torch.einsum("ik,jl->ijkl", I2, I2) + torch.einsum("il,jk->ijkl", I2, I2)
        nn = torch.einsum("...ij,...kl->...ijkl", n, n)
        ddsdde[fm] = (
            self.C[fm]
            - A[..., None, None, None, None] * nn
            - B[..., None, None, None, None] * (1 / 2 * I4S - 1 / 3 * I4 - nn)
        )

        return stress_new, state_new, ddsdde


class IsotropicPlasticityPlaneStress(IsotropicElasticityPlaneStress):
    """Isotropic elastoplastic material for plane stress problems.

    This class extends `IsotropicElasticityPlaneStress` to incorporate
    isotropic plasticity with a von Mises yield criterion under the plane
    stress constraint.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor | float): Poisson's ratio.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        sigma_f (Callable): Yield stress function $\\sigma_f(q)$.
        sigma_f_prime (Callable): Derivative $\\sigma_f'(q)$.
        tolerance (float): Convergence tolerance. Default is `1e-5`.
        max_iter (int): Maximum Newton iterations. Default is `10`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption with plane stress condition.
        - One internal state variable (``n_state = 1``): equivalent plastic
          strain $q$.
        - Implementation follows de Souza Neto et al. , Box 9.3-9.6.

    References:
        de Souza Neto, E. A., Peri, D., Owen, D. R. J.
        *Computational Methods for Plasticity*, Chapter 9,
        https://doi.org/10.1002/9780470694626.ch9, 2008.
    """

    def __init__(
        self,
        E: float | Tensor,
        nu: float | Tensor,
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, nu, rho)
        self._C = stiffness2voigt(self.C)
        self._S = torch.linalg.inv(self._C)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 1
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int) -> IsotropicPlasticityPlaneStress:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicPlasticityPlaneStress: A new material instance with vectorized
                properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicPlasticityPlaneStress(
                E,
                nu,
                self.sigma_f,
                self.sigma_f_prime,
                self.tolerance,
                self.max_iter,
                rho,
            )

    def step(
        self,
        H_inc: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        de0: Tensor,
        cl: Tensor,
        iter: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Performs a strain increment with the plane stress return-mapping algorithm.

        A trial stress is computed as
        $\\pmb{\\sigma}_{\\text{trial}} = \\pmb{\\sigma}_n
        + \\mathbb{C} : \\Delta\\pmb{\\varepsilon}$
        and the squared flow potential in Voigt notation is evaluated as

        $$
            \\Psi = \\tfrac{1}{2} \\pmb{\\sigma}_{\\text{trial}}^\\top
                    \\mathbf{P} \\, \\pmb{\\sigma}_{\\text{trial}}
                    - \\tfrac{1}{3} \\sigma_f(q)^2
        $$

        with

        $$
            \\mathbf{P} = \\frac{1}{3}
                \\begin{bmatrix} 2 & -1 & 0 \\\\ -1 & 2 & 0 \\\\ 0 & 0 & 6
                \\end{bmatrix}.
        $$

        If $\\Psi > 0$, the stress is updated via

        $$
            \\pmb{\\sigma}_{n+1}
                = [\\mathbb{S} + \\Delta\\gamma \\, \\mathbf{P}]^{-1}
                  \\, \\mathbb{S} \\, \\pmb{\\sigma}_{\\text{trial}}
        $$

        and the algorithmic tangent is

        $$
            \\mathbb{C}^{\\text{alg}}
                = [\\mathbb{S} + \\Delta\\gamma \\, \\mathbf{P}]^{-1}
                - \\alpha \\, \\mathbf{n} \\otimes \\mathbf{n}.
        $$

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                Shape: `(..., 2, 2)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                Shape: `(..., 2, 2)`, same as `H_inc`.
            stress (Tensor): Current Cauchy stress tensor.
                Shape: `(..., 2, 2)`.
            state (Tensor): Internal state variables, here: equivalent plastic strain.
                Shape: `(..., 1)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                Shape: `(..., 2, 2)`.
            cl (Tensor): Characteristic lengths.
                Shape: `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated Cauchy stress tensor after plastic
                    update. Shape: `(..., 2, 2)`.
                - **state_new (Tensor)**: Updated internal state with updated plastic
                    strain. Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                    Shape: `(..., 2, 2, 2, 2)`.
        """

        # Projection operator
        P = 1 / 3 * torch.tensor([[2, -1, 0], [-1, 2, 0], [0, 0, 6]])

        # Compute small strain tensor in Voigt notation
        depsilon = strain2voigt(0.5 * (H_inc.transpose(-1, -2) + H_inc) - de0)
        # Convert stress to Voigt notation
        stress_voigt = stress2voigt(stress)

        # Solution variables
        stress_new = stress_voigt.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ddsdde = self._C.clone()

        # Compute trial stress
        s_trial = stress_voigt + torch.einsum("...kl,...l->...k", self._C, depsilon)

        # Flow potential
        a1 = (s_trial[..., 0] + s_trial[..., 1]) ** 2
        a2 = (s_trial[..., 1] - s_trial[..., 0]) ** 2
        a3 = s_trial[..., 2] ** 2
        xi_trial = 1 / 6 * a1 + 1 / 2 * a2 + 2 * a3
        psi = 1 / 2 * xi_trial - 1 / 3 * self.sigma_f(q) ** 2

        # Flow mask
        fm = psi > 0

        # Local Newton solver to find plastic strain increment
        dGamma = torch.zeros_like(psi[fm])
        E = self.E[fm]
        G = self.G[fm]
        nu = self.nu[fm]
        for j in range(self.max_iter):
            # Compute xi and some short hands
            xi = (
                a1[fm] / (6 * (1 + E * dGamma / (3 * (1 - nu))) ** 2)
                + (1 / 2 * a2[fm] + 2 * a3[fm]) / (1 + 2 * G * dGamma) ** 2
            )
            sxi = torch.sqrt(xi)
            qq = q[fm] + dGamma * torch.sqrt(2 / 3 * xi)

            # Compute residual
            res = 1 / 2 * xi - 1 / 3 * self.sigma_f(qq) ** 2

            # Compute derivative of residual w.r.t dGamma
            H = self.sigma_f_prime(qq)
            xi_p = (
                -a1[fm] / (9 * (1 + E * dGamma / (3 * (1 - nu))) ** 3) * E / (1 - nu)
                - 2 * G * (a2[fm] + 4 * a3[fm]) / (1 + 2 * G * dGamma) ** 3
            )
            H_p = (
                2
                * self.sigma_f(qq)
                * H
                * sqrt(2 / 3)
                * (sxi + dGamma * xi_p / (2 * sxi))
            )
            res_prime = 1 / 2 * xi_p - 1 / 3 * H_p

            # Update dGamma
            dGamma -= res / res_prime

            if (torch.abs(res) < self.tolerance).all():
                break
        if (torch.abs(res) > self.tolerance).any():
            print("Local Newton iteration did not converge.")

        # Compute inverse operator
        inv = torch.linalg.inv(self._S[fm] + dGamma[:, None, None] * P)

        # Update stress
        stress_new[~fm] = s_trial[~fm]
        stress_new[fm] = (inv @ self._S[fm] @ s_trial[fm][:, :, None]).squeeze(-1)

        # Update state
        q[fm] = qq
        state_new[..., 0] = q

        # Update algorithmic tangent.
        s = stress_new[fm][:, :, None]
        xi = (s.transpose(-1, -2) @ P @ s).squeeze(-1).squeeze(-1)
        H = self.sigma_f_prime(q[fm])
        n = inv @ P @ s
        denom = (s.transpose(-1, -2) @ P @ n).squeeze(-1).squeeze(-1) + 2 * xi * H / (
            3 - 2 * H * dGamma
        )
        alpha = (1.0 / denom)[:, None, None]
        ddsdde[fm] = inv - alpha * n @ n.transpose(-1, -2)

        return voigt2stress(stress_new), state_new, voigt2stiffness(ddsdde)


class IsotropicPlasticityPlaneStrain(IsotropicElasticityPlaneStrain):
    """Isotropic elastoplastic material for plane strain problems.

    This class extends `IsotropicElasticityPlaneStrain` to incorporate
    isotropic plasticity with a von Mises yield criterion under the plane
    strain constraint.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor | float): Poisson's ratio.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        sigma_f (Callable): Yield stress function $\\sigma_f(q)$.
        sigma_f_prime (Callable): Derivative $\\sigma_f'(q)$.
        tolerance (float): Convergence tolerance. Default is `1e-5`.
        max_iter (int): Maximum Newton iterations. Default is `10`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption with plane strain condition.
        - Two internal state variables (``n_state = 2``): equivalent plastic
          strain $q$ and out-of-plane plastic strain $\\varepsilon_z^p$.
        - Supports batched/vectorized material parameters.
    """

    def __init__(
        self,
        E: float | Tensor,
        nu: float | Tensor,
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, nu, rho)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 2
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int) -> IsotropicPlasticityPlaneStrain:
        """Returns a vectorized copy of the material for `n_elem` elements.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicPlasticityPlaneStrain: A new material instance with vectorized
                properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicPlasticityPlaneStrain(
                E,
                nu,
                self.sigma_f,
                self.sigma_f_prime,
                self.tolerance,
                self.max_iter,
                rho,
            )

    def step(
        self,
        H_inc: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        de0: Tensor,
        cl: Tensor,
        iter: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Performs a strain increment with the plane strain return-mapping algorithm.

        The return mapping expands the in-plane stress to a full 3D tensor
        with the out-of-plane stress
        $$
            \\sigma_{zz} = \\nu (\\sigma_{xx} + \\sigma_{yy})
                - E \\, \\varepsilon_z^p
        $$
        and performs the deviatoric yield check and radial return in 3D.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                Shape: `(..., 2, 2)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                Shape: `(..., 2, 2)`, same as `H_inc`.
            stress (Tensor): Current Cauchy stress tensor.
                Shape: `(..., 2, 2)`.
            state (Tensor): Internal state variables, here: equivalent plastic strain
                and stress in the third direction. Shape: `(..., 2)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                Shape: `(..., 2, 2)`.
            cl (Tensor): Characteristic lengths.
                Shape: `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated Cauchy stress tensor after plastic
                    update. Shape: `(..., 2, 2)`.
                - **state_new (Tensor)**: Updated internal state with updated plastic
                    strain. Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                    Shape: `(..., 2, 2, 2, 2)`.
        """
        # Compute small strain tensor in Voigt notation
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)

        # Solution variables
        stress_new = stress.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ez = state_new[..., 1]
        ddsdde = self.C.clone()

        # Compute trial stress
        s_2D = stress + torch.einsum("...ijkl,...kl->...ij", self.C, de - de0)
        s_trial = torch.zeros(stress.shape[0], 3, 3)
        s_trial[..., :2, :2] = s_2D
        s_trial[..., 2, 2] = self.nu * (s_2D[..., 0, 0] + s_2D[..., 1, 1]) - self.E * ez

        # Compute the deviatoric trial stress
        s_trial_trace = s_trial[..., 0, 0] + s_trial[..., 1, 1] + s_trial[..., 2, 2]
        dev = s_trial.clone()
        dev[..., 0, 0] -= s_trial_trace / 3
        dev[..., 1, 1] -= s_trial_trace / 3
        dev[..., 2, 2] -= s_trial_trace / 3
        dev_norm = torch.linalg.norm(dev, dim=(-1, -2))

        # Flow potential
        f = dev_norm - sqrt(2.0 / 3.0) * self.sigma_f(q)
        fm = f > 0

        # Direction of flow
        n = dev[fm] / dev_norm[fm][..., None, None]

        # Local Newton solver to find plastic strain increment
        dGamma = torch.zeros_like(f[fm])
        G = self.G[fm]
        for _ in range(self.max_iter):
            res = (
                dev_norm[fm] - 2.0 * G * dGamma - sqrt(2.0 / 3.0) * self.sigma_f(q[fm])
            )
            ddGamma = res / (2.0 * G + 2.0 / 3.0 * self.sigma_f_prime(q[fm]))
            dGamma += ddGamma
            q[fm] += sqrt(2.0 / 3.0) * ddGamma

            # Check convergence for early stopping
            if (torch.abs(res) < self.tolerance).all():
                break

        # Check if the local Newton iteration converged
        if (torch.abs(res) > self.tolerance).any():
            print("Local Newton iteration did not converge")

        # Update stress
        stress_new[~fm] = s_trial[~fm][..., :2, :2]
        stress_new[fm] = (s_trial[fm] - (2.0 * G * dGamma)[:, None, None] * n)[
            ..., :2, :2
        ]

        # Update state
        state_new[..., 0] = q
        ez[fm] += dGamma * n[..., 2, 2]
        state_new[..., 1] = ez

        # Update algorithmic tangent
        A = 2.0 * G / (1.0 + self.sigma_f_prime(q[fm]) / (3.0 * G))
        B = 4.0 * G**2 * dGamma / dev_norm[fm]
        n0n1 = n[:, 0, 0] * n[:, 1, 1]
        ddsdde[fm, 0, 0, 0, 0] += -A * n[:, 0, 0] ** 2 - B * (2 / 3 - n[:, 0, 0] ** 2)
        ddsdde[fm, 1, 1, 1, 1] += -A * n[:, 1, 1] ** 2 - B * (2 / 3 - n[:, 1, 1] ** 2)
        ddsdde[fm, 0, 0, 1, 1] += -A * n0n1 - B * (-1 / 3 - n0n1)
        ddsdde[fm, 1, 1, 0, 0] += -A * n0n1 - B * (-1 / 3 - n0n1)
        ddsdde[fm, 0, 1, 0, 1] += -A * n[:, 0, 1] ** 2 - B * (1 / 2 - n[:, 0, 1] ** 2)
        ddsdde[fm, 0, 1, 1, 0] += -A * n[:, 0, 1] ** 2 - B * (1 / 2 - n[:, 1, 0] ** 2)
        ddsdde[fm, 1, 0, 0, 1] += -A * n[:, 1, 0] ** 2 - B * (1 / 2 - n[:, 0, 1] ** 2)
        ddsdde[fm, 1, 0, 1, 0] += -A * n[:, 1, 0] ** 2 - B * (1 / 2 - n[:, 1, 0] ** 2)

        return stress_new, state_new, ddsdde


class IsotropicPlasticity1D(IsotropicElasticity1D):
    """Isotropic elastoplastic material in 1D.

    This class extends `IsotropicElasticity1D` with isotropic hardening
    plasticity using a return-mapping algorithm.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        sigma_f (Callable): Yield stress function $\\sigma_f(q)$.
        sigma_f_prime (Callable): Derivative $\\sigma_f'(q)$.
        tolerance (float): Convergence tolerance. Default is `1e-5`.
        max_iter (int): Maximum Newton iterations. Default is `10`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - One internal state variable (``n_state = 1``): equivalent plastic
          strain $q$.
    """

    def __init__(
        self,
        E: float | Tensor,
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, rho)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 1
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int) -> IsotropicPlasticity1D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicPlasticity1D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicPlasticity1D(
                E, self.sigma_f, self.sigma_f_prime, self.tolerance, self.max_iter, rho
            )

    def step(
        self,
        H_inc: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        de0: Tensor,
        cl: Tensor,
        iter: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Performs a strain increment with the 1D return-mapping algorithm.

        A trial stress is computed as
        $\\sigma_{\\text{trial}} = \\sigma_n + E \\, \\Delta\\varepsilon$.
        The yield condition is $|\\sigma_{\\text{trial}}| \\le \\sigma_f(q)$.

        If yielding occurs, the plastic multiplier $\\Delta\\gamma$ is found
        by Newton's method on
        $$
            |\\sigma_{\\text{trial}}| - E \\, \\Delta\\gamma
                - \\sigma_f(q) = 0
        $$
        and the stress is updated as
        $$
            \\sigma_{n+1} = \\left(1
                - \\frac{E \\, \\Delta\\gamma}
                  {|\\sigma_{\\text{trial}}|}\\right)
                \\sigma_{\\text{trial}}, \\quad
            q_{n+1} = q_n + \\Delta\\gamma.
        $$
        The algorithmic tangent in the plastic regime is
        $\\frac{E \\, \\sigma_f'}{E + \\sigma_f'}$.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                *Shape:* `(..., 1, 1)`.
            F (Tensor): Current deformation gradient.
                *Shape:* `(..., 1, 1)`.
            stress (Tensor): Current stress tensor.
                *Shape:* `(..., 1, 1)`.
            state (Tensor): Internal state variables (equivalent plastic strain $q$).
                *Shape:* `(..., 1)`.
            de0 (Tensor): External strain increment (e.g., thermal).
                *Shape:* `(..., 1, 1)`.
            cl (Tensor): Characteristic lengths.
                *Shape:* `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            stress_new (Tensor): Updated stress tensor.
                *Shape:* `(..., 1, 1)`.
            state_new (Tensor): Updated internal state with updated plastic strain.
                *Shape:* `(..., 1)`.
            ddsdde (Tensor): Algorithmic tangent stiffness tensor.
                *Shape:* `(..., 1, 1, 1, 1)`.
        """
        # Solution variables
        stress_new = stress.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ddsdde = self.C.clone()

        # Compute trial stress
        s_trial = stress + torch.einsum("...ijkl,...kl->...ij", self.C, H_inc - de0)
        s_norm = torch.abs(s_trial).squeeze()

        # Flow potential
        f = s_norm - self.sigma_f(q)
        fm = f > 0

        # Local Newton solver to find plastic strain increment
        dGamma = torch.zeros_like(f[fm])
        E = self.E[fm]
        for _ in range(self.max_iter):
            res = s_norm[fm] - E * dGamma - self.sigma_f(q[fm])
            ddGamma = res / (E + self.sigma_f_prime(q[fm]))
            dGamma += ddGamma
            q[fm] += ddGamma

            # Check convergence for early stopping
            if (torch.abs(res) < self.tolerance).all():
                break

        # Check if the local Newton iteration converged
        if (torch.abs(res) > self.tolerance).any():
            print("Local Newton iteration did not converge.")

        # Update stress
        stress_new[~fm] = s_trial[~fm]
        stress_new[fm] = (1.0 - (dGamma * E) / s_norm[fm])[:, None, None] * s_trial[fm]

        # Update state
        state_new[..., 0] = q

        # Update algorithmic tangent
        if fm.sum() > 0:
            ddsdde[fm] = (
                E[:, None, None, None, None]
                * self.sigma_f_prime(q[fm])
                / (E[:, None, None, None, None] + self.sigma_f_prime(q[fm]))
            )

        return stress_new, state_new, ddsdde
