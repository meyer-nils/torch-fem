from abc import ABC, abstractmethod
from math import sqrt
from typing import Callable, Union

import torch
from torch import Tensor


class Material(ABC):
    """Base class for material models."""

    @abstractmethod
    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        pass

    @abstractmethod
    def step(self, depsilon: Tensor, epsilon: Tensor, sigma: Tensor, state: Tensor):
        """Perform a strain increment."""
        pass


class IsotropicElasticity3D(Material):
    def __init__(
        self,
        E: Union[float, Tensor],
        nu: Union[float, Tensor],
        eps0: Union[float, Tensor] = 0.0,
    ):
        # Convert float inputs to tensors
        if isinstance(E, float):
            E = torch.tensor(E)
        if isinstance(nu, float):
            nu = torch.tensor(nu)
        if isinstance(eps0, float):
            eps0 = torch.tensor(eps0)

        # Store material properties
        self.E = E
        self.nu = nu
        self.eps0 = eps0

        # There are no internal variables
        self.n_state = 0

        # Lame parameters
        self.lbd = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.G = self.E / (2.0 * (1.0 + self.nu))

        # Stiffness tensor
        z = torch.zeros_like(self.E)
        diag = self.lbd + 2.0 * self.G
        self.C = torch.stack(
            [
                torch.stack([diag, self.lbd, self.lbd, z, z, z], dim=-1),
                torch.stack([self.lbd, diag, self.lbd, z, z, z], dim=-1),
                torch.stack([self.lbd, self.lbd, diag, z, z, z], dim=-1),
                torch.stack([z, z, z, self.G, z, z], dim=-1),
                torch.stack([z, z, z, z, self.G, z], dim=-1),
                torch.stack([z, z, z, z, z, self.G], dim=-1),
            ],
            dim=-1,
        )

        # Stiffness tensor for shells
        self.Cs = torch.stack(
            [torch.stack([self.G, z], dim=-1), torch.stack([z, self.G], dim=-1)], dim=-1
        )

    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        E = self.E.repeat(n_elem)
        nu = self.nu.repeat(n_elem)
        eps0 = self.eps0.repeat(n_elem)
        return IsotropicElasticity3D(E, nu, eps0)

    def step(self, depsilon: Tensor, epsilon: Tensor, sigma: Tensor, state: Tensor):
        """Perform a strain increment."""
        epsilon_new = epsilon + depsilon
        sigma_new = sigma + torch.einsum("...ij,...j->...i", self.C, depsilon)
        state_new = state
        ddsdde = self.C
        return epsilon_new, sigma_new, state_new, ddsdde


class IsotropicPlasticity3D(IsotropicElasticity3D):
    """Isotropic plasticity with isotropic hardening"""

    def __init__(
        self,
        E: Union[float, Tensor],
        nu: Union[float, Tensor],
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
    ):
        super().__init__(E, nu)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 1
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        E = self.E.repeat(n_elem)
        nu = self.nu.repeat(n_elem)
        return IsotropicPlasticity3D(E, nu, self.sigma_f, self.sigma_f_prime)

    def step(self, depsilon: Tensor, epsilon: Tensor, sigma: Tensor, state: Tensor):
        """Perform a strain increment."""
        # Solution variables
        epsilon_new = epsilon + depsilon
        sigma_new = sigma.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ddsdde = self.C.clone()

        # Compute trial stress
        s_trial = sigma + torch.einsum("...kl,...l->...k", self.C, depsilon)

        # Compute the deviatoric trial stress
        s_trial_trace = s_trial[..., 0] + s_trial[..., 1] + s_trial[..., 2]
        dev = s_trial.clone()
        dev[..., 0:3] -= s_trial_trace[..., None] / 3
        dev_norm = torch.sqrt((dev[..., 0:3] ** 2 + 2 * dev[..., 3:6] ** 2).sum(dim=-1))

        # Flow potential
        f = dev_norm - sqrt(2.0 / 3.0) * self.sigma_f(q)
        fm = f > 0

        # Direction of flow
        n = dev[fm] / dev_norm[fm][..., None]

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
        sigma_new[~fm] = s_trial[~fm]
        sigma_new[fm] = s_trial[fm] - (2.0 * G * dGamma)[:, None] * n

        # Update state
        state_new[..., 0] = q

        # Update algorithmic tangent
        A = 2.0 * G / (1.0 + self.sigma_f_prime(q[fm]) / (3.0 * G))
        B = 4.0 * G**2 * dGamma / dev_norm[fm]
        C = self.C[fm]
        D = C.clone()
        D[:, 0, 0] = C[:, 0, 0] - A * n[:, 0] ** 2 - B * (2 / 3 - n[:, 0] ** 2)
        D[:, 1, 1] = C[:, 1, 1] - A * n[:, 1] ** 2 - B * (2 / 3 - n[:, 1] ** 2)
        D[:, 2, 2] = C[:, 2, 2] - A * n[:, 2] ** 2 - B * (2 / 3 - n[:, 2] ** 2)
        n0n1 = n[:, 0] * n[:, 1]
        D[:, 0, 1] = C[:, 0, 1] - A * n0n1 - B * (-1 / 3 - n0n1)
        D[:, 1, 0] = D[:, 0, 1]
        n0n2 = n[:, 0] * n[:, 2]
        D[:, 0, 2] = C[:, 0, 2] - A * n0n2 - B * (-1 / 3 - n0n2)
        D[:, 2, 0] = D[:, 0, 2]
        n1n2 = n[:, 1] * n[:, 2]
        D[:, 1, 2] = C[:, 1, 2] - A * n1n2 - B * (-1 / 3 - n1n2)
        D[:, 2, 1] = D[:, 1, 2]
        D[:, 3, 3] = C[:, 3, 3] - A * n[:, 3] ** 2 - B * (1 / 2 - n[:, 3] ** 2)
        D[:, 4, 4] = C[:, 4, 4] - A * n[:, 4] ** 2 - B * (1 / 2 - n[:, 4] ** 2)
        D[:, 5, 5] = C[:, 5, 5] - A * n[:, 5] ** 2 - B * (1 / 2 - n[:, 5] ** 2)
        ddsdde[fm] = D

        return epsilon_new, sigma_new, state_new, ddsdde


class IsotropicElasticityPlaneStress(IsotropicElasticity3D):
    """Isotropic 2D plane stress material."""

    def __init__(self, E: Union[float, Tensor], nu: Union[float, Tensor]):
        super().__init__(E, nu)

        # Overwrite the 3D stiffness tensor with a 2D plane stress tensor
        fac = self.E / (1.0 - self.nu**2)
        zero = torch.zeros_like(self.E)
        self.C = torch.stack(
            [
                torch.stack([fac, fac * self.nu, zero], dim=-1),
                torch.stack([fac * self.nu, fac, zero], dim=-1),
                torch.stack([zero, zero, fac * 0.5 * (1.0 - self.nu)], dim=-1),
            ],
            dim=-1,
        )

    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        E = self.E.repeat(n_elem)
        nu = self.nu.repeat(n_elem)
        return IsotropicElasticityPlaneStress(E, nu)


class IsotropicPlasticityPlaneStress(IsotropicElasticityPlaneStress):
    """Isotropic 2D plane stress material with isotropic hardening."""

    def __init__(
        self,
        E: Union[float, Tensor],
        nu: Union[float, Tensor],
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
    ):
        super().__init__(E, nu)
        self.S = torch.linalg.inv(self.C)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 1
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        E = self.E.repeat(n_elem)
        nu = self.nu.repeat(n_elem)
        return IsotropicPlasticityPlaneStress(E, nu, self.sigma_f, self.sigma_f_prime)

    def step(self, depsilon: Tensor, epsilon: Tensor, sigma: Tensor, state: Tensor):
        """Perform a strain increment.

        See: de Souza Neto, E. A., Peri, D., Owen, D. R. J. *Computational Methods for
        Plasticity*, Chapter 9: Plane Stress Plasticity, 2008.
        https://doi.org/10.1002/9780470694626.ch9
        """

        P = 1 / 3 * torch.tensor([[2, -1, 0], [-1, 2, 0], [0, 0, 6]])

        # Solution variables
        epsilon_new = epsilon + depsilon
        sigma_new = sigma.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ddsdde = self.C.clone()

        # Compute trial stress
        s_trial = sigma + torch.einsum("...kl,...l->...k", self.C, depsilon)

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
        inv = torch.linalg.inv(self.S[fm] + dGamma[None, :, None, None] * P)

        # Update stress
        sigma_new[~fm] = s_trial[~fm]
        sigma_new[fm] = (inv @ self.S[fm] @ s_trial[fm][:, :, None]).squeeze(-1)

        # Update state
        q[fm] = qq
        state_new[..., 0] = q

        # Update algorithmic tangent
        xi = sigma_new[fm][:, :, None].transpose(-1, -2) @ P @ sigma_new[fm][:, :, None]
        H = self.sigma_f_prime(q[fm])
        n = inv @ P @ sigma_new[fm][:, :, None]
        alpha = 1.0 / (
            sigma_new[fm][:, :, None].transpose(-1, -2) @ P @ n
            + 2 * xi * H / (3 - 2 * H * dGamma[:, None, None])
        )
        ddsdde[fm] = inv - alpha * n @ n.transpose(-1, -2)

        return epsilon_new, sigma_new, state_new, ddsdde


class IsotropicElasticityPlaneStrain(IsotropicElasticity3D):
    """Isotropic 2D plane strain material."""

    def __init__(self, E: Union[float, Tensor], nu: Union[float, Tensor]):
        super().__init__(E, nu)

        # Overwrite the 3D stiffness tensor with a 2D plane stress tensor
        lbd = self.lbd
        G = self.G
        zero = torch.zeros_like(self.E)
        self.C = torch.stack(
            [
                torch.stack([2.0 * G + lbd, lbd, zero], dim=-1),
                torch.stack([lbd, 2.0 * G + lbd, zero], dim=-1),
                torch.stack([zero, zero, G], dim=-1),
            ],
            dim=-1,
        )

    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements and `n_int`
        integration points."""
        E = self.E.repeat(n_elem)
        nu = self.nu.repeat(n_elem)
        return IsotropicElasticityPlaneStrain(E, nu)


class OrthotropicElasticity3D(Material):
    """Orthotropic material."""

    def __init__(
        self,
        E_1: Union[float, Tensor],
        E_2: Union[float, Tensor],
        E_3: Union[float, Tensor],
        nu_12: Union[float, Tensor],
        nu_13: Union[float, Tensor],
        nu_23: Union[float, Tensor],
        G_12: Union[float, Tensor],
        G_13: Union[float, Tensor],
        G_23: Union[float, Tensor],
    ):
        # Convert float inputs to tensors
        if isinstance(E_1, float):
            E_1 = torch.tensor(E_1)
        if isinstance(E_2, float):
            E_2 = torch.tensor(E_2)
        if isinstance(E_3, float):
            E_3 = torch.tensor(E_3)
        if isinstance(nu_12, float):
            nu_12 = torch.tensor(nu_12)
        if isinstance(nu_13, float):
            nu_13 = torch.tensor(nu_13)
        if isinstance(nu_23, float):
            nu_23 = torch.tensor(nu_23)
        if isinstance(G_12, float):
            G_12 = torch.tensor(G_12)
        if isinstance(G_13, float):
            G_13 = torch.tensor(G_13)
        if isinstance(G_23, float):
            G_23 = torch.tensor(G_23)

        # Store material properties
        self.E_1 = E_1
        self.E_2 = E_2
        self.E_3 = E_3
        self.nu_12 = nu_12
        self.nu_21 = E_2 / E_1 * nu_12
        self.nu_13 = nu_13
        self.nu_31 = E_3 / E_1 * nu_13
        self.nu_23 = nu_23
        self.nu_32 = E_3 / E_2 * nu_23
        self.G_12 = G_12
        self.G_13 = G_13
        self.G_23 = G_23

        # There are no internal variables
        self.n_state = 0

        # Full stiffness tensor
        if self.E_1.dim() == 0:
            self._C = torch.zeros(3, 3, 3, 3)
            z = torch.tensor(0.0)
        else:
            self._C = torch.zeros(*E_1.shape, 3, 3, 3, 3)
            z = torch.zeros_like(self.E_1)
        F = 1 / (
            1
            - self.nu_12 * self.nu_21
            - self.nu_13 * self.nu_31
            - self.nu_23 * self.nu_32
            - 2 * self.nu_21 * self.nu_32 * self.nu_13
        )
        self._C[..., 0, 0, 0, 0] = self.E_1 * (1 - self.nu_23 * self.nu_32) * F
        self._C[..., 1, 1, 1, 1] = self.E_2 * (1 - self.nu_13 * self.nu_31) * F
        self._C[..., 2, 2, 2, 2] = self.E_3 * (1 - self.nu_12 * self.nu_21) * F
        self._C[..., 0, 0, 1, 1] = self.E_1 * (self.nu_21 + self.nu_31 * self.nu_23) * F
        self._C[..., 1, 1, 0, 0] = self._C[..., 0, 0, 1, 1]
        self._C[..., 0, 0, 2, 2] = self.E_1 * (self.nu_31 + self.nu_21 * self.nu_32) * F
        self._C[..., 2, 2, 0, 0] = self._C[..., 0, 0, 2, 2]
        self._C[..., 1, 1, 2, 2] = self.E_2 * (self.nu_32 + self.nu_12 * self.nu_31) * F
        self._C[..., 2, 2, 1, 1] = self._C[..., 1, 1, 2, 2]
        self._C[..., 0, 1, 0, 1] = self.G_12
        self._C[..., 1, 0, 1, 0] = self.G_12
        self._C[..., 0, 1, 1, 0] = self.G_12
        self._C[..., 1, 0, 0, 1] = self.G_12
        self._C[..., 0, 2, 0, 2] = self.G_13
        self._C[..., 2, 0, 2, 0] = self.G_13
        self._C[..., 0, 2, 2, 0] = self.G_13
        self._C[..., 2, 0, 0, 2] = self.G_13
        self._C[..., 1, 2, 1, 2] = self.G_23
        self._C[..., 2, 1, 2, 1] = self.G_23
        self._C[..., 1, 2, 2, 1] = self.G_23
        self._C[..., 2, 1, 1, 2] = self.G_23

        # Stiffness tensor in Voigt notation
        self.C = torch.stack(
            [
                torch.stack(
                    [
                        self._C[..., 0, 0, 0, 0],
                        self._C[..., 0, 0, 1, 1],
                        self._C[..., 0, 0, 2, 2],
                        z,
                        z,
                        z,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        self._C[..., 1, 1, 0, 0],
                        self._C[..., 1, 1, 1, 1],
                        self._C[..., 1, 1, 2, 2],
                        z,
                        z,
                        z,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        self._C[..., 2, 2, 0, 0],
                        self._C[..., 2, 2, 1, 1],
                        self._C[..., 2, 2, 2, 2],
                        z,
                        z,
                        z,
                    ],
                    dim=-1,
                ),
                torch.stack([z, z, z, self._C[..., 0, 1, 0, 1], z, z], dim=-1),
                torch.stack([z, z, z, z, self._C[..., 0, 2, 0, 2], z], dim=-1),
                torch.stack([z, z, z, z, z, self._C[..., 1, 2, 1, 2]], dim=-1),
            ],
            dim=-1,
        )

    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements and `n_int`
        integration points."""
        E_1 = self.E_1.repeat(n_elem)
        E_2 = self.E_2.repeat(n_elem)
        E_3 = self.E_3.repeat(n_elem)
        nu_12 = self.nu_12.repeat(n_elem)
        nu_13 = self.nu_13.repeat(n_elem)
        nu_23 = self.nu_23.repeat(n_elem)
        G_12 = self.G_12.repeat(n_elem)
        G_13 = self.G_13.repeat(n_elem)
        G_23 = self.G_23.repeat(n_elem)
        return OrthotropicElasticity3D(
            E_1, E_2, E_3, nu_12, nu_13, nu_23, G_12, G_13, G_23
        )

    def step(self, depsilon: Tensor, epsilon: Tensor, sigma: Tensor, state: Tensor):
        """Perform a strain increment."""
        epsilon_new = epsilon + depsilon
        sigma_new = sigma + torch.einsum("...ij,...j->...i", self.C, depsilon)
        state_new = state
        ddsdde = self.C
        return epsilon_new, sigma_new, state_new, ddsdde


class OrthotropicElasticityPlaneStress(Material):
    """Orthotropic 2D plane stress material."""

    def __init__(
        self,
        E_1: Union[float, Tensor],
        E_2: Union[float, Tensor],
        nu_12: Union[float, Tensor],
        G_12: Union[float, Tensor],
        G_13: Union[float, Tensor] = 0.0,
        G_23: Union[float, Tensor] = 0.0,
    ):
        # Convert float inputs to tensors
        if isinstance(E_1, float):
            E_1 = torch.tensor(E_1)
        if isinstance(E_2, float):
            E_2 = torch.tensor(E_2)
        if isinstance(nu_12, float):
            nu_12 = torch.tensor(nu_12)
        if isinstance(G_12, float):
            G_12 = torch.tensor(G_12)
        if isinstance(G_13, float):
            G_13 = torch.tensor(G_13)
        if isinstance(G_23, float):
            G_23 = torch.tensor(G_23)

        # Store material properties
        self.E_1 = E_1
        self.E_2 = E_2
        self.nu_12 = nu_12
        nu_21 = E_2 / E_1 * nu_12
        self.nu_21 = nu_21
        self.G_12 = G_12
        self.G_13 = G_13
        self.G_23 = G_23

        # There are no internal variables
        self.n_state = 0

        # Stiffness tensor
        z = torch.zeros_like(self.E_1)
        nu2 = self.nu_12 * self.nu_21
        self.C = torch.stack(
            [
                torch.stack([E_1 / (1 - nu2), nu_12 * E_2 / (1 - nu2), z], dim=-1),
                torch.stack([nu_21 * E_1 / (1 - nu2), E_2 / (1 - nu2), z], dim=-1),
                torch.stack([z, z, G_12], dim=-1),
            ],
            dim=-1,
        )

        # Transverse shear stiffness matrix for shells
        self.Cs = torch.stack(
            [torch.stack([self.G_13, z], dim=-1), torch.stack([z, self.G_23], dim=-1)],
            dim=-1,
        )

    def vectorize(self, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        E_1 = self.E_1.repeat(n_elem)
        E_2 = self.E_2.repeat(n_elem)
        nu_12 = self.nu_12.repeat(n_elem)
        G_12 = self.G_12.repeat(n_elem)
        G_13 = self.G_13.repeat(n_elem)
        G_23 = self.G_23.repeat(n_elem)
        return OrthotropicElasticityPlaneStress(E_1, E_2, nu_12, G_12, G_13, G_23)

    def step(self, depsilon: Tensor, epsilon: Tensor, sigma: Tensor, state: Tensor):
        """Perform a strain increment."""
        epsilon_new = epsilon + depsilon
        sigma_new = sigma + torch.einsum("...ij,...j->...i", self.C, depsilon)
        state_new = state
        ddsdde = self.C
        return epsilon_new, sigma_new, state_new, ddsdde
