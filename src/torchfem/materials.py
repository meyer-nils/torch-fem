from math import sqrt
from typing import Callable, Union

import torch


class Isotropic:
    def __init__(
        self,
        E: Union[float, torch.Tensor],
        nu: Union[float, torch.Tensor],
        eps0: Union[float, torch.Tensor] = 0.0,
    ):
        # Convert float inputs to tensors
        if isinstance(E, float) and isinstance(nu, float) and isinstance(eps0, float):
            E = torch.tensor(E)
            nu = torch.tensor(nu)
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

    def vectorize(self, n_int: int, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements and `n_int`
        integration points."""
        E = self.E.repeat(n_int, n_elem)
        nu = self.nu.repeat(n_int, n_elem)
        eps0 = self.eps0.repeat(n_int, n_elem)
        return Isotropic(E, nu, eps0)

    def step(self, depsilon, epsilon, sigma, state):
        """Perform a strain increment."""
        epsilon_new = epsilon + depsilon
        sigma_new = sigma + torch.einsum("...ij,...j->...i", self.C, depsilon)
        state_new = state
        ddsdde = self.C
        return epsilon_new, sigma_new, state_new, ddsdde


class IsotropicPlasticity(Isotropic):
    """Isotropic plasticity with isotropic hardening"""

    def __init__(
        self,
        E: Union[float, torch.Tensor],
        nu: Union[float, torch.Tensor],
        sigma_f: Callable,
        sigma_f_prime: Callable,
    ):
        super().__init__(E, nu)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 1

    def vectorize(self, n_int: int, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements."""
        E = self.E.repeat(n_int, n_elem)
        nu = self.nu.repeat(n_int, n_elem)
        return IsotropicPlasticity(E, nu, self.sigma_f, self.sigma_f_prime)

    def step(self, depsilon, epsilon, sigma, state):
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
        for j in range(10):
            res = (
                dev_norm[fm] - 2.0 * G * dGamma - sqrt(2.0 / 3.0) * self.sigma_f(q[fm])
            )
            ddGamma = res / (2.0 * G + 2.0 / 3.0 * self.sigma_f_prime(q[fm]))
            dGamma += ddGamma
            q[fm] += sqrt(2.0 / 3.0) * ddGamma
        if (torch.abs(res) > 1e-10).any():
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
        D[:, 0, 1] = (
            C[:, 0, 1] - A * n[:, 0] * n[:, 1] - B * (-1 / 3 - n[:, 0] * n[:, 1])
        )
        D[:, 1, 0] = D[:, 0, 1]
        D[:, 0, 2] = (
            C[:, 0, 2] - A * n[:, 0] * n[:, 2] - B * (-1 / 3 - n[:, 0] * n[:, 2])
        )
        D[:, 2, 0] = D[:, 0, 2]
        D[:, 1, 2] = (
            C[:, 1, 2] - A * n[:, 1] * n[:, 2] - B * (-1 / 3 - n[:, 1] * n[:, 2])
        )
        D[:, 2, 1] = D[:, 1, 2]
        D[:, 3, 3] = C[:, 3, 3] - A * n[:, 3] ** 2 - B * (1 / 2 - n[:, 3] ** 2)
        D[:, 4, 4] = C[:, 4, 4] - A * n[:, 4] ** 2 - B * (1 / 2 - n[:, 4] ** 2)
        D[:, 5, 5] = C[:, 5, 5] - A * n[:, 5] ** 2 - B * (1 / 2 - n[:, 5] ** 2)
        ddsdde[fm] = D

        return epsilon_new, sigma_new, state_new, ddsdde


class IsotropicPlaneStress(Isotropic):
    """Isotropic 2D plane stress material."""

    def __init__(self, E: Union[float, torch.Tensor], nu: Union[float, torch.Tensor]):
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

    def vectorize(self, n_int: int, n_elem: int):
        """Create a vectorized copy of the material for `n_elm` elements and `n_int`
        integration points."""
        E = self.E.repeat(n_int, n_elem)
        nu = self.nu.repeat(n_int, n_elem)
        return IsotropicPlaneStress(E, nu)


class IsotropicPlaneStrain(Isotropic):
    """Isotropic 2D plane strain material."""

    def C(self) -> torch.Tensor:
        """Returns a plane strain stiffness tensor in Voigt notation."""
        lbd = self.lbd()
        G = self.G()
        return torch.tensor(
            [
                [2.0 * G + lbd, lbd, 0.0],
                [lbd, 2.0 * G + lbd, 0.0],
                [0.0, 0.0, G],
            ]
        )


class Orthotropic:
    """Orthotropic material."""

    def __init__(
        self,
        E_1: float,
        E_2: float,
        E_3: float,
        nu_12: float,
        nu_13: float,
        nu_23: float,
        G_12: float,
        G_13: float,
        G_23: float,
    ):
        self._E_1 = E_1
        self._E_2 = E_2
        self._E_3 = E_3
        self._nu_12 = nu_12
        self._nu_21 = E_2 / E_1 * nu_12
        self._nu_13 = nu_13
        self._nu_31 = E_3 / E_1 * nu_13
        self._nu_23 = nu_23
        self._nu_32 = E_3 / E_2 * nu_23
        self._G_12 = G_12
        self._G_13 = G_13
        self._G_23 = G_23

        self._C = torch.zeros(3, 3, 3, 3)
        F = 1 / (
            1
            - self._nu_12 * self._nu_21
            - self._nu_13 * self._nu_31
            - self._nu_23 * self._nu_32
            - 2 * self._nu_21 * self._nu_32 * self._nu_13
        )
        self._C[0, 0, 0, 0] = self._E_1 * (1 - self._nu_23 * self._nu_32) * F
        self._C[1, 1, 1, 1] = self._E_2 * (1 - self._nu_13 * self._nu_31) * F
        self._C[2, 2, 2, 2] = self._E_3 * (1 - self._nu_12 * self._nu_21) * F
        self._C[0, 0, 1, 1] = self._E_1 * (self._nu_21 + self._nu_31 * self._nu_23) * F
        self._C[1, 1, 0, 0] = self._C[0, 0, 1, 1]
        self._C[0, 0, 2, 2] = self._E_1 * (self._nu_31 + self._nu_21 * self._nu_32) * F
        self._C[2, 2, 0, 0] = self._C[0, 0, 2, 2]
        self._C[1, 1, 2, 2] = self._E_2 * (self._nu_32 + self._nu_12 * self._nu_31) * F
        self._C[2, 2, 1, 1] = self._C[1, 1, 2, 2]
        self._C[0, 1, 0, 1] = self._G_12
        self._C[1, 0, 1, 0] = self._G_12
        self._C[0, 1, 1, 0] = self._G_12
        self._C[1, 0, 0, 1] = self._G_12
        self._C[0, 2, 0, 2] = self._G_13
        self._C[2, 0, 2, 0] = self._G_13
        self._C[0, 2, 2, 0] = self._G_13
        self._C[2, 0, 0, 2] = self._G_13
        self._C[1, 2, 1, 2] = self._G_23
        self._C[2, 1, 2, 1] = self._G_23
        self._C[1, 2, 2, 1] = self._G_23
        self._C[2, 1, 1, 2] = self._G_23

    def C(self) -> torch.Tensor:
        """Returns a stiffness tensor of an orthotropic material in Voigt notation."""
        c = [
            [self._C[0, 0, 0, 0], self._C[0, 0, 1, 1], self._C[0, 0, 2, 2], 0, 0, 0],
            [self._C[1, 1, 0, 0], self._C[1, 1, 1, 1], self._C[1, 1, 2, 2], 0, 0, 0],
            [self._C[2, 2, 0, 0], self._C[2, 2, 1, 1], self._C[2, 2, 2, 2], 0, 0, 0],
            [0, 0, 0, self._C[0, 1, 0, 1], 0, 0],
            [0, 0, 0, 0, self._C[0, 2, 0, 2], 0],
            [0, 0, 0, 0, 0, self._C[1, 2, 1, 2]],
        ]
        return torch.tensor(c)


class OrthotropicPlaneStress:
    """Orthotropic 2D plane stress material."""

    def __init__(
        self,
        E_1: float,
        E_2: float,
        nu_12: float,
        G_12: float,
        G_13: float = 0.0,
        G_23: float = 0.0,
    ):
        self._E_1 = E_1
        self._E_2 = E_2
        self._nu_12 = nu_12
        self._nu_21 = E_2 / E_1 * nu_12
        self._G_12 = G_12
        self._G_13 = G_13
        self._G_23 = G_23

    def C(self) -> torch.Tensor:
        """Returns a plane stress stiffness tensor in Voigt notation."""
        nu2 = self._nu_12 * self._nu_21
        return torch.tensor(
            [
                [self._E_1 / (1 - nu2), self._nu_12 * self._E_2 / (1 - nu2), 0],
                [self._nu_21 * self._E_1 / (1 - nu2), self._E_2 / (1 - nu2), 0],
                [0, 0, self._G_12],
            ]
        )

    def Cs(self) -> torch.Tensor:
        """Shear stiffness matrix for shells."""
        return torch.tensor([[self._G_13, 0], [0.0, self._G_23]])
