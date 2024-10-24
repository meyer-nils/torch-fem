from typing import Union

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
        return Isotropic(E, nu, eps0)

    def step(self, depsilon, epsilon, sigma):
        """Perform a strain increment."""
        epsilon_new = epsilon + depsilon
        sigma_new = sigma + torch.einsum("...ij,...j->...i", self.C, depsilon)
        ddsdde = self.C
        return epsilon_new, sigma_new, ddsdde


class IsotropicPlaneStress(Isotropic):
    """Isotropic 2D plane stress material."""

    def C(self) -> torch.Tensor:
        """Returns a plane stress stiffness tensor in Voigt notation."""
        fac = self._E / (1.0 - self._nu**2)
        return fac * torch.tensor(
            [
                [1.0, self._nu, 0.0],
                [self._nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - self._nu)],
            ]
        )


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
