import torch


class Isotropic:
    def __init__(self, E: float, nu: float):
        self._E = E
        self._nu = nu

    def E(self) -> float:
        """Young's modulus."""
        return self._E

    def nu(self) -> float:
        """Poisson's ration."""
        return self._nu

    def lbd(self) -> float:
        """Lamè parameter."""
        return (self._E * self._nu) / ((1.0 + self._nu) * (1.0 - 2.0 * self._nu))

    def G(self) -> float:
        """Shear modulus."""
        return self._E / (2.0 * (1.0 + self._nu))

    def K(self) -> float:
        """Bulk modulus."""
        return self._E / (3.0 * (1.0 - 2.0 * self._nu))

    def C(self) -> torch.Tensor:
        """Stiffness tensor in Voigt notation."""

        lbd = self.lbd()
        G = self.G()

        # Return stiffness tensor
        return torch.tensor(
            [
                [lbd + 2.0 * G, lbd, lbd, 0.0, 0.0, 0],
                [lbd, lbd + 2.0 * G, lbd, 0.0, 0.0, 0],
                [lbd, lbd, lbd + 2.0 * G, 0.0, 0.0, 0],
                [0.0, 0.0, 0.0, G, 0.0, 0],
                [0.0, 0.0, 0.0, 0.0, G, 0],
                [0.0, 0.0, 0.0, 0.0, 0.0, G],
            ]
        )

    def Cs(self) -> torch.Tensor:
        """Shear stiffness matrix for shells."""
        return torch.tensor([[self.G(), 0], [0.0, self.G()]])


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
