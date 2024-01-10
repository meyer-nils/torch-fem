import torch


class Isotropic:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

    def E(self):
        """Young's modulus"""
        return self.E

    def nu(self):
        """Poisson's ration"""
        return self.nu

    def lbd(self):
        """Lamè parameter."""
        return (self.E * self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def G(self):
        """Shear modulus"""
        return self.E / (2.0 * (1.0 + self.nu))

    def K(self):
        """Bulk modulus."""
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    def C(self):
        """Stiffness tensor in notation

        C_xxxx C_xxyy C_xxzz C_xxyz C_xxxz C_xxxy
               C_yyyy C_yyzz C_yyyz C_yyxz C_yyxy
                      C_zzzz C_zzyz C_zzxz C_zzxy
                             C_yzyz C_yzxz C_yzxy
                                    C_xzxz C_xzxy
        symm.                              C_xyxy
        """

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

    def Cs(self):
        """Shear stiffness matrix for shells."""
        return torch.tensor([[self.G(), 0], [0.0, self.G()]])


class IsotropicPlaneStress(Isotropic):
    """Isotropic 2D plane stress material."""

    def C(self):
        """Returns a plane stress stiffness tensor in notation

        C_xxxx C_xxyy C_xxxy
               C_yyyy C_yyxy
        symm.         C_xyxy
        """
        fac = self.E / (1.0 - self.nu**2)
        return fac * torch.tensor(
            [
                [1.0, self.nu, 0.0],
                [self.nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - self.nu)],
            ]
        )


class IsotropicPlaneStrain(Isotropic):
    """Isotropic 2D plane strain material."""

    def C(self):
        """Returns a plane strain stiffness tensor in notation

        C_xxxx C_xxyy C_xxxy
               C_yyyy C_yyxy
        symm.         C_xyxy
        """
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

    def __init__(self, E_1, E_2, E_3, nu_12, nu_13, nu_23, G_12, G_13, G_23):
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

        self.C = torch.zeros(3, 3, 3, 3)
        Gamma = 1 / (
            1
            - self.nu_12 * self.nu_21
            - self.nu_13 * self.nu_31
            - self.nu_23 * self.nu_32
            - 2 * self.nu_21 * self.nu_32 * self.nu_13
        )
        self.C[0, 0, 0, 0] = self.E_1 * (1 - self.nu_23 * self.nu_32) * Gamma
        self.C[1, 1, 1, 1] = self.E_2 * (1 - self.nu_13 * self.nu_31) * Gamma
        self.C[2, 2, 2, 2] = self.E_3 * (1 - self.nu_12 * self.nu_21) * Gamma
        self.C[0, 0, 1, 1] = self.E_1 * (self.nu_21 + self.nu_31 * self.nu_23) * Gamma
        self.C[1, 1, 0, 0] = self.C[0, 0, 1, 1]
        self.C[0, 0, 2, 2] = self.E_1 * (self.nu_31 + self.nu_21 * self.nu_32) * Gamma
        self.C[2, 2, 0, 0] = self.C[0, 0, 2, 2]
        self.C[1, 1, 2, 2] = self.E_2 * (self.nu_32 + self.nu_12 * self.nu_31) * Gamma
        self.C[2, 2, 1, 1] = self.C[1, 1, 2, 2]
        self.C[0, 1, 0, 1] = self.G_12
        self.C[1, 0, 1, 0] = self.G_12
        self.C[0, 1, 1, 0] = self.G_12
        self.C[1, 0, 0, 1] = self.G_12
        self.C[0, 2, 0, 2] = self.G_13
        self.C[2, 0, 2, 0] = self.G_13
        self.C[0, 2, 2, 0] = self.G_13
        self.C[2, 0, 0, 2] = self.G_13
        self.C[1, 2, 1, 2] = self.G_23
        self.C[2, 1, 2, 1] = self.G_23
        self.C[1, 2, 2, 1] = self.G_23
        self.C[2, 1, 1, 2] = self.G_23

    def C(self):
        """Returns a stiffness tensor of an orthotropic material in the notation

        C_xxxx C_xxyy C_xxzz C_xxyz C_xxxz C_xxxy
               C_yyyy C_yyzz C_yyyz C_yyxz C_yyxy
                      C_zzzz C_zzyz C_zzxz C_zzxy
                             C_yzyz C_yzxz C_yzxy
                                    C_xzxz C_xzxy
        symm.                              C_xyxy

        If the shape is (3,3,3,3), it returns it as 3x3x3x3 tensor.
        """

        # Return stiffness tensor
        return torch.tensor(
            [
                [self.C[0, 0, 0, 0], self.C[0, 0, 1, 1], self.C[0, 0, 2, 2], 0, 0, 0],
                [self.C[1, 1, 0, 0], self.C[1, 1, 1, 1], self.C[1, 1, 2, 2], 0, 0, 0],
                [self.C[2, 2, 0, 0], self.C[2, 2, 1, 1], self.C[2, 2, 2, 2], 0, 0, 0],
                [0, 0, 0, self.C[0, 1, 0, 1], 0, 0],
                [0, 0, 0, 0, self.C[0, 2, 0, 2], 0],
                [0, 0, 0, 0, 0, self.C[1, 2, 1, 2]],
            ]
        )


class OrthotropicPlaneStress:
    """Orthotropic 2D plane stress material."""

    def __init__(self, E_1, E_2, nu_12, G_12, G_13=0.0, G_23=0.0):
        self.E_1 = E_1
        self.E_2 = E_2
        self.nu_12 = nu_12
        self.nu_21 = E_2 / E_1 * nu_12
        self.G_12 = G_12
        self.G_13 = G_13
        self.G_23 = G_23

    def C(self):
        """Returns a plane stress stiffness tensor in notation

        C_xxxx C_xxyy C_xxxy
               C_yyyy C_yyxy
        symm.         C_xyxy
        """
        nu2 = self.nu_12 * self.nu_21
        return torch.tensor(
            [
                [self.E_1 / (1 - nu2), self.nu_12 * self.E_2 / (1 - nu2), 0],
                [self.nu_21 * self.E_1 / (1 - nu2), self.E_2 / (1 - nu2), 0],
                [0, 0, self.G_12],
            ]
        )

    def Cs(self):
        """Shear stiffness matrix for shells."""
        return torch.tensor([[self.G_13, 0], [0.0, self.G_23]])
