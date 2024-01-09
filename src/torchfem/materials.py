import torch


class Isotropic:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

    def lbd(self):
        """Lamè parameter."""
        return (self.E * self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def G(self):
        """Shear modulus"""
        return self.E / (2.0 * (1.0 + self.nu))

    def K(self):
        """Bulk modulus."""
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    def Cs(self):
        """Shear stiffness matrix for shells."""
        return torch.tensor([[self.G(), 0], [0.0, self.G()]])


class Isotropic3D(Isotropic):
    """Isotropic 3D material."""

    def C(self):
        """Returns a stiffness tensor in notation

        C_xxxx C_xxyy C_xxzz C_xxyz C_xxxz C_xxxy
               C_yyyy C_yyzz C_yyyz C_yyxz C_yyxy
                      C_zzzz C_zzyz C_zzxz C_zzxy
                             C_yzyz C_yzxz C_yzxy
                                    C_xzxz C_xzxy
        symm.                              C_xyxy
        """
        # Compute Lamé parameters
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
