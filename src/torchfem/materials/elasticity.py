from __future__ import annotations

import torch
from torch import Tensor

from ..utils import stiffness2voigt
from .base import Material


class IsotropicElasticity3D(Material):
    """Isotropic elastic material in 3D.

    Args:
        E (Tensor | float): Young's modulus. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor | float): Poisson's ratio. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        rho (Tensor | float): Mass density. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.

    Notes:
        - Small-strain assumption.
        - No internal state variables (``n_state = 0``).
        - Supports batched/vectorized material parameters.

    Info: Definition of stiffness tensor
        This class represents a 3D isotropic linear elastic material under small-strain
        assumptions, defined by (batched) Young's modulus $E$ and (batched) Poisson's
        ratio $\\nu$. The constructor derives the (batched) stiffness tensor
        $\\mathbb{C}$ with components
        $$
        C_{ijkl} = \\lambda \\delta_{ij} \\delta_{kl} + G (\\delta_{ik} \\delta_{jl}
            + \\delta_{il} \\delta_{jk})
        $$
        based on these parameters with
        $$
            \\lambda = \\frac{E \\nu}{(1 + \\nu)(1 - 2 \\nu)},
            \\quad G = \\frac{E}{2(1 + \\nu)}.
        $$
    """

    def __init__(
        self, E: Tensor | float, nu: Tensor | float, rho: Tensor | float = 1.0
    ):
        # Convert float inputs to tensors
        self.E = torch.as_tensor(E)
        self.nu = torch.as_tensor(nu)
        self.rho = torch.as_tensor(rho)

        # There are no internal variables
        self.n_state = 0

        # Check if the material is vectorized
        self.is_vectorized = self.E.dim() > 0

        # Lame parameters
        self.lbd = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.G = self.E / (2.0 * (1.0 + self.nu))

        # Identity tensors
        I2 = torch.eye(3)
        I4 = torch.einsum("ij,kl->ijkl", I2, I2)
        I4S = torch.einsum("ik,jl->ijkl", I2, I2) + torch.einsum("il,jk->ijkl", I2, I2)

        # Stiffness tensor
        lbd = self.lbd[..., None, None, None, None]
        G = self.G[..., None, None, None, None]
        self.C = lbd * I4 + G * I4S

    def vectorize(self, n_elem: int) -> IsotropicElasticity3D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicElasticity3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicElasticity3D(E, nu, rho)

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
        """Performs an incremental step in the small-strain isotropic elasticity model.

        This function updates the stress and algorithmic tangent stiffness based on a
        small-strain assumption and linear elasticity as
        $$
            \\pmb{\\sigma}_{n+1} = \\pmb{\\sigma}_n + \\mathbb{C} :
                (\\Delta \\pmb{\\varepsilon} - \\Delta \\pmb{\\varepsilon}^0)
        $$
        and
        $$
            \\frac{\\partial \\Delta \\pmb{\\sigma}}{\\partial \\Delta \\mathbf{H}} =
                \\mathbb{C}
        $$

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                *Shape:* `(..., d, d)`.
            F (Tensor): Current deformation gradient.
                *Shape:* `(..., d, d)`.
            stress (Tensor): Current Cauchy stress tensor.
                *Shape:* `(..., d, d)`.
            state (Tensor): Internal state variables (unused in linear elasticity).
                *Shape:* `(..., 0)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                *Shape:* `(..., d, d)`.
            cl (Tensor): Characteristic lengths.
                *Shape:* `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            stress_new (Tensor): Updated Cauchy stress tensor.
                *Shape:* `(..., d, d)`.
            state_new (Tensor): Updated internal state (unchanged).
                *Shape:* `(..., 0)`.
            ddsdde (Tensor): Algorithmic tangent stiffness tensor.
                *Shape:* `(..., d, d, d, d)`.
        """
        # Compute small strain tensor
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        # Compute new stress (1st PK = Cauchy for small strains)
        stress_new = stress + torch.einsum("...ijkl,...kl->...ij", self.C, de - de0)
        # Update internal state (this material does not change state)
        state_new = state
        # Algorithmic tangent
        ddsdde = self.C
        return stress_new, state_new, ddsdde


class IsotropicElasticityPlaneStress(IsotropicElasticity3D):
    """Isotropic elastic material for planar stress problems.

    Args:
        E (Tensor): Young's modulus. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor): Poisson's ratio. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        rho (Tensor): Mass density. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.

    Notes:
        - Small-strain assumption.
        - Plane stress condition ($\\sigma_{33} = 0$).
        - No internal state variables (``n_state = 0``).
        - Supports batched/vectorized material parameters.

    Info: Definition of plane stress stiffness tensor
        This class represents a 2D isotropic linear elastic material for plane stress
        under small-strain assumptions, defined by (batched) Young's modulus $E$ and
        (batched) Poisson's ratio $\\nu$. The constructor derives the (batched)
        stiffness tensor $\\mathbb{C}$ by enforcing the plane stress condition
        $\\sigma_{33} = 0$ as
        $$
            C_{0000} = C_{1111} = \\frac{E}{1 - \\nu^2}
        $$
        $$
            C_{0011} = C_{1100} = \\frac{E \\nu}{1 - \\nu^2}
        $$
        $$
            C_{0101} = C_{0110} = C_{1001} = C_{1010} = \\frac{E}{2(1 + \\nu)}.
        $$
    """

    def __init__(
        self, E: float | Tensor, nu: float | Tensor, rho: float | Tensor = 1.0
    ):
        super().__init__(E, nu, rho)

        # Overwrite the 3D stiffness tensor with a 2D plane stress tensor
        fac = self.E / (1.0 - self.nu**2)
        if self.E.dim() == 0:
            self.C = torch.zeros(2, 2, 2, 2)
        else:
            self.C = torch.zeros(*self.E.shape, 2, 2, 2, 2)
        self.C[..., 0, 0, 0, 0] = fac
        self.C[..., 0, 0, 1, 1] = fac * self.nu
        self.C[..., 1, 1, 0, 0] = fac * self.nu
        self.C[..., 1, 1, 1, 1] = fac
        self.C[..., 0, 1, 0, 1] = fac * 0.5 * (1.0 - self.nu)
        self.C[..., 0, 1, 1, 0] = fac * 0.5 * (1.0 - self.nu)
        self.C[..., 1, 0, 0, 1] = fac * 0.5 * (1.0 - self.nu)
        self.C[..., 1, 0, 1, 0] = fac * 0.5 * (1.0 - self.nu)

    def vectorize(self, n_elem: int) -> IsotropicElasticityPlaneStress:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicElasticityPlaneStress: A new material instance with vectorized
                properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicElasticityPlaneStress(E, nu, rho)


class IsotropicElasticityPlaneStrain(IsotropicElasticity3D):
    """Isotropic elastic material for planar strain problems.

    Args:
        E (Tensor): Young's modulus. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor): Poisson's ratio. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        rho (Tensor): Mass density. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.

    Notes:
        - Small-strain assumption.
        - Plane strain condition ($\\varepsilon_{33} = 0$).
        - No internal state variables (``n_state = 0``).
        - Supports batched/vectorized material parameters.

    Info: Definition of plane strain stiffness tensor
        This class represents a 2D isotropic linear elastic material for plane strain
        under small-strain assumptions, defined by (batched) Young's modulus $E$ and
        (batched) Poisson's ratio $\\nu$. The constructor derives the (batched)
        stiffness tensor $\\mathbb{C}$ by enforcing the plane strain condition
        $\\varepsilon_{33} = 0$ as
        $$
            C_{0000} = C_{1111} = 2 G + \\lambda
                = \\frac{E (1 - \\nu)}{(1 + \\nu)(1 - 2 \\nu)}
        $$
        $$
            C_{0011} = C_{1100} = \\lambda
                = \\frac{E \\nu}{(1 + \\nu)(1 - 2 \\nu)}
        $$
        $$
            C_{0101} = C_{0110} = C_{1001} = C_{1010} = G
                = \\frac{E}{2(1 + \\nu)}.
        $$
    """

    def __init__(
        self, E: Tensor | float, nu: Tensor | float, rho: Tensor | float = 1.0
    ):
        super().__init__(E, nu, rho)

        # Overwrite the 3D stiffness tensor with a 2D plane strain tensor
        lbd = self.lbd
        G = self.G
        if self.E.dim() == 0:
            self.C = torch.zeros(2, 2, 2, 2)
        else:
            self.C = torch.zeros(*self.E.shape, 2, 2, 2, 2)
        self.C[..., 0, 0, 0, 0] = 2.0 * G + lbd
        self.C[..., 0, 0, 1, 1] = lbd
        self.C[..., 1, 1, 0, 0] = lbd
        self.C[..., 1, 1, 1, 1] = 2.0 * G + lbd
        self.C[..., 0, 1, 0, 1] = G
        self.C[..., 0, 1, 1, 0] = G
        self.C[..., 1, 0, 0, 1] = G
        self.C[..., 1, 0, 1, 0] = G

    def vectorize(self, n_elem: int) -> IsotropicElasticityPlaneStrain:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicElasticityPlaneStrain: A new material instance with vectorized
                properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicElasticityPlaneStrain(E, nu, rho)


class IsotropicElasticity1D(Material):
    """Isotropic elastic material in 1D.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption.
        - No internal state variables (``n_state = 0``).
        - The stiffness "tensor" is simply $C_{0000} = E$.

    Info: 1D stiffness tensor
        The 1D stiffness "tensor" is simply $C_{0000} = E$.
    """

    def __init__(self, E: float | Tensor, rho: float | Tensor = 1.0):
        # Convert float inputs to tensors
        self.E = torch.as_tensor(E)
        self.rho = torch.as_tensor(rho)

        # Check if the material is vectorized
        self.is_vectorized = self.E.dim() > 0

        # There are no internal variables
        self.n_state = 0

        # Stiffness tensor (in 1D, this is a 1x1x1x1 "tensor")
        self.C = self.E[..., None, None, None, None]

    def vectorize(self, n_elem: int) -> IsotropicElasticity1D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicElasticity1D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicElasticity1D(E, rho)

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
        """Performs an incremental step for the 1D elastic material.

        The stress is updated as
        $$
            \\sigma_{n+1} = \\sigma_n + E \\, (\\Delta\\varepsilon
                - \\Delta\\varepsilon^0)
        $$
        with algorithmic tangent $\\frac{\\partial \\Delta\\sigma}
        {\\partial \\Delta H} = E$.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                *Shape:* `(..., 1, 1)`.
            F (Tensor): Current deformation gradient.
                *Shape:* `(..., 1, 1)`.
            stress (Tensor): Current stress tensor.
                *Shape:* `(..., 1, 1)`.
            state (Tensor): Internal state variables (unused).
                *Shape:* `(..., 0)`.
            de0 (Tensor): External strain increment (e.g., thermal).
                *Shape:* `(..., 1, 1)`.
            cl (Tensor): Characteristic lengths.
                *Shape:* `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            stress_new (Tensor): Updated stress tensor.
                *Shape:* `(..., 1, 1)`.
            state_new (Tensor): Updated internal state (unchanged).
                *Shape:* `(..., 0)`.
            ddsdde (Tensor): Algorithmic tangent stiffness tensor.
                *Shape:* `(..., 1, 1, 1, 1)`.
        """
        stress_new = stress + torch.einsum("...ijkl,...kl->...ij", self.C, H_inc - de0)
        state_new = state
        ddsdde = self.C
        return stress_new, state_new, ddsdde


class OrthotropicElasticity3D(Material):
    """Orthotropic elastic material in 3D.

    Args:
        E_1 (Tensor | float): Young's modulus in direction 1.
        E_2 (Tensor | float): Young's modulus in direction 2.
        E_3 (Tensor | float): Young's modulus in direction 3.
        nu_12 (Tensor | float): Poisson's ratio $\\nu_{12}$.
        nu_13 (Tensor | float): Poisson's ratio $\\nu_{13}$.
        nu_23 (Tensor | float): Poisson's ratio $\\nu_{23}$.
        G_12 (Tensor | float): Shear modulus $G_{12}$.
        G_13 (Tensor | float): Shear modulus $G_{13}$.
        G_23 (Tensor | float): Shear modulus $G_{23}$.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption.
        - No internal state variables (``n_state = 0``).
        - Supports batched/vectorized material parameters.
        - Supports rotation of the material coordinate system via `rotate()`.

    Info: Orthotropic stiffness tensor
        The orthotropic stiffness tensor in Voigt notation has the structure
        $$
            \\mathbb{C} =
            \\begin{bmatrix}
                C_{11} & C_{12} & C_{13} & 0 & 0 & 0 \\cr
                C_{12} & C_{22} & C_{23} & 0 & 0 & 0 \\cr
                C_{13} & C_{23} & C_{33} & 0 & 0 & 0 \\cr
                0 & 0 & 0 & G_{23} & 0 & 0 \\cr
                0 & 0 & 0 & 0 & G_{13} & 0 \\cr
                0 & 0 & 0 & 0 & 0 & G_{12}
            \\end{bmatrix}
        $$
        where the upper-left block contains the normal stiffness components
        and the lower-right block contains the shear moduli. The nine independent
        components are derived from nine independent engineering constants
        ($E_1, E_2, E_3, \\nu_{12}, \\nu_{13}, \\nu_{23},
        G_{12}, G_{13}, G_{23}$). The actual stiffness tensor $C_{ijkl}$ is stored as
        full 4th-order tensor with major and minor symmetries.
    """

    def __init__(
        self,
        E_1: float | Tensor,
        E_2: float | Tensor,
        E_3: float | Tensor,
        nu_12: float | Tensor,
        nu_13: float | Tensor,
        nu_23: float | Tensor,
        G_12: float | Tensor,
        G_13: float | Tensor,
        G_23: float | Tensor,
        rho: float | Tensor = 1.0,
    ):
        # Convert float inputs to tensors
        self.E_1 = torch.as_tensor(E_1)
        self.E_2 = torch.as_tensor(E_2)
        self.E_3 = torch.as_tensor(E_3)
        self.nu_12 = torch.as_tensor(nu_12)
        self.nu_21 = self.E_2 / self.E_1 * self.nu_12
        self.nu_13 = torch.as_tensor(nu_13)
        self.nu_31 = self.E_3 / self.E_1 * self.nu_13
        self.nu_23 = torch.as_tensor(nu_23)
        self.nu_32 = self.E_3 / self.E_2 * self.nu_23
        self.G_12 = torch.as_tensor(G_12)
        self.G_13 = torch.as_tensor(G_13)
        self.G_23 = torch.as_tensor(G_23)
        self.rho = torch.as_tensor(rho)

        # Check if the material is vectorized
        self.is_vectorized = self.E_1.dim() > 0

        # There are no internal variables
        self.n_state = 0

        # Full stiffness tensor
        if self.E_1.dim() == 0:
            self.C = torch.zeros(3, 3, 3, 3)
        else:
            self.C = torch.zeros(*self.E_1.shape, 3, 3, 3, 3)
        F = 1 / (
            1
            - self.nu_12 * self.nu_21
            - self.nu_13 * self.nu_31
            - self.nu_23 * self.nu_32
            - 2 * self.nu_21 * self.nu_32 * self.nu_13
        )
        self.C[..., 0, 0, 0, 0] = self.E_1 * (1 - self.nu_23 * self.nu_32) * F
        self.C[..., 1, 1, 1, 1] = self.E_2 * (1 - self.nu_13 * self.nu_31) * F
        self.C[..., 2, 2, 2, 2] = self.E_3 * (1 - self.nu_12 * self.nu_21) * F
        self.C[..., 0, 0, 1, 1] = self.E_1 * (self.nu_21 + self.nu_31 * self.nu_23) * F
        self.C[..., 1, 1, 0, 0] = self.C[..., 0, 0, 1, 1]
        self.C[..., 0, 0, 2, 2] = self.E_1 * (self.nu_31 + self.nu_21 * self.nu_32) * F
        self.C[..., 2, 2, 0, 0] = self.C[..., 0, 0, 2, 2]
        self.C[..., 1, 1, 2, 2] = self.E_2 * (self.nu_32 + self.nu_12 * self.nu_31) * F
        self.C[..., 2, 2, 1, 1] = self.C[..., 1, 1, 2, 2]
        self.C[..., 0, 1, 0, 1] = self.G_12
        self.C[..., 1, 0, 1, 0] = self.G_12
        self.C[..., 0, 1, 1, 0] = self.G_12
        self.C[..., 1, 0, 0, 1] = self.G_12
        self.C[..., 0, 2, 0, 2] = self.G_13
        self.C[..., 2, 0, 2, 0] = self.G_13
        self.C[..., 0, 2, 2, 0] = self.G_13
        self.C[..., 2, 0, 0, 2] = self.G_13
        self.C[..., 1, 2, 1, 2] = self.G_23
        self.C[..., 2, 1, 2, 1] = self.G_23
        self.C[..., 1, 2, 2, 1] = self.G_23
        self.C[..., 2, 1, 1, 2] = self.G_23

    def vectorize(self, n_elem: int) -> OrthotropicElasticity3D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            OrthotropicElasticity3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E_1 = self.E_1.repeat(n_elem)
            E_2 = self.E_2.repeat(n_elem)
            E_3 = self.E_3.repeat(n_elem)
            nu_12 = self.nu_12.repeat(n_elem)
            nu_13 = self.nu_13.repeat(n_elem)
            nu_23 = self.nu_23.repeat(n_elem)
            G_12 = self.G_12.repeat(n_elem)
            G_13 = self.G_13.repeat(n_elem)
            G_23 = self.G_23.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return OrthotropicElasticity3D(
                E_1, E_2, E_3, nu_12, nu_13, nu_23, G_12, G_13, G_23, rho
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
        """Performs an incremental step in the orthotropic elasticity model.

        The stress is updated as
        $$
            \\pmb{\\sigma}_{n+1} = \\pmb{\\sigma}_n + \\mathbb{C} :
                (\\Delta \\pmb{\\varepsilon} - \\Delta \\pmb{\\varepsilon}^0)
        $$

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                *Shape:* `(..., 3, 3)`.
            F (Tensor): Current deformation gradient.
                *Shape:* `(..., 3, 3)`.
            stress (Tensor): Current Cauchy stress tensor.
                *Shape:* `(..., 3, 3)`.
            state (Tensor): Internal state variables (unused).
                *Shape:* `(..., 0)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                *Shape:* `(..., 3, 3)`.
            cl (Tensor): Characteristic lengths.
                *Shape:* `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            stress_new (Tensor): Updated Cauchy stress tensor.
                *Shape:* `(..., 3, 3)`.
            state_new (Tensor): Updated internal state (unchanged).
                *Shape:* `(..., 0)`.
            ddsdde (Tensor): Algorithmic tangent stiffness tensor.
                *Shape:* `(..., 3, 3, 3, 3)`.
        """
        # Compute small strain tensor
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)
        # Compute new stress
        stress_new = stress + torch.einsum("...ijkl,...kl->...ij", self.C, de - de0)
        # Update internal state (this material does not change state)
        state_new = state
        # Algorithmic tangent
        ddsdde = self.C
        return stress_new, state_new, ddsdde

    def rotate(self, R: Tensor) -> OrthotropicElasticity3D:
        """Rotates the material coordinate system with a rotation matrix $\\mathbf{R}$.

        The rotated stiffness tensor is computed as
        $C'_{mnop} = R_{mi} R_{nj} R_{ok} R_{pl} \\, C_{ijkl}$
        and the engineering constants are re-extracted from the compliance matrix.

        Args:
            R (Tensor): Rotation matrix.
                *Shape:* `(..., 3, 3)`.

        Returns:
            OrthotropicElasticity3D: The material itself with rotated properties.
        """
        if R.shape[-2] != 3 or R.shape[-1] != 3:
            raise ValueError("Rotation matrix must be a 3x3 tensor.")

        # Compute rotated stiffness tensor
        self.C = torch.einsum(
            "...ijkl,...mi,...nj,...ok,...pl->...mnop", self.C, R, R, R, R
        )

        # Compute rotated internal variables
        S = torch.linalg.inv(stiffness2voigt(self.C))
        self.E_1 = 1 / S[..., 0, 0]
        self.E_2 = 1 / S[..., 1, 1]
        self.E_3 = 1 / S[..., 2, 2]
        self.nu_12 = -S[..., 0, 1] / S[..., 0, 0]
        self.nu_13 = -S[..., 0, 2] / S[..., 0, 0]
        self.nu_23 = -S[..., 1, 2] / S[..., 1, 1]
        self.G_23 = 1 / S[..., 3, 3]
        self.G_13 = 1 / S[..., 4, 4]
        self.G_12 = 1 / S[..., 5, 5]
        return self


class TransverseIsotropicElasticity3D(OrthotropicElasticity3D):
    """Transversely isotropic elastic material in 3D.

    A special case of `OrthotropicElasticity3D` where one direction (the
    longitudinal direction 1) is the axis of symmetry and the transverse
    plane (2-3) is isotropic.

    Args:
        E_L (Tensor | float): Longitudinal Young's modulus.
        E_T (Tensor | float): Transverse Young's modulus.
        nu_L (Tensor | float): Longitudinal Poisson's ratio
            $\\nu_L = \\nu_{12} = \\nu_{13}$.
        nu_T (Tensor | float): Transverse Poisson's ratio $\\nu_T = \\nu_{23}$.
        G_L (Tensor | float): Longitudinal shear modulus $G_L = G_{12} = G_{13}$.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Only five independent constants: $E_L, E_T, \\nu_L, \\nu_T, G_L$.
        - The transverse shear modulus is derived as
          $G_T = E_T / (2(1 + \\nu_T))$.
        - Raises `ValueError` if $G_L > E_L / (2(1 + \\nu_L))$.
    """

    def __init__(
        self,
        E_L: float | Tensor,
        E_T: float | Tensor,
        nu_L: float | Tensor,
        nu_T: float | Tensor,
        G_L: float | Tensor,
        rho: float | Tensor = 1.0,
    ):
        # https://webpages.tuni.fi/rakmek/jmm/slides/jmm_lect_06.pdf
        if G_L > E_L / (2 * (1 + nu_L)):
            raise ValueError("G must be less than E_L/(2*(1+nu_L))")

        E_1 = E_L
        E_2 = E_T
        E_3 = E_T
        nu_12 = nu_L
        nu_13 = nu_L
        nu_23 = nu_T
        G_12 = G_L
        G_13 = G_L
        G_23 = E_2 / (2 * (1 + nu_23))

        super().__init__(E_1, E_2, E_3, nu_12, nu_13, nu_23, G_12, G_13, G_23, rho)


class OrthotropicElasticityPlaneStress(OrthotropicElasticity3D):
    """Orthotropic elastic material for plane stress problems.

    Args:
        E_1 (Tensor | float): Young's modulus in direction 1.
        E_2 (Tensor | float): Young's modulus in direction 2.
        nu_12 (Tensor | float): Poisson's ratio $\\nu_{12}$.
        G_12 (Tensor | float): In-plane shear modulus $G_{12}$.
        G_13 (Tensor | float): Transverse shear modulus $G_{13}$. Default is `0.0`.
        G_23 (Tensor | float): Transverse shear modulus $G_{23}$. Default is `0.0`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption with plane stress condition.
        - No internal state variables (``n_state = 0``).
        - Supports rotation of the material coordinate system via `rotate()`.

    Info: Plane stress orthotropic stiffness
        The in-plane stiffness components are
        $$
            C_{1111} = \\frac{E_1}{1 - \\nu_{12} \\nu_{21}}, \\quad
            C_{2222} = \\frac{E_2}{1 - \\nu_{12} \\nu_{21}}, \\quad
            C_{1122} = \\frac{\\nu_{12} E_2}{1 - \\nu_{12} \\nu_{21}}
        $$
        with $\\nu_{21} = \\nu_{12} E_2 / E_1$ and
        $C_{1212} = G_{12}$.
    """

    def __init__(
        self,
        E_1: float | Tensor,
        E_2: float | Tensor,
        nu_12: float | Tensor,
        G_12: float | Tensor,
        G_13: float | Tensor = 0.0,
        G_23: float | Tensor = 0.0,
        rho: float | Tensor = 1.0,
    ):
        # Convert float inputs to tensors
        self.E_1 = torch.as_tensor(E_1)
        self.E_2 = torch.as_tensor(E_2)
        self.nu_12 = torch.as_tensor(nu_12)
        self.nu_21 = self.E_2 / self.E_1 * self.nu_12
        self.G_12 = torch.as_tensor(G_12)
        self.G_13 = torch.as_tensor(G_13)
        self.G_23 = torch.as_tensor(G_23)
        self.rho = torch.as_tensor(rho)

        # Check if the material is vectorized
        self.is_vectorized = self.E_1.dim() > 0

        # There are no internal variables
        self.n_state = 0

        # Stiffness tensor
        if self.E_1.dim() == 0:
            self.C = torch.zeros(2, 2, 2, 2)
        else:
            self.C = torch.zeros(*self.E_1.shape, 2, 2, 2, 2)
        nu2 = self.nu_12 * self.nu_21
        self.C[..., 0, 0, 0, 0] = E_1 / (1 - nu2)
        self.C[..., 0, 0, 1, 1] = nu_12 * E_2 / (1 - nu2)
        self.C[..., 1, 1, 0, 0] = nu_12 * E_2 / (1 - nu2)
        self.C[..., 1, 1, 1, 1] = E_2 / (1 - nu2)
        self.C[..., 0, 1, 0, 1] = G_12
        self.C[..., 0, 1, 1, 0] = G_12
        self.C[..., 1, 0, 0, 1] = G_12
        self.C[..., 1, 0, 1, 0] = G_12

    def vectorize(self, n_elem: int) -> OrthotropicElasticityPlaneStress:
        """Returns a vectorized copy of the material for `n_elem` elements.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            OrthotropicElasticityPlaneStress: A new material instance with vectorized
                properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E_1 = self.E_1.repeat(n_elem)
            E_2 = self.E_2.repeat(n_elem)
            nu_12 = self.nu_12.repeat(n_elem)
            G_12 = self.G_12.repeat(n_elem)
            G_13 = self.G_13.repeat(n_elem)
            G_23 = self.G_23.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return OrthotropicElasticityPlaneStress(
                E_1, E_2, nu_12, G_12, G_13, G_23, rho
            )

    def rotate(self, R: Tensor) -> OrthotropicElasticityPlaneStress:
        """Rotates the material coordinate system with a rotation matrix $\\mathbf{R}$.

        Args:
            R (Tensor): Rotation matrix.
                *Shape:* `(..., 2, 2)`.

        Returns:
            OrthotropicElasticityPlaneStress: The material itself with rotated
                properties.
        """
        if R.shape[-2] != 2 or R.shape[-1] != 2:
            raise ValueError("Rotation matrix must be a 2x2 tensor.")

        # Compute rotated stiffness tensor
        self.C = torch.einsum(
            "...ijkl,...mi,...nj,...ok,...pl->...mnop", self.C, R, R, R, R
        )

        # Compute rotated internal variables
        S = torch.linalg.inv(self.C)
        self.E_1 = 1 / S[..., 0, 0, 0, 0]
        self.E_2 = 1 / S[..., 1, 1, 1, 1]
        self.nu_12 = -S[..., 0, 0, 0, 0] * S[..., 0, 0, 1, 1]
        self.G_12 = 1 / S[..., 0, 1, 0, 1]
        return self


class OrthotropicElasticityPlaneStrain(OrthotropicElasticity3D):
    """Orthotropic elastic material for plane strain problems.

    Args:
        E_1 (Tensor | float): Young's modulus in direction 1.
        E_2 (Tensor | float): Young's modulus in direction 2.
        E_3 (Tensor | float): Young's modulus in direction 3.
        nu_12 (Tensor | float): Poisson's ratio $\\nu_{12}$.
        nu_13 (Tensor | float): Poisson's ratio $\\nu_{13}$.
        nu_23 (Tensor | float): Poisson's ratio $\\nu_{23}$.
        G_12 (Tensor | float): In-plane shear modulus $G_{12}$.
        G_13 (Tensor | float): Transverse shear modulus $G_{13}$. Default is `0.0`.
        G_23 (Tensor | float): Transverse shear modulus $G_{23}$. Default is `0.0`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption with plane strain condition ($\\varepsilon_{33} = 0$).
        - No internal state variables (``n_state = 0``).
        - Supports rotation of the material coordinate system via `rotate()`.

    Info: Plane strain orthotropic stiffness
        The in-plane stiffness tensor is derived from the full 3D orthotropic
        stiffness by enforcing $\\varepsilon_{33} = 0$. The in-plane
        components are populated using the same reciprocal relations as the
        3D case, with the full factor
        $F = (1 - \\nu_{12}\\nu_{21} - \\nu_{13}\\nu_{31}
        - \\nu_{23}\\nu_{32} - 2\\nu_{21}\\nu_{32}\\nu_{13})^{-1}$.
    """

    def __init__(
        self,
        E_1: float | Tensor,
        E_2: float | Tensor,
        E_3: float | Tensor,
        nu_12: float | Tensor,
        nu_13: float | Tensor,
        nu_23: float | Tensor,
        G_12: float | Tensor,
        G_13: float | Tensor = 0.0,
        G_23: float | Tensor = 0.0,
        rho: float | Tensor = 1.0,
    ):
        # Convert float inputs to tensors
        self.E_1 = torch.as_tensor(E_1)
        self.E_2 = torch.as_tensor(E_2)
        self.E_3 = torch.as_tensor(E_3)
        self.nu_12 = torch.as_tensor(nu_12)
        self.nu_21 = self.E_2 / self.E_1 * self.nu_12
        self.nu_13 = torch.as_tensor(nu_13)
        self.nu_31 = self.E_3 / self.E_1 * self.nu_13
        self.nu_23 = torch.as_tensor(nu_23)
        self.nu_32 = self.E_3 / self.E_2 * self.nu_23
        self.G_12 = torch.as_tensor(G_12)
        self.G_13 = torch.as_tensor(G_13)
        self.G_23 = torch.as_tensor(G_23)
        self.rho = torch.as_tensor(rho)

        # Check if the material is vectorized
        self.is_vectorized = self.E_1.dim() > 0

        # There are no internal variables
        self.n_state = 0

        # Full stiffness tensor
        if self.E_1.dim() == 0:
            self.C = torch.zeros(2, 2, 2, 2)
        else:
            self.C = torch.zeros(*self.E_1.shape, 2, 2, 2, 2)
        F = 1 / (
            1
            - self.nu_12 * self.nu_21
            - self.nu_13 * self.nu_31
            - self.nu_23 * self.nu_32
            - 2 * self.nu_21 * self.nu_32 * self.nu_13
        )
        self.C[..., 0, 0, 0, 0] = self.E_1 * (1 - self.nu_23 * self.nu_32) * F
        self.C[..., 1, 1, 1, 1] = self.E_2 * (1 - self.nu_13 * self.nu_31) * F
        self.C[..., 0, 0, 1, 1] = self.E_1 * (self.nu_21 + self.nu_31 * self.nu_23) * F
        self.C[..., 1, 1, 0, 0] = self.C[..., 0, 0, 1, 1]
        self.C[..., 0, 1, 0, 1] = self.G_12
        self.C[..., 1, 0, 1, 0] = self.G_12
        self.C[..., 0, 1, 1, 0] = self.G_12
        self.C[..., 1, 0, 0, 1] = self.G_12

    def vectorize(self, n_elem: int) -> OrthotropicElasticityPlaneStrain:
        """Returns a vectorized copy of the material for `n_elem` elements.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            OrthotropicElasticityPlaneStrain: A new material instance with vectorized
                properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E_1 = self.E_1.repeat(n_elem)
            E_2 = self.E_2.repeat(n_elem)
            E_3 = self.E_3.repeat(n_elem)
            nu_12 = self.nu_12.repeat(n_elem)
            nu_13 = self.nu_13.repeat(n_elem)
            nu_23 = self.nu_23.repeat(n_elem)
            G_12 = self.G_12.repeat(n_elem)
            G_13 = self.G_13.repeat(n_elem)
            G_23 = self.G_23.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return OrthotropicElasticityPlaneStrain(
                E_1, E_2, E_3, nu_12, nu_13, nu_23, G_12, G_13, G_23, rho
            )

    def rotate(self, R: Tensor) -> OrthotropicElasticityPlaneStrain:
        """Rotates the material coordinate system with a rotation matrix $\\mathbf{R}$.

        Args:
            R (Tensor): Rotation matrix.
                *Shape:* `(..., 2, 2)`.

        Returns:
            OrthotropicElasticityPlaneStrain: The material itself with rotated
                properties.
        """
        if R.shape[-2] != 2 or R.shape[-1] != 2:
            raise ValueError("Rotation matrix must be a 2x2 tensor.")

        # Compute rotated stiffness tensor
        self.C = torch.einsum(
            "...ijkl,...mi,...nj,...ok,...pl->...mnop", self.C, R, R, R, R
        )

        # Compute rotated internal variables
        S = torch.linalg.inv(self.C)
        self.E_1 = 1 / S[..., 0, 0]
        self.E_2 = 1 / S[..., 1, 1]
        self.nu_12 = -S[..., 0, 0] * S[..., 0, 1]
        self.G_12 = 1 / S[..., 2, 2]
        return self
