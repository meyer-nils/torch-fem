from __future__ import annotations

import torch
from torch import Tensor

from .base import Material


class IsotropicConductivity3D(Material):
    """Isotropic heat conductivity material.

    This class represents a 3D isotropic heat conductivity material, defined by the
    thermal conductivity k.

    Attributes:
        k (Tensor | float): Thermal conductivity. Converted, if a float is provided.
            Shape: `()` for a scalar or `(N,)` for a batch of materials.
    """

    def __init__(self, kappa: Tensor | float, rho: Tensor | float = 1.0):
        # Convert float inputs to tensors
        self.kappa = torch.as_tensor(kappa)
        self.rho = torch.as_tensor(rho)

        # There are no internal variables
        self.n_state = 0

        # Check if the material is vectorized
        self.is_vectorized = self.kappa.dim() > 0

        # Identity tensors
        I2 = torch.eye(3)

        # Stiffness tensor
        self.KAPPA = self.kappa[..., None, None] * I2

    def vectorize(self, n_elem: int) -> IsotropicConductivity3D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicConductivity3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            kappa = self.kappa.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicConductivity3D(kappa, rho)

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

        This function updates the deformation gradient, stress, and internal state
        variables based on a small-strain assumption.

        Args:
            H_inc (Tensor): Incremental temperature gradient increment.
                - Shape: `(..., 3, 1)`, where `...` represents batch dimensions.
            F (Tensor): Current temperature gradient.
                - Shape: `(..., 3, 1)`, same as `H_inc`.
            stress (Tensor): Current heat flux.
                - Shape: `(..., 3, 1)`.
            state (Tensor): Internal state variables (unused in heat conductivity).
                - Shape: Arbitrary, remains unchanged.
            de0 (Tensor): External temperature gradient increment.
                - Shape: `(..., 3, 1)`.
            cl (Tensor): Characteristic lengths.
                - Shape: `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **heat_flux_new (Tensor)**: Updated heat flux.
                Shape: `(..., 3, 1)`.
                - **state_new (Tensor)**: Updated internal state (unchanged).
                Shape: same as `state`.
                - **ddheat_flux_ddtemp_grad (Tensor)**: Algorithmic tangent tensor.
                Shape: `(..., 3, 3)`.
        """
        # Interpretation of inputs
        temp_grad_inc = H_inc
        heat_flux = stress

        # Compute new heat flux
        heat_flux_new = heat_flux + torch.einsum(
            "...ij,...kj->...ki", self.KAPPA, temp_grad_inc - de0
        )
        # Update internal state (this material does not change state)
        state_new = state
        # Algorithmic tangent
        ddheat_flux_ddtemp_grad = self.KAPPA
        return heat_flux_new, state_new, ddheat_flux_ddtemp_grad


class IsotropicConductivity2D(IsotropicConductivity3D):
    def __init__(self, kappa: Tensor | float, rho: Tensor | float = 1.0):
        super().__init__(kappa, rho)
        self.KAPPA = self.KAPPA[..., :2, :2]

    def vectorize(self, n_elem: int) -> IsotropicConductivity2D:
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            return IsotropicConductivity2D(
                self.kappa.repeat(n_elem), self.rho.repeat(n_elem)
            )


class IsotropicConductivity1D(IsotropicConductivity2D):
    def __init__(self, kappa: Tensor | float, rho: Tensor | float = 1.0):
        super().__init__(kappa, rho)
        self.KAPPA = self.KAPPA[..., :1, :1]

    def vectorize(self, n_elem: int) -> IsotropicConductivity1D:
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            return IsotropicConductivity1D(
                self.kappa.repeat(n_elem),
                self.rho.repeat(n_elem),
            )


class OrthotropicConductivity3D(IsotropicConductivity3D):
    def __init__(
        self,
        kappa_1: Tensor | float,
        kappa_2: Tensor | float,
        kappa_3: Tensor | float,
        rho: Tensor | float = 1.0,
    ):
        self.kappa_1 = torch.as_tensor(kappa_1)
        self.kappa_2 = torch.as_tensor(kappa_2)
        self.kappa_3 = torch.as_tensor(kappa_3)
        self.rho = torch.as_tensor(rho)

        # There are no internal variables
        self.n_state = 0

        e1, e2, e3 = torch.eye(3)
        P1 = torch.outer(e1, e1)
        P2 = torch.outer(e2, e2)
        P3 = torch.outer(e3, e3)

        self.KAPPA = (
            self.kappa_1[..., None, None] * P1
            + self.kappa_2[..., None, None] * P2
            + self.kappa_3[..., None, None] * P3
        )

        self.is_vectorized = self.kappa_1.dim() > 0

    def vectorize(self, n_elem: int) -> OrthotropicConductivity3D:
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            return OrthotropicConductivity3D(
                self.kappa_1.repeat(n_elem),
                self.kappa_2.repeat(n_elem),
                self.kappa_3.repeat(n_elem),
                self.rho.repeat(n_elem),
            )

    def rotate(self, R: Tensor) -> OrthotropicConductivity3D:
        """Rotate the material with rotation matrix R."""
        if R.shape[-2] != 3 or R.shape[-1] != 3:
            raise ValueError("Rotation matrix must be a 3x3 tensor.")

        # compute rotated conductivity tensor
        self.KAPPA = torch.einsum("...ik, ...jl, ...kl -> ...ij", R, R, self.KAPPA)
        return self


class OrthotropicConductivity2D(IsotropicConductivity2D):
    def __init__(
        self,
        kappa_1: Tensor | float,
        kappa_2: Tensor | float,
        rho: Tensor | float = 1.0,
    ):
        self.kappa_1 = torch.as_tensor(kappa_1)
        self.kappa_2 = torch.as_tensor(kappa_2)
        self.rho = torch.as_tensor(rho)

        # There are no internal variables
        self.n_state = 0

        e1, e2 = torch.eye(2)
        P1 = torch.outer(e1, e1)
        P2 = torch.outer(e2, e2)

        self.KAPPA = (
            self.kappa_1[..., None, None] * P1 + self.kappa_2[..., None, None] * P2
        )

        self.is_vectorized = self.kappa_1.dim() > 0

    def vectorize(self, n_elem: int) -> OrthotropicConductivity2D:
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            return OrthotropicConductivity2D(
                self.kappa_1.repeat(n_elem),
                self.kappa_2.repeat(n_elem),
                self.rho.repeat(n_elem),
            )

    def rotate(self, R: Tensor) -> OrthotropicConductivity2D:
        """Rotate the material with rotation matrix R."""
        if R.shape[-2] != 2 or R.shape[-1] != 2:
            raise ValueError("Rotation matrix must be a 2x2 tensor.")

        # compute rotated conductivity tensor
        self.KAPPA = torch.einsum("...ik, ...jl, ...kl -> ...ij", R, R, self.KAPPA)
        return self
