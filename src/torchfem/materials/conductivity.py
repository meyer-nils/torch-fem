from __future__ import annotations

import copy

import torch
from torch import Tensor

from .base import Material


class IsotropicConductivity3D(Material):
    """Isotropic heat conductivity material in 3D.

    This class represents a 3D isotropic heat conductivity material, defined by the
    thermal conductivity $\\kappa$. The constructor derives the conductivity tensor
    $\\pmb{\\kappa} = \\kappa \\mathbf{I}$.

    Args:
        kappa (Tensor | float): Thermal conductivity. If a float is provided, it is
            converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        rho (Tensor | float): Mass density. If a float is provided, it is converted.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.

    Notes:
        - No internal state variables (``n_state = 0``).
        - Supports batched/vectorized material parameters.
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
        """Performs an incremental step in the isotropic heat conduction model.

        This function updates the heat flux from the temperature gradient increment
        via Fourier's law. The arguments follow the generic `Material.step(...)`
        signature, so the mechanical names carry thermal meaning here.

        Args:
            H_inc (Tensor): Incremental temperature gradient increment.
                *Shape:* `(..., 1, 3)`, where `...` represents batch dimensions.
            F (Tensor): Current temperature gradient.
                *Shape:* `(..., 1, 3)`, same as `H_inc`.
            stress (Tensor): Current heat flux.
                *Shape:* `(..., 1, 3)`.
            state (Tensor): Internal state variables (unused in heat conductivity).
                *Shape:* Arbitrary, remains unchanged.
            de0 (Tensor): External temperature gradient increment.
                *Shape:* `(..., 1, 3)`.
            cl (Tensor): Characteristic lengths.
                *Shape:* `(...)`.
            iter (int): Current iteration number.

        Returns:
            heat_flux_new (Tensor): Updated heat flux.
                *Shape:* `(..., 1, 3)`.
            state_new (Tensor): Updated internal state (unchanged).
                *Shape:* same as `state`.
            ddheat_flux_ddtemp_grad (Tensor): Algorithmic tangent tensor.
                *Shape:* `(..., 3, 3)`.
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
    """Isotropic heat conductivity material in 2D.

    Uses the same constitutive law as the 3D class with the conductivity tensor
    reduced to the in-plane 2x2 block.

    The inherited `step` method operates on thermal tensors with shapes
    `(..., 1, 2)` and returns an algorithmic tangent of shape `(..., 2, 2)`.
    """

    def __init__(self, kappa: Tensor | float, rho: Tensor | float = 1.0):
        """Create a 2D isotropic conductivity material.

        Args:
            kappa (Tensor | float): In-plane thermal conductivity.
            rho (Tensor | float): Mass density.
        """
        super().__init__(kappa, rho)
        self.KAPPA = self.KAPPA[..., :2, :2]


class IsotropicConductivity1D(IsotropicConductivity2D):
    """Isotropic heat conductivity material in 1D.

    Uses the same constitutive law as the higher-dimensional classes with a
    scalar 1x1 conductivity tensor.

    The inherited `step` method operates on thermal tensors with shapes
    `(..., 1, 1)` and returns an algorithmic tangent of shape `(..., 1, 1)`.
    """

    def __init__(self, kappa: Tensor | float, rho: Tensor | float = 1.0):
        """Create a 1D isotropic conductivity material.

        Args:
            kappa (Tensor | float): Thermal conductivity.
            rho (Tensor | float): Mass density.
        """
        super().__init__(kappa, rho)
        self.KAPPA = self.KAPPA[..., :1, :1]


class OrthotropicConductivity3D(IsotropicConductivity3D):
    """Orthotropic heat conductivity material in 3D.

    The principal conductivities are aligned with the material axes and can be
    rotated into the global frame with `rotate`.
    """

    def __init__(
        self,
        kappa_1: Tensor | float,
        kappa_2: Tensor | float,
        kappa_3: Tensor | float,
        rho: Tensor | float = 1.0,
    ):
        """Create a 3D orthotropic conductivity material.

        Args:
            kappa_1 (Tensor | float): Conductivity along local axis 1.
            kappa_2 (Tensor | float): Conductivity along local axis 2.
            kappa_3 (Tensor | float): Conductivity along local axis 3.
            rho (Tensor | float): Mass density.
        """
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

    def rotate(self, R: Tensor) -> OrthotropicConductivity3D:
        """Returns a copy with its conductivity tensor rotated by `R`."""
        if R.shape[-2] != 3 or R.shape[-1] != 3:
            raise ValueError("Rotation matrix must be a 3x3 tensor.")

        # compute rotated conductivity tensor
        new = copy.copy(self)
        new.KAPPA = torch.einsum("...ik, ...jl, ...kl -> ...ij", R, R, self.KAPPA)
        return new


class OrthotropicConductivity2D(IsotropicConductivity2D):
    """Orthotropic heat conductivity material in 2D.

    The two principal in-plane conductivities are aligned with local axes and
    can be rotated into the global frame with `rotate`.
    """

    def __init__(
        self,
        kappa_1: Tensor | float,
        kappa_2: Tensor | float,
        rho: Tensor | float = 1.0,
    ):
        """Create a 2D orthotropic conductivity material.

        Args:
            kappa_1 (Tensor | float): Conductivity along local axis 1.
            kappa_2 (Tensor | float): Conductivity along local axis 2.
            rho (Tensor | float): Mass density.
        """
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

    def rotate(self, R: Tensor) -> OrthotropicConductivity2D:
        """Returns a copy with its conductivity tensor rotated by `R`."""
        if R.shape[-2] != 2 or R.shape[-1] != 2:
            raise ValueError("Rotation matrix must be a 2x2 tensor.")

        # compute rotated conductivity tensor
        new = copy.copy(self)
        new.KAPPA = torch.einsum("...ik, ...jl, ...kl -> ...ij", R, R, self.KAPPA)
        return new
