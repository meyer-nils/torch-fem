from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Material(ABC):
    """Base class for all material models.

    Attributes:
        n_state (int): Number of internal state variables.
        is_vectorized (bool): Indicates if material parameters are batched.
        rho (Tensor): Mass density.
            *Shape:* `(..., 1)` or scalar.
    """

    def __init__(self):
        self.n_state: int = 0
        self.is_vectorized: bool = False
        self.rho: Tensor = torch.tensor(1.0)

    @abstractmethod
    def vectorize(self, n_elem: int) -> Material:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.
        """
        pass

    @abstractmethod
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
        """Performs an incremental step of the material model.

        This function has to update the stress, internal state, and algorithmic tangent
        stiffness.

        Args:
            H_inc (Tensor): Incremental displacement gradient $\\Delta \\mathbf{H}$.
                *Shape:* `(..., d, d)`.
            F (Tensor): Current deformation gradient $\\mathbf{F}_n$.
                *Shape:* `(..., d, d)`.
            stress (Tensor): Current stress tensor $\\pmb{\\sigma}_n$ or
                $\\mathbf{P}_n$.
                *Shape:* `(..., d, d)`.
            state (Tensor): Internal state variables $\\pmb{\\alpha}_n$.
                *Shape:* `(..., <number of state variables>)`.
            de0 (Tensor): External strain increment (e.g., thermal).
                *Shape:* `(..., d, d)`.
            cl (Tensor): Characteristic lengths for regularization.
                *Shape:* `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            stress_new (Tensor): Updated stress tensor $\\pmb{\\sigma}_{n+1}$
                or $\\mathbf{P}_{n+1}$.
                *Shape:* `(..., d, d)`.
            state_new (Tensor): Updated internal state $\\pmb{\\alpha}_{n+1}$.
                *Shape:* ``(..., n_state)``
            ddsdde (Tensor): Algorithmic tangent stiffness tensor
                $\\frac{\\partial \\Delta \\pmb{\\sigma}}{\\partial \\Delta
                \\mathbf{H}}$. Shape: `(..., d, d, d, d)`.
        """
        pass

    def rotate(self, R: Tensor) -> Material:
        """Rotation of the material properties.

        Args:
            R (Tensor): Rotation tensor.
                *Shape:* `(..., d, d)`.

        Returns:
            Material: The material itself, but with rotated properties, if applicable.
        """
        return self
