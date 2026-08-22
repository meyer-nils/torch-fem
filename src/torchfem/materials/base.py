from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Material(ABC):
    """Base class for all material models.

    `vectorize(...)` and `rotate(...)` never modify the material they are called on
    and return it unchanged where there is nothing to do. Their results share its
    tensors, so no material may be written into in place.

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

    def vectorize(self, n_elem: int) -> Material:
        """Returns the material batched over `n_elem` elements.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            Material: A material carrying one entry per element, or itself if it
                is vectorized already.
        """
        if self.is_vectorized:
            return self
        new = copy.copy(self)
        for key, value in list(vars(new).items()):
            if isinstance(value, Tensor):
                setattr(new, key, value.repeat(n_elem, *(value.dim() * [1])))
        new.is_vectorized = True
        return new

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
                *Shape:* `(..., n_state)`.
            ddsdde (Tensor): Algorithmic tangent stiffness tensor
                $\\frac{\\partial \\Delta \\pmb{\\sigma}}{\\partial \\Delta
                \\mathbf{H}}$.
                *Shape:* `(..., d, d, d, d)`.
        """
        pass

    def rotate(self, R: Tensor) -> Material:
        """Returns the material with its properties rotated by `R`.

        Args:
            R (Tensor): Rotation tensor.
                *Shape:* `(..., d, d)`.

        Returns:
            Material: A material with rotated properties, or itself if isotropic.
        """
        return self
