from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor

from .elasticity import IsotropicElasticity3D


class IsotropicDamage3D(IsotropicElasticity3D):
    """Isotropic damage material model in 3D.

    This class extends `IsotropicElasticity3D` to incorporate isotropic damage
    with a scalar damage variable $D \\in [0, 1]$.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor | float): Poisson's ratio.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        d (Callable): Damage evolution function $D(\\kappa, l_c)$.
        d_prime (Callable): Derivative of the damage evolution
            $D'(\\kappa, l_c)$.
        eq_strain (Literal["rankine", "mises"]): Type of equivalent strain
            measure used for damage driving.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption.
        - Two internal state variables (``n_state = 2``):
          $\\kappa$ (damage driving variable) and $D$ (damage variable).
        - Supports batched/vectorized material parameters.

    Info: Isotropic damage model
        The stress is degraded by a scalar damage variable $D$ as

        $$
            \\pmb{\\sigma} = (1 - D) \\, \\mathbb{C} : \\pmb{\\varepsilon}
        $$

        where $\\mathbb{C}$ is the undamaged elastic stiffness tensor.

        The damage is driven by an equivalent strain measure
        $\\tilde{\\varepsilon}$. For ``eq_strain="rankine"``, this is the
        largest principal strain. The history variable $\\kappa$ tracks the
        maximum equivalent strain ever reached:

        $$
            \\kappa_{n+1} = \\max(\\kappa_n,\\, \\tilde{\\varepsilon}_{n+1})
        $$

        and the damage evolves irreversibly as $D = d(\\kappa, l_c)$, where
        the characteristic length $l_c$ is used for fracture energy
        regularization.
    """

    def __init__(
        self,
        E: float | Tensor,
        nu: float | Tensor,
        d: Callable,
        d_prime: Callable,
        eq_strain: Literal["rankine", "mises"],
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, nu, rho)
        self.d = d
        self.d_prime = d_prime
        self.n_state = 2
        self.eq_strain: Literal["rankine", "mises"] = eq_strain

    def vectorize(self, n_elem: int) -> IsotropicDamage3D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicDamage3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicDamage3D(E, nu, self.d, self.d_prime, self.eq_strain, rho)

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
        """Performs a strain increment with the isotropic damage model.

        The stress is computed as
        $\\pmb{\\sigma} = (1 - D) \\, \\mathbb{C} : \\pmb{\\varepsilon}$
        and the algorithmic tangent stiffness is

        $$
            C^{\\text{alg}}_{ijkl} = (1 - D)  C_{ijkl}
                - D'(\\kappa, l_c) \\, \\sigma^{\\text{trial}}_{ij} \\,
                  n_k \\, n_l
        $$

        where $\\mathbf{n}$ is the direction of the damage-driving
        principal strain.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                - Shape: `(..., 3, 3)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                - Shape: `(..., 3, 3)`, same as `H_inc`.
            stress (Tensor): Current Cauchy stress tensor.
                - Shape: `(..., 3, 3)`.
            state (Tensor): Internal state variables, here: equivalent plastic strain.
                - Shape: `(..., 1)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                - Shape: `(..., 3, 3)`.
            cl (Tensor): Characteristic lengths.
                - Shape: `(..., 1)`.
            iter (int): Newton iteration.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated Cauchy stress tensor after plastic
                    update. Shape: `(..., 3, 3)`.
                - **state_new (Tensor)**: Updated internal state with updated plastic
                    strain. Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                    Shape: `(..., 3, 3, 3, 3)`.
        """
        # Compute total strain
        H_new = (F - torch.eye(H_inc.shape[-1])) + H_inc
        eps_new = 0.5 * (H_new.transpose(-1, -2) + H_new)

        # Extract state variables
        kappa = state[..., 0]
        D = state[..., 1]

        # Initialize solution variables
        stress_new = stress.clone()
        state_new = state.clone()

        # Calculate equivalent strain
        if self.eq_strain == "rankine":
            L, Q = torch.linalg.eigh(eps_new)
            # Find largest eigenvalue by magnitude
            idx = L.abs().argmax(dim=-1, keepdim=True)
            eps_eq = torch.take_along_dim(L, idx, dim=-1).squeeze(-1)
            n = torch.take_along_dim(Q, idx.unsqueeze(-2), dim=-1).squeeze(-1)
        else:
            raise NotImplementedError(
                f"Equivalent strain type '{self.eq_strain}' is not implemented."
            )

        # Update kappa and damage
        kappa_new = torch.maximum(kappa, eps_eq)
        D_new = self.d(kappa_new, cl)
        D_prime = self.d_prime(kappa_new, cl)

        # Update stress
        sigma_trial = torch.einsum("...ijkl,...kl->...ij", self.C, eps_new - de0)
        stress_new = (1 - D_new)[:, None, None] * sigma_trial

        # Update state variables
        state_new[..., 0] = kappa_new
        state_new[..., 1] = D_new

        # Update tangent stiffness
        ddsdde = (1.0 - D_new)[..., None, None, None, None] * self.C
        if iter > 0:
            active = D_new > D
            ddsdde[active] -= D_prime[active, None, None, None, None] * torch.einsum(
                "...ij,...k,...l->...ijkl", sigma_trial[active], n[active], n[active]
            )
        return stress_new, state_new, ddsdde
