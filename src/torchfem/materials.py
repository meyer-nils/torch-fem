from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt
from typing import Callable, Literal

import torch
from torch import Tensor
from torch.func import jacrev, vmap

from .utils import (
    stiffness2voigt,
    strain2voigt,
    stress2voigt,
    voigt2stiffness,
    voigt2stress,
)


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


class Hyperelastic3D(Material):
    """Hyperelastic material in 3D.

    This class implements a hyperelastic material model for large deformations
    using automatic differentiation. The user provides a strain energy density
    function $\\psi(\\mathbf{F}, \\texttt{params})$ and the model computes
    stress and tangent stiffness automatically via `torch.func`.

    Args:
        psi (Callable): Strain energy density function
            $\\psi(\\mathbf{F}, \\texttt{params})$ that takes the deformation
            gradient and material parameters and returns a scalar energy density.
        params (list | Tensor): Material parameters passed to `psi`.
            *Shape:* `(p,)` for a scalar or `(N, p)` for a batch of materials.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Finite-strain (large deformation) framework.
        - No internal state variables (``n_state = 0``).
        - Stress and tangent are computed via automatic differentiation of $\\psi$.

    Info: Hyperelastic constitutive law
        A hyperelastic material is defined by a strain energy density function
        $\\psi(\\mathbf{F})$ of the deformation gradient $\\mathbf{F}$. The first
        Piola-Kirchhoff stress is obtained as
        $$
            \\mathbf{P} = \\frac{\\partial \\psi}{\\partial \\mathbf{F}}
        $$
        and the material tangent as
        $$
            C_{iJkL}
                = \\frac{\\partial^2 \\psi}{\\partial F_{iJ} \\partial F_{kL}}.
        $$
        Both derivatives are computed automatically using `torch.func.jacrev`.

        Common choices for $\\psi$ include the Neo-Hookean model
        $$
            \\psi(\\mathbf{F}) = \\frac{\\mu}{2}
                \\left( \\text{tr}(\\mathbb{C}) - 3 \\right)
                - \\mu \\ln J
                + \\frac{\\lambda}{2} (\\ln J)^2
        $$
        with $\\mathbb{C} = \\mathbf{F}^\\top \\mathbf{F}$,
        $J = \\det(\\mathbf{F})$, and Lam\u00e9 parameters $\\mu, \\lambda$,
        or the Saint Venant-Kirchhoff model
        $$
            \\psi(\\mathbf{F}) = \\frac{\\lambda}{2}
                \\left( \\text{tr}(\\mathbf{E}) \\right)^2
                + \\mu \\, \\text{tr}(\\mathbf{E}^2)
        $$
        with the Green-Lagrange strain $\\mathbf{E} = \\frac{1}{2}
        (\\mathbb{C} - \\mathbb{I})$.
    """

    def __init__(self, psi: Callable, params: list | Tensor, rho: Tensor | float = 1.0):
        # Store the strain energy density function
        self.psi = psi

        # Store material parameters
        self.params = torch.as_tensor(params)

        # There are no internal variables
        self.n_state = 0

        # Density
        self.rho = torch.as_tensor(rho)

        # Check if the material is vectorized
        self.is_vectorized = self.params.dim() > 1

    def vectorize(self, n_elem: int) -> Hyperelastic3D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            Hyperelastic3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            rho = self.rho.repeat(n_elem)
            params = self.params.repeat(n_elem, 1)
            return Hyperelastic3D(self.psi, params, rho)

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
        """Performs an incremental step for a hyperelastic material.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                - Shape: `(..., 3, 3)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                - Shape: `(..., 3, 3)`, same as `H_inc`.
            stress (Tensor): Current 1st PK stress tensor.
                - Shape: `(..., 3, 3)`.
            state (Tensor): Internal state variables (unused in hyperelasticity).
                - Shape: Arbitrary, remains unchanged.
            de0 (Tensor): External deformation gradient increment (e.g., thermal).
                - Shape: `(..., 3, 3)`.
            cl (Tensor): Characteristic lengths.
                - Shape: `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated (1st PK) stress tensor.
                Shape: `(..., 3, 3)`.
                - **state_new (Tensor)**: Updated internal state (unchanged).
                Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                Shape: `(..., 3, 3, 3, 3)`.
        """
        with torch.enable_grad():
            # Compute deformation gradient.
            F_new = (F + H_inc).requires_grad_(True)
            # Compute first Piola-Kirchhoff stress tensor.
            P_new = vmap(jacrev(self.psi))(F_new, self.params)
            # Compute algorithmic tangent stiffness.
            ddsdde = vmap(jacrev(jacrev(self.psi)))(F_new, self.params)

        # Hyperelastic materials have no internal state update.
        state_new = state
        return P_new, state_new, ddsdde


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


class IsotropicPlasticity3D(IsotropicElasticity3D):
    """Isotropic elastoplastic material model in 3D.

    This class extends `IsotropicElasticity3D` to incorporate isotropic
    plasticity with a von Mises yield criterion and associative flow rule.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor | float): Poisson's ratio.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        sigma_f (Callable): Yield stress function $\\sigma_f(q)$ of the
            equivalent plastic strain $q$.
        sigma_f_prime (Callable): Derivative $\\sigma_f'(q)$.
        tolerance (float): Convergence tolerance for the local Newton solver.
            Default is `1e-5`.
        max_iter (int): Maximum number of Newton iterations. Default is `10`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption.
        - One internal state variable (``n_state = 1``): equivalent plastic
          strain $q$.
        - Supports batched/vectorized material parameters.
    """

    def __init__(
        self,
        E: float | Tensor,
        nu: float | Tensor,
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, nu, rho)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 1
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int) -> IsotropicPlasticity3D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicPlasticity3D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicPlasticity3D(
                E,
                nu,
                self.sigma_f,
                self.sigma_f_prime,
                self.tolerance,
                self.max_iter,
                rho,
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
        """Performs a strain increment with the radial return-mapping algorithm.

        In each increment, a trial stress is computed as

        $$
            \\pmb{\\sigma}_{\\text{trial}} = \\pmb{\\sigma}_n
                + \\mathbb{C} : \\Delta\\pmb{\\varepsilon}
        $$

        and the yield condition is checked via the flow potential

        $$
            f = \\|\\pmb{\\sigma}'_{\\text{trial}}\\|
                - \\sqrt{\\tfrac{2}{3}} \\, \\sigma_f(q)
        $$

        where $\\pmb{\\sigma}'$ denotes the deviatoric stress.

        **Elastic step** ($f \\le 0$): The trial stress is accepted.

        **Plastic step** ($f > 0$): The flow direction is
        $\\mathbf{n} = \\pmb{\\sigma}'_{\\text{trial}}
        / \\|\\pmb{\\sigma}'_{\\text{trial}}\\|$
        and the updates are

        $$
            \\pmb{\\sigma}_{n+1} = \\pmb{\\sigma}_{\\text{trial}}
                - 2 G \\, \\Delta\\gamma \\, \\mathbf{n}, \\quad
            q_{n+1} = q_n + \\sqrt{\\tfrac{2}{3}} \\, \\Delta\\gamma.
        $$

        The consistent (algorithmic) tangent is

        $$
            \\mathbb{C}^{\\text{alg}} = \\mathbb{C}
                - \\frac{2 G}{1 + \\frac{\\sigma_f'}{3 G}}
                  \\, \\mathbf{n} \\otimes \\mathbf{n}
                - \\frac{4 G^2 \\Delta\\gamma}
                  {\\|\\pmb{\\sigma}'_{\\text{trial}}\\|}
                  \\left( \\mathbb{I}^{\\text{dev}}
                  - \\mathbf{n} \\otimes \\mathbf{n} \\right)
        $$

        with $\\mathbb{I}^{\\text{dev}}_{ijkl} = \\frac{1}{2}(
        \\delta_{ik}\\delta_{jl} + \\delta_{il}\\delta_{jk})
        - \\frac{1}{3}\\delta_{ij}\\delta_{kl}$.

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
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated Cauchy stress tensor after plastic
                    update. Shape: `(..., 3, 3)`.
                - **state_new (Tensor)**: Updated internal state with updated plastic
                    strain. Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                    Shape: `(..., 3, 3, 3, 3)`.
        """
        # Second order identity tensor
        I2 = torch.eye(H_inc.shape[-1])
        # Compute small strain tensor
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)

        # Initialize solution variables
        stress_new = stress.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ddsdde = self.C.clone()

        # Compute trial stress
        s_trial = stress + torch.einsum("...ijkl,...kl->...ij", self.C, de - de0)

        # Compute the deviatoric trial stress
        s_trial_trace = s_trial[..., 0, 0] + s_trial[..., 1, 1] + s_trial[..., 2, 2]
        dev = s_trial.clone()
        dev[..., 0, 0] -= s_trial_trace / 3
        dev[..., 1, 1] -= s_trial_trace / 3
        dev[..., 2, 2] -= s_trial_trace / 3
        dev_norm = torch.linalg.norm(dev, dim=(-1, -2))

        # Flow potential
        f = dev_norm - sqrt(2.0 / 3.0) * self.sigma_f(q)
        fm = f > 0

        # Direction of flow
        n = dev[fm] / dev_norm[fm][..., None, None]

        # Local Newton solver to find plastic strain increment
        dGamma = torch.zeros_like(f[fm])
        G = self.G[fm]
        for _ in range(self.max_iter):
            res = (
                dev_norm[fm] - 2.0 * G * dGamma - sqrt(2.0 / 3.0) * self.sigma_f(q[fm])
            )
            ddGamma = res / (2.0 * G + 2.0 / 3.0 * self.sigma_f_prime(q[fm]))
            dGamma += ddGamma
            q[fm] += sqrt(2.0 / 3.0) * ddGamma

            # Check convergence for early stopping
            if (torch.abs(res) < self.tolerance).all():
                break

        # Check if the local Newton iteration converged
        if (torch.abs(res) > self.tolerance).any():
            print("Local Newton iteration did not converge")

        # Update stress
        stress_new[~fm] = s_trial[~fm]
        stress_new[fm] = s_trial[fm] - (2.0 * G * dGamma)[:, None, None] * n

        # Update state
        state_new[..., 0] = q

        # Update algorithmic tangent
        A = 2.0 * G / (1.0 + self.sigma_f_prime(q[fm]) / (3.0 * G))
        B = 4.0 * G**2 * dGamma / dev_norm[fm]
        I2 = torch.eye(3)
        I4 = torch.einsum("ij,kl->ijkl", I2, I2)
        I4S = torch.einsum("ik,jl->ijkl", I2, I2) + torch.einsum("il,jk->ijkl", I2, I2)
        nn = torch.einsum("...ij,...kl->...ijkl", n, n)
        ddsdde[fm] = (
            self.C[fm]
            - A[..., None, None, None, None] * nn
            - B[..., None, None, None, None] * (1 / 2 * I4S - 1 / 3 * I4 - nn)
        )

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


class HyperelasticPlaneStress(Hyperelastic3D):
    """Hyperelastic plane stress material.

    This class extends `Hyperelastic3D` for plane stress problems by enforcing
    the out-of-plane stress condition $P_{33} = 0$ through a local Newton
    iteration on the out-of-plane stretch $\\lambda_3$.

    Args:
        psi (Callable): Strain energy density function
            $\\psi(\\mathbf{F}, \\texttt{params})$. This must be a 3D energy;
            the plane stress condensation is performed internally.
        params (list | Tensor): Material parameters passed to `psi`.
            *Shape:* `(p,)` for a scalar or `(N, p)` for a batch of materials.
        rho (Tensor | float): Mass density. Default is `1.0`.
        tolerance (float): Convergence tolerance for the local Newton solver.
            Default is `1e-5`.
        max_iter (int): Maximum number of iterations for the local Newton solver.
            Default is `10`.

    Notes:
        - Finite-strain (large deformation) framework.
        - One internal state variable (``n_state = 1``): the out-of-plane
          stretch increment $\\lambda_3 - 1$.
        - The 3D deformation gradient is reconstructed as
          $\\mathbf{F} = \\text{diag}(\\mathbf{F}^{2D},\\, \\lambda_3)$.

    Info: Plane stress condensation
        Given a 3D strain energy $\\psi(\\mathbf{F})$, the plane stress
        condition requires $P_{33} = 0$. The out-of-plane stretch $\\lambda_3$
        is found iteratively by solving
        $$
            P_{33} (\\mathbf{F}^{\\text{2D}}, \\lambda_3) = 0
        $$
        with a Newton update
        $$
            \\lambda_3 \\leftarrow \\lambda_3
                - \\frac{P_{33}}{C_{3333}}.
        $$
        The in-plane tangent is obtained via static condensation:
        $$
            C_{ijkl}^{\\text{2D}}
                = C_{ijkl}
                - \\frac{C_{ij33}\\,C_{33kl}}{C_{3333}}.
        $$
    """

    def __init__(
        self,
        psi: Callable,
        params: list | Tensor,
        rho: float | Tensor = 1.0,
        tolerance: float = 1e-5,
        max_iter: int = 10,
    ):
        super().__init__(psi, params, rho)
        self.tolerance = tolerance
        self.max_iter = max_iter

        # State variable for out-of-plane stretch
        self.n_state = 1

    def vectorize(self, n_elem: int) -> HyperelasticPlaneStress:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            HyperelasticPlaneStress: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            rho = self.rho.repeat(n_elem)
            params = self.params.repeat(n_elem, 1)
            return HyperelasticPlaneStress(
                self.psi, params, rho, self.tolerance, self.max_iter
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
        """Performs an incremental step for a hyperelastic material.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                - Shape: `(..., 2, 2)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                - Shape: `(..., 2, 2)`, same as `H_inc`.
            stress (Tensor): Current 1st PK stress tensor.
                - Shape: `(..., 2, 2)`.
            state (Tensor): Internal state variables (unused in hyperelasticity).
                - Shape: Arbitrary, remains unchanged.
            de0 (Tensor): External deformation gradient increment (e.g., thermal).
                - Shape: `(..., 2, 2)`.
            cl (Tensor): Characteristic lengths.
                - Shape: `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated 1st PK stress tensor.
                Shape: `(..., 2, 2)`.
                - **state_new (Tensor)**: Updated internal state (unchanged).
                Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                Shape: `(..., 2, 2, 2, 2)`.
        """

        # Extract out-of plane stretch
        lbd_z = 1.0 + state[..., 0]

        F3D = torch.zeros(F.shape[0], 3, 3)
        F3D[..., 0:2, 0:2] = F
        F3D[..., 2, 2] = lbd_z

        # Local Newton solver to find out-of-plane stretch with plane stress condition
        for _ in range(self.max_iter):
            # Update deformation gradient increment
            F3D_new = torch.zeros(F.shape[0], 3, 3)
            F3D_new[..., 0:2, 0:2] = F + H_inc
            F3D_new[..., 2, 2] = lbd_z
            H_inc_3D = F3D_new - F3D
            # Call parent class step function
            P, _, ddsdde = super().step(H_inc_3D, F3D, stress, state, de0, cl, iter)
            # Evaluate plane stress condition
            res = P[..., 2, 2]
            # Update out-of plane stretch
            lbd_z -= res / ddsdde[..., 2, 2, 2, 2]
            # Check convergence
            if (torch.abs(res) < self.tolerance).all():
                break
        if (torch.abs(res) > self.tolerance).any():
            print("Local Newton iteration did not converge.")

        # Update stress
        P_new = P[..., 0:2, 0:2]

        # Update internal state
        state_new = lbd_z.unsqueeze(-1) - 1.0

        # Update algorithmic tangent
        tangent_correction = torch.einsum(
            "...ij,...kl,...->...ijkl",
            ddsdde[..., 0:2, 0:2, 2, 2],
            ddsdde[..., 2, 2, 0:2, 0:2],
            1 / ddsdde[..., 2, 2, 2, 2],
        )
        ddsdde_new = ddsdde[..., 0:2, 0:2, 0:2, 0:2] - tangent_correction

        # Update stress
        return P_new, state_new, ddsdde_new


class IsotropicPlasticityPlaneStress(IsotropicElasticityPlaneStress):
    """Isotropic elastoplastic material for plane stress problems.

    This class extends `IsotropicElasticityPlaneStress` to incorporate
    isotropic plasticity with a von Mises yield criterion under the plane
    stress constraint.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor | float): Poisson's ratio.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        sigma_f (Callable): Yield stress function $\\sigma_f(q)$.
        sigma_f_prime (Callable): Derivative $\\sigma_f'(q)$.
        tolerance (float): Convergence tolerance. Default is `1e-5`.
        max_iter (int): Maximum Newton iterations. Default is `10`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption with plane stress condition.
        - One internal state variable (``n_state = 1``): equivalent plastic
          strain $q$.
        - Implementation follows de Souza Neto et al. , Box 9.3-9.6.

    References:
        de Souza Neto, E. A., Peri, D., Owen, D. R. J.
        *Computational Methods for Plasticity*, Chapter 9,
        https://doi.org/10.1002/9780470694626.ch9, 2008.
    """

    def __init__(
        self,
        E: float | Tensor,
        nu: float | Tensor,
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, nu, rho)
        self._C = stiffness2voigt(self.C)
        self._S = torch.linalg.inv(self._C)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 1
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int) -> IsotropicPlasticityPlaneStress:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicPlasticityPlaneStress: A new material instance with vectorized
                properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicPlasticityPlaneStress(
                E,
                nu,
                self.sigma_f,
                self.sigma_f_prime,
                self.tolerance,
                self.max_iter,
                rho,
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
        """Performs a strain increment with the plane stress return-mapping algorithm.

        A trial stress is computed as
        $\\pmb{\\sigma}_{\\text{trial}} = \\pmb{\\sigma}_n
        + \\mathbb{C} : \\Delta\\pmb{\\varepsilon}$
        and the squared flow potential in Voigt notation is evaluated as

        $$
            \\Psi = \\tfrac{1}{2} \\pmb{\\sigma}_{\\text{trial}}^\\top
                    \\mathbf{P} \\, \\pmb{\\sigma}_{\\text{trial}}
                    - \\tfrac{1}{3} \\sigma_f(q)^2
        $$

        with

        $$
            \\mathbf{P} = \\frac{1}{3}
                \\begin{bmatrix} 2 & -1 & 0 \\\\ -1 & 2 & 0 \\\\ 0 & 0 & 6
                \\end{bmatrix}.
        $$

        If $\\Psi > 0$, the stress is updated via

        $$
            \\pmb{\\sigma}_{n+1}
                = [\\mathbb{S} + \\Delta\\gamma \\, \\mathbf{P}]^{-1}
                  \\, \\mathbb{S} \\, \\pmb{\\sigma}_{\\text{trial}}
        $$

        and the algorithmic tangent is

        $$
            \\mathbb{C}^{\\text{alg}}
                = [\\mathbb{S} + \\Delta\\gamma \\, \\mathbf{P}]^{-1}
                - \\alpha \\, \\mathbf{n} \\otimes \\mathbf{n}.
        $$

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                Shape: `(..., 2, 2)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                Shape: `(..., 2, 2)`, same as `H_inc`.
            stress (Tensor): Current Cauchy stress tensor.
                Shape: `(..., 2, 2)`.
            state (Tensor): Internal state variables, here: equivalent plastic strain.
                Shape: `(..., 1)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                Shape: `(..., 2, 2)`.
            cl (Tensor): Characteristic lengths.
                Shape: `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated Cauchy stress tensor after plastic
                    update. Shape: `(..., 2, 2)`.
                - **state_new (Tensor)**: Updated internal state with updated plastic
                    strain. Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                    Shape: `(..., 2, 2, 2, 2)`.
        """

        # Projection operator
        P = 1 / 3 * torch.tensor([[2, -1, 0], [-1, 2, 0], [0, 0, 6]])

        # Compute small strain tensor in Voigt notation
        depsilon = strain2voigt(0.5 * (H_inc.transpose(-1, -2) + H_inc) - de0)
        # Convert stress to Voigt notation
        stress_voigt = stress2voigt(stress)

        # Solution variables
        stress_new = stress_voigt.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ddsdde = self._C.clone()

        # Compute trial stress
        s_trial = stress_voigt + torch.einsum("...kl,...l->...k", self._C, depsilon)

        # Flow potential
        a1 = (s_trial[..., 0] + s_trial[..., 1]) ** 2
        a2 = (s_trial[..., 1] - s_trial[..., 0]) ** 2
        a3 = s_trial[..., 2] ** 2
        xi_trial = 1 / 6 * a1 + 1 / 2 * a2 + 2 * a3
        psi = 1 / 2 * xi_trial - 1 / 3 * self.sigma_f(q) ** 2

        # Flow mask
        fm = psi > 0

        # Local Newton solver to find plastic strain increment
        dGamma = torch.zeros_like(psi[fm])
        E = self.E[fm]
        G = self.G[fm]
        nu = self.nu[fm]
        for j in range(self.max_iter):
            # Compute xi and some short hands
            xi = (
                a1[fm] / (6 * (1 + E * dGamma / (3 * (1 - nu))) ** 2)
                + (1 / 2 * a2[fm] + 2 * a3[fm]) / (1 + 2 * G * dGamma) ** 2
            )
            sxi = torch.sqrt(xi)
            qq = q[fm] + dGamma * torch.sqrt(2 / 3 * xi)

            # Compute residual
            res = 1 / 2 * xi - 1 / 3 * self.sigma_f(qq) ** 2

            # Compute derivative of residual w.r.t dGamma
            H = self.sigma_f_prime(qq)
            xi_p = (
                -a1[fm] / (9 * (1 + E * dGamma / (3 * (1 - nu))) ** 3) * E / (1 - nu)
                - 2 * G * (a2[fm] + 4 * a3[fm]) / (1 + 2 * G * dGamma) ** 3
            )
            H_p = (
                2
                * self.sigma_f(qq)
                * H
                * sqrt(2 / 3)
                * (sxi + dGamma * xi_p / (2 * sxi))
            )
            res_prime = 1 / 2 * xi_p - 1 / 3 * H_p

            # Update dGamma
            dGamma -= res / res_prime

            if (torch.abs(res) < self.tolerance).all():
                break
        if (torch.abs(res) > self.tolerance).any():
            print("Local Newton iteration did not converge.")

        # Compute inverse operator
        inv = torch.linalg.inv(self._S[fm] + dGamma[None, :, None, None] * P)

        # Update stress
        stress_new[~fm] = s_trial[~fm]
        stress_new[fm] = (inv @ self._S[fm] @ s_trial[fm][:, :, None]).squeeze(-1)

        # Update state
        q[fm] = qq
        state_new[..., 0] = q

        # Update algorithmic tangent
        xi = (
            stress_new[fm][:, :, None].transpose(-1, -2)
            @ P
            @ stress_new[fm][:, :, None]
        )
        H = self.sigma_f_prime(q[fm])
        n = inv @ P @ stress_new[fm][:, :, None]
        alpha = 1.0 / (
            stress_new[fm][:, :, None].transpose(-1, -2) @ P @ n
            + 2 * xi * H / (3 - 2 * H * dGamma[:, None, None])
        )
        ddsdde[fm] = inv - alpha * n @ n.transpose(-1, -2)

        return voigt2stress(stress_new), state_new, voigt2stiffness(ddsdde)


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


class HyperelasticPlaneStrain(Hyperelastic3D):
    """Hyperelastic plane strain material.

    This class extends `Hyperelastic3D` for plane strain problems by enforcing
    $F_{33} = 1$ (no out-of-plane deformation). The 3D energy function is
    evaluated with an extended deformation gradient.

    Args:
        psi (Callable): Strain energy density function
            $\\psi(\\mathbf{F}, \\texttt{params})$. This must be a 3D energy;
            the plane strain constraint is enforced internally.
        params (list | Tensor): Material parameters passed to `psi`.
            *Shape:* `(p,)` for a scalar or `(N, p)` for a batch of materials.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Finite-strain (large deformation) framework.
        - No internal state variables (``n_state = 0``).
        - The 3D deformation gradient is reconstructed as
          $\\mathbf{F} = \\text{diag}(\\mathbf{F}^\\text{2D},\\, 1)$.

    Info: Plane strain constraint
        The plane strain condition fixes the out-of-plane component to
        $F_{33} = 1$ and $H_{33} = 0$. The 3D deformation gradient is
        extended as
        $$
            \\mathbf{F}
              = \\begin{bmatrix}
                F_{11} & F_{12} & 0 \\cr
                F_{21} & F_{22} & 0 \\cr
                0 & 0 & 1
            \\end{bmatrix}
        $$
        and the in-plane stress and tangent are extracted from the 3D
        response.
    """

    def vectorize(self, n_elem: int) -> HyperelasticPlaneStrain:
        """Returns a vectorized copy of the material for `n_elem` elements.

        This function creates a batched version of the material properties. If the
        material is already vectorized (`self.is_vectorized == True`), the function
        simply returns `self` without modification.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            HyperelasticPlaneStrain: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            rho = self.rho.repeat(n_elem)
            params = self.params.repeat(n_elem, 1)
            return HyperelasticPlaneStrain(self.psi, params, rho)

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
        """Performs an incremental step for a hyperelastic material.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                - Shape: `(..., 2, 2)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                - Shape: `(..., 2, 2)`, same as `H_inc`.
            stress (Tensor): Current 1st PK stress tensor.
                - Shape: `(..., 2, 2)`.
            state (Tensor): Internal state variables (unused in hyperelasticity).
                - Shape: Arbitrary, remains unchanged.
            de0 (Tensor): External deformation gradient increment (e.g., thermal).
                - Shape: `(..., 2, 2)`.
            cl (Tensor): Characteristic lengths.
                - Shape: `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated 1st PK stress tensor.
                Shape: `(..., 2, 2)`.
                - **state_new (Tensor)**: Updated internal state (unchanged).
                Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                Shape: `(..., 2, 2, 2, 2)`.
        """
        # Extend deformation gradient to 3D
        F3D = torch.zeros(F.shape[0], 3, 3)
        F3D[..., 0:2, 0:2] = F
        F3D[..., 2, 2] = 1.0
        H_inc_3D = torch.zeros(H_inc.shape[0], 3, 3)
        H_inc_3D[..., 0:2, 0:2] = H_inc
        # Call parent class step function
        P_new, state_new, ddsdde_new = super().step(
            H_inc_3D, F3D, stress, state, de0, cl, iter
        )
        # Return only in-plane components
        return P_new[..., 0:2, 0:2], state_new, ddsdde_new[..., 0:2, 0:2, 0:2, 0:2]


class IsotropicPlasticityPlaneStrain(IsotropicElasticityPlaneStrain):
    """Isotropic elastoplastic material for plane strain problems.

    This class extends `IsotropicElasticityPlaneStrain` to incorporate
    isotropic plasticity with a von Mises yield criterion under the plane
    strain constraint.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        nu (Tensor | float): Poisson's ratio.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        sigma_f (Callable): Yield stress function $\\sigma_f(q)$.
        sigma_f_prime (Callable): Derivative $\\sigma_f'(q)$.
        tolerance (float): Convergence tolerance. Default is `1e-5`.
        max_iter (int): Maximum Newton iterations. Default is `10`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - Small-strain assumption with plane strain condition.
        - Two internal state variables (``n_state = 2``): equivalent plastic
          strain $q$ and out-of-plane plastic strain $\\varepsilon_z^p$.
        - Supports batched/vectorized material parameters.
    """

    def __init__(
        self,
        E: float | Tensor,
        nu: float | Tensor,
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, nu, rho)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 2
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int) -> IsotropicPlasticityPlaneStrain:
        """Returns a vectorized copy of the material for `n_elem` elements.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicPlasticityPlaneStrain: A new material instance with vectorized
                properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            nu = self.nu.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicPlasticityPlaneStrain(
                E,
                nu,
                self.sigma_f,
                self.sigma_f_prime,
                self.tolerance,
                self.max_iter,
                rho,
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
        """Performs a strain increment with the plane strain return-mapping algorithm.

        The return mapping expands the in-plane stress to a full 3D tensor
        with the out-of-plane stress
        $$
            \\sigma_{zz} = \\nu (\\sigma_{xx} + \\sigma_{yy})
                - E \\, \\varepsilon_z^p
        $$
        and performs the deviatoric yield check and radial return in 3D.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                Shape: `(..., 2, 2)`, where `...` represents batch dimensions.
            F (Tensor): Current deformation gradient.
                Shape: `(..., 2, 2)`, same as `H_inc`.
            stress (Tensor): Current Cauchy stress tensor.
                Shape: `(..., 2, 2)`.
            state (Tensor): Internal state variables, here: equivalent plastic strain
                and stress in the third direction. Shape: `(..., 2)`.
            de0 (Tensor): External small strain increment (e.g., thermal).
                Shape: `(..., 2, 2)`.
            cl (Tensor): Characteristic lengths.
                Shape: `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            tuple:
                - **stress_new (Tensor)**: Updated Cauchy stress tensor after plastic
                    update. Shape: `(..., 2, 2)`.
                - **state_new (Tensor)**: Updated internal state with updated plastic
                    strain. Shape: same as `state`.
                - **ddsdde (Tensor)**: Algorithmic tangent stiffness tensor.
                    Shape: `(..., 2, 2, 2, 2)`.
        """
        # Compute small strain tensor in Voigt notation
        de = 0.5 * (H_inc.transpose(-1, -2) + H_inc)

        # Solution variables
        stress_new = stress.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ez = state_new[..., 1]
        ddsdde = self.C.clone()

        # Compute trial stress
        s_2D = stress + torch.einsum("...ijkl,...kl->...ij", self.C, de - de0)
        s_trial = torch.zeros(stress.shape[0], 3, 3)
        s_trial[..., :2, :2] = s_2D
        s_trial[..., 2, 2] = self.nu * (s_2D[..., 0, 0] + s_2D[..., 1, 1]) - self.E * ez

        # Compute the deviatoric trial stress
        s_trial_trace = s_trial[..., 0, 0] + s_trial[..., 1, 1] + s_trial[..., 2, 2]
        dev = s_trial.clone()
        dev[..., 0, 0] -= s_trial_trace / 3
        dev[..., 1, 1] -= s_trial_trace / 3
        dev[..., 2, 2] -= s_trial_trace / 3
        dev_norm = torch.linalg.norm(dev, dim=(-1, -2))

        # Flow potential
        f = dev_norm - sqrt(2.0 / 3.0) * self.sigma_f(q)
        fm = f > 0

        # Direction of flow
        n = dev[fm] / dev_norm[fm][..., None, None]

        # Local Newton solver to find plastic strain increment
        dGamma = torch.zeros_like(f[fm])
        G = self.G[fm]
        for _ in range(self.max_iter):
            res = (
                dev_norm[fm] - 2.0 * G * dGamma - sqrt(2.0 / 3.0) * self.sigma_f(q[fm])
            )
            ddGamma = res / (2.0 * G + 2.0 / 3.0 * self.sigma_f_prime(q[fm]))
            dGamma += ddGamma
            q[fm] += sqrt(2.0 / 3.0) * ddGamma

            # Check convergence for early stopping
            if (torch.abs(res) < self.tolerance).all():
                break

        # Check if the local Newton iteration converged
        if (torch.abs(res) > self.tolerance).any():
            print("Local Newton iteration did not converge")

        # Update stress
        stress_new[~fm] = s_trial[~fm][..., :2, :2]
        stress_new[fm] = (s_trial[fm] - (2.0 * G * dGamma)[:, None, None] * n)[
            ..., :2, :2
        ]

        # Update state
        state_new[..., 0] = q
        ez[fm] += dGamma * n[..., 2, 2]
        state_new[..., 1] = ez

        # Update algorithmic tangent
        A = 2.0 * G / (1.0 + self.sigma_f_prime(q[fm]) / (3.0 * G))
        B = 4.0 * G**2 * dGamma / dev_norm[fm]
        n0n1 = n[:, 0, 0] * n[:, 1, 1]
        ddsdde[fm, 0, 0, 0, 0] += -A * n[:, 0, 0] ** 2 - B * (2 / 3 - n[:, 0, 0] ** 2)
        ddsdde[fm, 1, 1, 1, 1] += -A * n[:, 1, 1] ** 2 - B * (2 / 3 - n[:, 1, 1] ** 2)
        ddsdde[fm, 0, 0, 1, 1] += -A * n0n1 - B * (-1 / 3 - n0n1)
        ddsdde[fm, 1, 1, 0, 0] += -A * n0n1 - B * (-1 / 3 - n0n1)
        ddsdde[fm, 0, 1, 0, 1] += -A * n[:, 0, 1] ** 2 - B * (1 / 2 - n[:, 0, 1] ** 2)
        ddsdde[fm, 0, 1, 1, 0] += -A * n[:, 0, 1] ** 2 - B * (1 / 2 - n[:, 1, 0] ** 2)
        ddsdde[fm, 1, 0, 0, 1] += -A * n[:, 1, 0] ** 2 - B * (1 / 2 - n[:, 0, 1] ** 2)
        ddsdde[fm, 1, 0, 1, 0] += -A * n[:, 1, 0] ** 2 - B * (1 / 2 - n[:, 1, 0] ** 2)

        return stress_new, state_new, ddsdde


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


class IsotropicPlasticity1D(IsotropicElasticity1D):
    """Isotropic elastoplastic material in 1D.

    This class extends `IsotropicElasticity1D` with isotropic hardening
    plasticity using a return-mapping algorithm.

    Args:
        E (Tensor | float): Young's modulus.
            *Shape:* `()` for a scalar or `(N,)` for a batch of materials.
        sigma_f (Callable): Yield stress function $\\sigma_f(q)$.
        sigma_f_prime (Callable): Derivative $\\sigma_f'(q)$.
        tolerance (float): Convergence tolerance. Default is `1e-5`.
        max_iter (int): Maximum Newton iterations. Default is `10`.
        rho (Tensor | float): Mass density. Default is `1.0`.

    Notes:
        - One internal state variable (``n_state = 1``): equivalent plastic
          strain $q$.
    """

    def __init__(
        self,
        E: float | Tensor,
        sigma_f: Callable,
        sigma_f_prime: Callable,
        tolerance: float = 1e-5,
        max_iter: int = 10,
        rho: float | Tensor = 1.0,
    ):
        super().__init__(E, rho)
        self.sigma_f = sigma_f
        self.sigma_f_prime = sigma_f_prime
        self.n_state = 1
        self.tolerance = tolerance
        self.max_iter = max_iter

    def vectorize(self, n_elem: int) -> IsotropicPlasticity1D:
        """Returns a vectorized copy of the material for `n_elem` elements.

        Args:
            n_elem (int): Number of elements to vectorize the material for.

        Returns:
            IsotropicPlasticity1D: A new material instance with vectorized properties.
        """
        if self.is_vectorized:
            print("Material is already vectorized.")
            return self
        else:
            E = self.E.repeat(n_elem)
            rho = self.rho.repeat(n_elem)
            return IsotropicPlasticity1D(
                E, self.sigma_f, self.sigma_f_prime, self.tolerance, self.max_iter, rho
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
        """Performs a strain increment with the 1D return-mapping algorithm.

        A trial stress is computed as
        $\\sigma_{\\text{trial}} = \\sigma_n + E \\, \\Delta\\varepsilon$.
        The yield condition is $|\\sigma_{\\text{trial}}| \\le \\sigma_f(q)$.

        If yielding occurs, the plastic multiplier $\\Delta\\gamma$ is found
        by Newton's method on
        $$
            |\\sigma_{\\text{trial}}| - E \\, \\Delta\\gamma
                - \\sigma_f(q) = 0
        $$
        and the stress is updated as
        $$
            \\sigma_{n+1} = \\left(1
                - \\frac{E \\, \\Delta\\gamma}
                  {|\\sigma_{\\text{trial}}|}\\right)
                \\sigma_{\\text{trial}}, \\quad
            q_{n+1} = q_n + \\Delta\\gamma.
        $$
        The algorithmic tangent in the plastic regime is
        $\\frac{E \\, \\sigma_f'}{E + \\sigma_f'}$.

        Args:
            H_inc (Tensor): Incremental displacement gradient.
                *Shape:* `(..., 1, 1)`.
            F (Tensor): Current deformation gradient.
                *Shape:* `(..., 1, 1)`.
            stress (Tensor): Current stress tensor.
                *Shape:* `(..., 1, 1)`.
            state (Tensor): Internal state variables (equivalent plastic strain $q$).
                *Shape:* `(..., 1)`.
            de0 (Tensor): External strain increment (e.g., thermal).
                *Shape:* `(..., 1, 1)`.
            cl (Tensor): Characteristic lengths.
                *Shape:* `(..., 1)`.
            iter (int): Current iteration number.

        Returns:
            stress_new (Tensor): Updated stress tensor.
                *Shape:* `(..., 1, 1)`.
            state_new (Tensor): Updated internal state with updated plastic strain.
                *Shape:* `(..., 1)`.
            ddsdde (Tensor): Algorithmic tangent stiffness tensor.
                *Shape:* `(..., 1, 1, 1, 1)`.
        """
        # Solution variables
        stress_new = stress.clone()
        state_new = state.clone()
        q = state_new[..., 0]
        ddsdde = self.C.clone()

        # Compute trial stress
        s_trial = stress + torch.einsum("...ijkl,...kl->...ij", self.C, H_inc - de0)
        s_norm = torch.abs(s_trial).squeeze()

        # Flow potential
        f = s_norm - self.sigma_f(q)
        fm = f > 0

        # Local Newton solver to find plastic strain increment
        dGamma = torch.zeros_like(f[fm])
        E = self.E[fm]
        for _ in range(self.max_iter):
            res = s_norm[fm] - E * dGamma - self.sigma_f(q[fm])
            ddGamma = res / (E + self.sigma_f_prime(q[fm]))
            dGamma += ddGamma
            q[fm] += ddGamma

            # Check convergence for early stopping
            if (torch.abs(res) < self.tolerance).all():
                break

        # Check if the local Newton iteration converged
        if (torch.abs(res) > self.tolerance).any():
            print("Local Newton iteration did not converge.")

        # Update stress
        stress_new[~fm] = s_trial[~fm]
        stress_new[fm] = (1.0 - (dGamma * E) / s_norm[fm])[:, None, None] * s_trial[fm]

        # Update state
        state_new[..., 0] = q

        # Update algorithmic tangent
        if fm.sum() > 0:
            ddsdde[fm] = (
                E[:, None, None, None, None]
                * self.sigma_f_prime(q[fm])
                / (E[:, None, None, None, None] + self.sigma_f_prime(q[fm]))
            )

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
