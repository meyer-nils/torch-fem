from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torch.func import jacrev, vmap

from .base import Material


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
