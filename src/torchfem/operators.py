"""
Physics operators for PDE-agnostic FEM framework.

This module provides abstract operator interface and concrete
implementations for different PDE types (mechanics, diffusion, etc.).
"""

from abc import ABC, abstractmethod
from typing import Callable
import torch
from torch import Tensor

from .materials import Material


class Operator(ABC):
    """
    Abstract base class for physics operators.

    Replaces Material interface with generic nomenclature:
    - gradient: generalized gradient (strain, grad(T), etc.)
    - flux: generalized flux (stress, heat flux, etc.)
    - state: internal state variables
    - tangent: flux-gradient tangent operator
    """

    @abstractmethod
    def __init__(self):
        self.n_state: int
        self.is_vectorized: bool
        pass

    @property
    @abstractmethod
    def n_dof_per_node(self) -> int:
        """
        Number of degrees of freedom per node.

        Returns:
            1 for scalar fields (temperature, concentration)
            n_dim for vector fields (displacement)
        """
        pass

    @abstractmethod
    def vectorize(self, n_elem: int) -> 'Operator':
        """Create vectorized operator for n_elem elements."""
        pass

    @abstractmethod
    def evaluate(
        self,
        gradient_inc: Tensor,
        deformation: Tensor,
        flux: Tensor,
        state: Tensor,
        external_inc: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate operator response to gradient increment.

        Args:
            gradient_inc: Incremental gradient (strain, grad(dT))
            deformation: Cumulative deformation measure (F, identity)
            flux: Current flux (stress, heat flux)
            state: Internal state variables
            external_inc: External gradient increment (thermal strain)

        Returns:
            (flux_new, state_new, tangent)
        """
        pass

    @abstractmethod
    def compute_element_gradient(
        self,
        field_elem: Tensor,
        B: Tensor,
    ) -> Tensor:
        """
        Compute physics-specific gradient from element field.

        Args:
            field_elem: Element field values
                Shape: (n_elem, n_nodes_per_elem, n_dof_per_node)
            B: Shape function gradient matrix
                Shape: (n_elem, n_dim, n_nodes_per_elem)

        Returns:
            Gradient tensor with physics-specific shape:
                Vector fields: (n_elem, n_dim, n_dim)
                Scalar fields: (n_elem, n_dim)
        """
        pass

    @abstractmethod
    def assemble_element_stiffness(
        self,
        B: Tensor,
        tangent: Tensor,
        detJ: Tensor,
        weight: float,
    ) -> Tensor:
        """
        Assemble element stiffness matrix.

        Args:
            B: Shape function gradient matrix
                Shape: (n_elem, n_dim, n_nodes_per_elem)
            tangent: Tangent operator from evaluate()
            detJ: Jacobian determinant
                Shape: (n_elem,) or scalar
            weight: Quadrature weight

        Returns:
            Element stiffness matrix
                Shape: (n_elem, n_nodes, n_dofs, n_nodes, n_dofs)
        """
        pass


class MechanicsOperator(Operator):
    """
    Adapter wrapping Material models as Operators.

    Maps mechanics nomenclature to generic operator interface:
    - gradient = strain
    - flux = stress
    - deformation = deformation gradient F
    """

    def __init__(self, material: Material):
        """
        Wrap existing Material model as Operator.

        Args:
            material: torch-fem Material instance
        """
        self.material = material
        self.n_state = material.n_state
        self.is_vectorized = material.is_vectorized
        self._n_dim = None

    @property
    def n_dof_per_node(self) -> int:
        """Vector field: n_dim DOFs per node."""
        if self._n_dim is None:
            # Infer from material stiffness tensor
            self._n_dim = self.material.C.shape[-1]
        return self._n_dim

    def vectorize(self, n_elem: int) -> 'MechanicsOperator':
        if self.is_vectorized:
            return self
        return MechanicsOperator(self.material.vectorize(n_elem))

    def evaluate(
        self,
        gradient_inc: Tensor,
        deformation: Tensor,
        flux: Tensor,
        state: Tensor,
        external_inc: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate stress from strain increment via wrapped Material.

        Args:
            gradient_inc: Displacement gradient increment H_inc
            deformation: Deformation gradient F
            flux: Cauchy stress sigma
            state: Material state (plastic strain, etc.)
            external_inc: External strain (thermal, etc.)

        Returns:
            (stress_new, state_new, ddsdde)
        """
        return self.material.step(
            gradient_inc, deformation, flux, state, external_inc
        )

    def compute_element_gradient(
        self,
        field_elem: Tensor,
        B: Tensor,
    ) -> Tensor:
        """
        Compute displacement gradient H = ∂u/∂X.

        Args:
            field_elem: Element displacements
                Shape: (n_elem, n_nodes_per_elem, n_dim)
            B: Shape function gradient matrix
                Shape: (n_elem, n_dim, n_nodes_per_elem)

        Returns:
            Displacement gradient H
                Shape: (n_elem, n_dim, n_dim)
        """
        # H_ij = ∂u_i/∂X_j = B_jk * u_ik
        return torch.einsum("...jk,...kj->...ij", B, field_elem)

    def assemble_element_stiffness(
        self,
        B: Tensor,
        tangent: Tensor,
        detJ: Tensor,
        weight: float,
    ) -> Tensor:
        """
        Assemble element stiffness: K = ∫ B^T C B dV.

        Args:
            B: Shape function gradients (n_elem, n_dim, n_nodes_per_elem)
            tangent: Material tangent C (n_elem, n_dim, n_dim, n_dim, n_dim)
            detJ: Jacobian determinant (n_elem,) or scalar
            weight: Quadrature weight

        Returns:
            Element stiffness (n_elem, n_nodes_per_elem, n_dim,
                              n_nodes_per_elem, n_dim)
        """
        # BCB_lijkp = C_ijpq * B_qk * B_il
        BCB = torch.einsum(
            "...ijpq,...qk,...il->...likp", tangent, B, B
        )

        # Weight by detJ and quadrature weight
        if isinstance(detJ, Tensor):
            K_elem = BCB * (detJ[..., None, None, None, None] * weight)
        else:
            K_elem = BCB * (detJ * weight)

        return K_elem

    @property
    def C(self) -> Tensor:
        """Expose material stiffness tensor for mechanics."""
        return self.material.C


class DiffusionOperator(Operator):
    """
    Operator for diffusion/Laplace problems.

    Implements Fourier/Fick's law: q = -k * grad(phi)
    where phi is temperature, concentration, potential, etc.
    """

    def __init__(
        self,
        conductivity: Tensor | float,
        n_dim: int = 3
    ):
        """
        Isotropic diffusion operator.

        Args:
            conductivity: Scalar conductivity k (thermal, diffusive)
            n_dim: Spatial dimension (2 or 3)
        """
        self.k = torch.as_tensor(conductivity)
        self.n_dim = n_dim
        self.n_state = 0  # No history for linear diffusion
        self.is_vectorized = self.k.dim() > 0

        # Construct conductivity tensor K = k * I
        I_nd = torch.eye(n_dim)
        if self.is_vectorized:
            self.K = self.k[:, None, None] * I_nd
        else:
            self.K = self.k * I_nd

    @property
    def n_dof_per_node(self) -> int:
        """Scalar field: 1 DOF per node."""
        return 1

    def vectorize(self, n_elem: int) -> 'DiffusionOperator':
        if self.is_vectorized:
            return self
        k_vec = self.k.repeat(n_elem)
        return DiffusionOperator(k_vec, self.n_dim)

    def evaluate(
        self,
        gradient_inc: Tensor,
        deformation: Tensor,
        flux: Tensor,
        state: Tensor,
        external_inc: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate heat/diffusion flux from gradient.

        For linear steady-state diffusion:
        q_new = -K @ grad(phi)

        Args:
            gradient_inc: grad(phi) at current state
                Shape: (..., n_dim, n_dim) but only diagonal used
            deformation: Unused (identity for diffusion)
            flux: Previous flux (unused for steady-state)
            state: Unused (no history)
            external_inc: Source term (unused here)

        Returns:
            (flux_new, state_new, tangent)
        """
        # Extract gradient vector from gradient_inc
        # For diffusion, gradient is grad(phi) = [dphi/dx, dphi/dy, ...]
        # stored in diagonal of gradient_inc in mechanics formulation
        grad_phi = torch.diagonal(gradient_inc, dim1=-2, dim2=-1)

        # Physical flux: q_physical = -k * grad(phi)
        # But for FEM weak form ∫ ∇ψ · k∇φ dΩ, we need positive sign
        # for internal force F_int = ∫ ∇N^T · k∇φ dV to match K·φ
        flux_new = torch.einsum('...ij,...j->...i', self.K, grad_phi)

        # Convert back to tensor form matching mechanics convention
        flux_tensor = torch.diag_embed(flux_new)

        # No state evolution for linear diffusion
        state_new = state

        # Tangent: K for FEM weak form ∫ ∇ψ · K ∇φ dΩ
        # Must be 4th-order for FEM assembly: tangent_ijkl
        # For isotropic diffusion: tangent_ijkl = k * delta_ik * delta_jl
        # This gives B^T @ tangent @ B = k * (B^T @ B) which is correct
        I2 = torch.eye(self.n_dim)
        I4 = torch.einsum('ik,jl->ijkl', I2, I2)

        if self.is_vectorized:
            # Shape: (n_elem, n_dim, n_dim, n_dim, n_dim)
            tangent = self.k[:, None, None, None, None] * I4
        else:
            # Shape: (n_dim, n_dim, n_dim, n_dim)
            tangent = self.k * I4

        return flux_tensor, state_new, tangent

    def compute_element_gradient(
        self,
        field_elem: Tensor,
        B: Tensor,
    ) -> Tensor:
        """
        Compute scalar field gradient ∇φ.

        Args:
            field_elem: Element scalar field values
                Shape: (n_elem, n_nodes_per_elem, 1)
            B: Shape function gradient matrix
                Shape: (n_elem, n_dim, n_nodes_per_elem)

        Returns:
            Gradient vector ∇φ
                Shape: (n_elem, n_dim)
        """
        # Remove singleton dimension: (n_elem, n_nodes_per_elem, 1)
        # -> (n_elem, n_nodes_per_elem)
        phi = field_elem.squeeze(-1)

        # grad_phi_i = B_ik * phi_k
        return torch.einsum("...ik,...k->...i", B, phi)

    def assemble_element_stiffness(
        self,
        B: Tensor,
        tangent: Tensor,
        detJ: Tensor,
        weight: float,
    ) -> Tensor:
        """
        Assemble diffusion stiffness: K = ∫ ∇N^T k ∇N dV.

        Args:
            B: Shape function gradients (n_elem, n_dim, n_nodes_per_elem)
            tangent: Conductivity k (scalar or n_elem tensor)
            detJ: Jacobian determinant (n_elem,) or scalar
            weight: Quadrature weight

        Returns:
            Element stiffness (n_elem, n_nodes_per_elem, n_nodes_per_elem)
        """
        # K_ij = ∫ (∂N_i/∂x_k) k (∂N_j/∂x_k) dV
        # K_ij = B_ki * k * B_kj
        K_elem = torch.einsum("...ki,...kj->...ij", B, B)

        # Weight by conductivity, detJ, and quadrature weight
        if self.is_vectorized:
            k_weight = self.k
        else:
            k_weight = self.k

        if isinstance(detJ, Tensor):
            K_elem = K_elem * (k_weight[..., None, None] *
                               detJ[..., None, None] * weight)
        else:
            K_elem = K_elem * (k_weight[..., None, None] *
                               detJ * weight)

        return K_elem

    @property
    def C(self) -> Tensor:
        """
        Expose conductivity as 4th-order tensor matching
        mechanics interface.
        """
        # C_ijkl such that q_i = -C_ijkl * dphi/dx_j (for i=j, k=l)
        I2 = torch.eye(self.n_dim)
        if self.is_vectorized:
            C = self.k[:, None, None, None, None] * \
                torch.einsum('ij,kl->ijkl', I2, I2)
        else:
            C = self.k * torch.einsum('ij,kl->ijkl', I2, I2)
        return C


class AnisotropicDiffusionOperator(Operator):
    """
    Anisotropic diffusion operator with full conductivity tensor.

    q = -K @ grad(phi) where K is n_dim x n_dim
    """

    def __init__(self, conductivity_tensor: Tensor):
        """
        Args:
            conductivity_tensor: Full K matrix, shape
                (n_dim, n_dim) or (n_elem, n_dim, n_dim)
        """
        self.K = conductivity_tensor
        self.n_dim = conductivity_tensor.shape[-1]
        self.n_state = 0
        self.is_vectorized = self.K.dim() > 2

    @property
    def n_dof_per_node(self) -> int:
        """Scalar field: 1 DOF per node."""
        return 1

    def vectorize(self, n_elem: int) -> 'AnisotropicDiffusionOperator':
        if self.is_vectorized:
            return self
        K_vec = self.K.unsqueeze(0).repeat(n_elem, 1, 1)
        return AnisotropicDiffusionOperator(K_vec)

    def evaluate(
        self,
        gradient_inc: Tensor,
        deformation: Tensor,
        flux: Tensor,
        state: Tensor,
        external_inc: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Anisotropic diffusion flux evaluation."""
        grad_phi = torch.diagonal(gradient_inc, dim1=-2, dim2=-1)
        flux_new = -torch.einsum('...ij,...j->...i', self.K, grad_phi)
        flux_tensor = torch.diag_embed(flux_new)
        state_new = state

        # Tangent tensor
        I2 = torch.eye(self.n_dim)
        tangent = -torch.einsum('...ij,kl->...ikjl', self.K, I2)

        return flux_tensor, state_new, tangent

    def compute_element_gradient(
        self,
        field_elem: Tensor,
        B: Tensor,
    ) -> Tensor:
        """
        Compute scalar field gradient ∇φ.

        Args:
            field_elem: Element scalar field values
                Shape: (n_elem, n_nodes_per_elem, 1)
            B: Shape function gradient matrix
                Shape: (n_elem, n_dim, n_nodes_per_elem)

        Returns:
            Gradient vector ∇φ
                Shape: (n_elem, n_dim)
        """
        phi = field_elem.squeeze(-1)
        return torch.einsum("...ik,...k->...i", B, phi)

    def assemble_element_stiffness(
        self,
        B: Tensor,
        tangent: Tensor,
        detJ: Tensor,
        weight: float,
    ) -> Tensor:
        """
        Assemble anisotropic diffusion stiffness: K = ∫ ∇N^T K ∇N dV.

        Args:
            B: Shape function gradients (n_elem, n_dim, n_nodes_per_elem)
            tangent: Conductivity tensor K (n_elem, n_dim, n_dim)
            detJ: Jacobian determinant (n_elem,) or scalar
            weight: Quadrature weight

        Returns:
            Element stiffness (n_elem, n_nodes_per_elem, n_nodes_per_elem)
        """
        # K_ij = ∫ (∂N_i/∂x_k) K_kl (∂N_j/∂x_l) dV
        # KB = K_kl * B_lj
        KB = torch.einsum("...kl,...lj->...kj", self.K, B)
        # K_elem = B_ki * KB_kj
        K_elem = torch.einsum("...ki,...kj->...ij", B, KB)

        # Weight by detJ and quadrature weight
        if isinstance(detJ, Tensor):
            K_elem = K_elem * (detJ[..., None, None] * weight)
        else:
            K_elem = K_elem * (detJ * weight)

        return K_elem

    @property
    def C(self) -> Tensor:
        """4th-order conductivity tensor."""
        I2 = torch.eye(self.n_dim)
        C = torch.einsum('...ij,kl->...ikjl', self.K, I2)
        return C


class NonlinearDiffusionOperator(Operator):
    """
    Nonlinear diffusion with state-dependent conductivity.

    k = k(phi) specified by user function.
    """

    def __init__(
        self,
        conductivity_fn: Callable[[Tensor], Tensor],
        conductivity_derivative_fn: Callable[[Tensor], Tensor],
        n_dim: int = 3
    ):
        """
        Args:
            conductivity_fn: k(phi) function
            conductivity_derivative_fn: dk/dphi function
            n_dim: Spatial dimension
        """
        self.k_fn = conductivity_fn
        self.dk_fn = conductivity_derivative_fn
        self.n_dim = n_dim
        self.n_state = 1  # Store phi for history-dependent k
        self.is_vectorized = True

    @property
    def n_dof_per_node(self) -> int:
        """Scalar field: 1 DOF per node."""
        return 1

    def vectorize(self, n_elem: int) -> 'NonlinearDiffusionOperator':
        return self  # Already vectorizable

    def evaluate(
        self,
        gradient_inc: Tensor,
        deformation: Tensor,
        flux: Tensor,
        state: Tensor,
        external_inc: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Nonlinear diffusion with phi-dependent conductivity."""
        # Extract phi from state
        phi = state[..., 0]
        k = self.k_fn(phi)
        dk = self.dk_fn(phi)

        grad_phi = torch.diagonal(gradient_inc, dim1=-2, dim2=-1)
        flux_new = -k[..., None] * grad_phi
        flux_tensor = torch.diag_embed(flux_new)

        # Update state with new phi (would need phi itself)
        state_new = state  # Placeholder

        # Tangent includes dk/dphi term
        I2 = torch.eye(self.n_dim)
        K_tensor = k[..., None, None] * I2
        tangent = -torch.einsum('...ij,kl->...ikjl', K_tensor, I2)

        return flux_tensor, state_new, tangent

    def compute_element_gradient(
        self,
        field_elem: Tensor,
        B: Tensor,
    ) -> Tensor:
        """
        Compute scalar field gradient ∇φ.

        Args:
            field_elem: Element scalar field values
                Shape: (n_elem, n_nodes_per_elem, 1)
            B: Shape function gradient matrix
                Shape: (n_elem, n_dim, n_nodes_per_elem)

        Returns:
            Gradient vector ∇φ
                Shape: (n_elem, n_dim)
        """
        phi = field_elem.squeeze(-1)
        return torch.einsum("...ik,...k->...i", B, phi)

    def assemble_element_stiffness(
        self,
        B: Tensor,
        tangent: Tensor,
        detJ: Tensor,
        weight: float,
    ) -> Tensor:
        """
        Assemble nonlinear diffusion stiffness: K = ∫ ∇N^T k(φ) ∇N dV.

        Args:
            B: Shape function gradients (n_elem, n_dim, n_nodes_per_elem)
            tangent: k(phi) evaluated at current state
            detJ: Jacobian determinant (n_elem,) or scalar
            weight: Quadrature weight

        Returns:
            Element stiffness (n_elem, n_nodes_per_elem, n_nodes_per_elem)
        """
        # For nonlinear diffusion, tangent varies per element
        # Extract k from evaluate() which stored it in tangent
        # K_ij = ∫ (∂N_i/∂x_k) k (∂N_j/∂x_k) dV
        K_elem = torch.einsum("...ki,...kj->...ij", B, B)

        # Note: k weighting should be done per quadrature point
        # This is simplified; full implementation needs k evaluation
        if isinstance(detJ, Tensor):
            K_elem = K_elem * (detJ[..., None, None] * weight)
        else:
            K_elem = K_elem * (detJ * weight)

        return K_elem

    @property
    def C(self) -> Tensor:
        raise NotImplementedError(
            "Nonlinear diffusion C depends on solution"
        )
