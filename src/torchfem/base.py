from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cached_property
from typing import Literal, Tuple

import torch
from torch import Tensor

from .elements import Element
from .materials import Material
from .sparse import CachedSolve, differentiable_sparse_solve, newton_solve


class FEM(ABC):
    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize a general FEM problem."""

        # Store nodes and elements
        self.nodes = nodes
        self.elements = elements

        # Compute problem size
        self.n_dofs = self.n_dof_per_node * nodes.shape[0]
        self.n_nod = nodes.shape[0]
        self.n_dim = nodes.shape[1]
        self.n_elem = len(self.elements)
        self.n_int = len(self.etype.iweights)

        # Initialize boundary conditions
        self._neumann = torch.zeros(self.n_nod, self.n_dof_per_node)
        self._dirichlet = torch.zeros(self.n_nod, self.n_dof_per_node)
        self._constraints = torch.zeros(
            self.n_nod, self.n_dof_per_node, dtype=torch.bool
        )
        self._external_gradient = torch.zeros(self.n_elem, *self.n_flux)

        # Compute mapping from local to global indices
        idx = (self.n_dof_per_node * self.elements).unsqueeze(-1) + torch.arange(
            self.n_dof_per_node
        )
        self.idx = idx.reshape(self.n_elem, -1)

        # Precompute global index mapping for sparse matrix assembly
        n = self.idx.shape[1]
        cols = self.idx.unsqueeze(1).expand(self.n_elem, n, -1).ravel()
        rows = self.idx.unsqueeze(-1).expand(self.n_elem, -1, n).ravel()
        diag = torch.arange(self.n_dofs, dtype=torch.int64)
        packed = torch.cat([(rows << 32) | cols, (diag << 32) | diag])
        glob_idx_packed, inverse = torch.unique(packed, return_inverse=True)
        self.glob_idx = torch.stack(
            [
                torch.div(glob_idx_packed, 2**32, rounding_mode="floor"),
                glob_idx_packed % 2**32,
            ]
        ).to(torch.int32)
        self.idx = self.idx.to(torch.int32)
        inverse = inverse.to(torch.int32)
        m = rows.numel()
        self.k_map = inverse[:m]
        self.diag_map = inverse[m:]

        # Vectorize material
        if material.is_vectorized:
            self.material = material
        else:
            self.material = material.vectorize(self.n_elem)

        # Cached solve for sparse linear systems
        self.cached_solve = CachedSolve()

    @property
    @abstractmethod
    def n_flux(self) -> list[int]:
        """Shape of the flux tensor."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_dof_per_node(self) -> int:
        """Number of DOFs per node"""
        raise NotImplementedError

    @property
    @abstractmethod
    def etype(self) -> type[Element]:
        """Element type."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def char_lengths(self) -> Tensor:
        """Characteristic lengths of the elements."""
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_grad(self) -> Tensor:
        """Initial gradient at integration points."""
        raise NotImplementedError

    @abstractmethod
    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        raise NotImplementedError

    @abstractmethod
    def plot(self, u: float | Tensor = 0.0, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_m(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def k0(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def integrate_material(
        self,
        u_prev: Tensor,
        grad_prev: Tensor,
        flux_prev: Tensor,
        state_prev: Tensor,
        du: Tensor,
        de0: Tensor,
        iter: int,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Integrate material response over element integration points."""
        raise NotImplementedError

    @property
    def constraints(self) -> Tensor:
        return self._constraints

    @constraints.setter
    def constraints(self, value: Tensor):
        if not value.shape == (self.n_nod, self.n_dof_per_node):
            raise ValueError("Constraints must have the same shape as nodes.")
        if value.dtype != torch.bool:
            raise TypeError("Constraints must be a boolean tensor.")
        self._constraints = value.to(self.nodes.device)

    def eval_shape_functions(self, xi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Gradient operator at integration points xi."""
        nodes = self.nodes[self.elements, :]
        b = self.etype.B(xi)
        J = torch.einsum("...iN, ANj -> ...Aij", b, nodes)
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise ValueError("Negative Jacobian. Check element numbering.")
        B = torch.einsum("...Eij,...jN->...EiN", torch.linalg.inv(J), b)
        return self.etype.N(xi), B, detJ

    def compute_B(self) -> Tensor:
        """Null space representing rigid body modes."""
        if self.n_dof_per_node == 3:
            B = torch.zeros((self.n_dof_per_node * self.n_nod, 6))
            B[0::3, 0] = 1
            B[1::3, 1] = 1
            B[2::3, 2] = 1
            B[1::3, 3] = -self.nodes[:, 2]
            B[2::3, 3] = self.nodes[:, 1]
            B[0::3, 4] = self.nodes[:, 2]
            B[2::3, 4] = -self.nodes[:, 0]
            B[0::3, 5] = -self.nodes[:, 1]
            B[1::3, 5] = self.nodes[:, 0]
        elif self.n_dof_per_node == 2:
            B = torch.zeros((self.n_dof_per_node * self.n_nod, 3))
            B[0::2, 0] = 1
            B[1::2, 1] = 1
            B[1::2, 2] = -self.nodes[:, 0]
            B[0::2, 2] = self.nodes[:, 1]
        elif self.n_dof_per_node == 1:
            B = torch.zeros((self.n_dof_per_node * self.n_nod, 1))
            B[0::1, 0] = 1
        else:
            B = torch.ones((self.n_dof_per_node * self.n_nod, 1))
        return B

    def integrate_field(self, field: Tensor | None = None) -> Tensor:
        """Integrate scalar field over elements."""

        # Default field is ones (to integrate volume)
        if field is None:
            field = torch.ones(self.n_nod)

        # Shape functions at integration points
        N, _, detJ = self.eval_shape_functions(self.etype.ipoints)

        # Integration weights
        weights = self.etype.iweights

        # Field at integration points
        f_ip = torch.einsum("ej,ij->ie", field[self.elements], N)

        # Integration
        return torch.einsum("i,ie,ie->e", weights, f_ip, detJ)

    def assemble_matrix(self, k: Tensor, con: Tensor) -> Tensor:
        """Assemble global generalized stiffness matrix."""

        # Fill in stiffness matrix values at appropriate indices
        val = torch.zeros(self.glob_idx.shape[1])
        val.index_add_(0, self.k_map, k.ravel())

        # Apply Dirichlet boundary conditions
        self.is_constrained = torch.zeros(self.n_dofs, dtype=torch.bool)
        self.is_constrained[con] = True
        row_con = self.is_constrained[self.glob_idx[0]]
        col_con = self.is_constrained[self.glob_idx[1]]
        val[row_con | col_con] = 0.0
        val[self.diag_map[con]] = 1.0

        # Create sparse global stiffness matrix
        K = torch.sparse_coo_tensor(
            self.glob_idx, val, size=(self.n_dofs, self.n_dofs), is_coalesced=True
        )
        return K

    def assemble_rhs(self, f: Tensor) -> Tensor:
        """Assemble global right hand side vector."""

        # Initialize global right hand side vector
        F = torch.zeros((self.n_dofs))

        # Ravel indices and values
        indices = self.idx.ravel()
        values = f.ravel()

        return F.index_add_(0, indices, values)

    def solve(
        self,
        increments: Tensor = torch.tensor([0.0, 1.0]),
        max_iter: int = 100,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        verbose: bool = False,
        method: Literal["spsolve", "minres", "cg", "pardiso"] | None = None,
        device: str | None = None,
        return_intermediate: bool = False,
        aggregate_integration_points: bool = True,
        use_cached_solve: bool = False,
        nlgeom: bool = False,
        differentiable_parameters: Tensor | Iterable[Tensor] | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Solve the FEM problem with the Newton-Raphson method.

        Args:
            increments (Tensor): Load increment stepping.
            max_iter (int): Maximum number of iterations during Newton-Raphson.
            rtol (float): Relative tolerance for Newton-Raphson convergence.
            atol (float): Absolute tolerance for Newton-Raphson convergence.
            stol (float): Solver tolerance for iterative methods.
            verbose (bool): Print iteration information.
            method (str): Method for linear solve ('spsolve','minres','cg','pardiso').
            device (str): Device to run the linear solve on.
            return_intermediate (bool): Return intermediate values if True.
            aggregate_integration_points (bool): Aggregate integration points if True.
            use_cached_solve (bool): Use cached solve, e.g. in topology optimization.
            nlgeom (bool): Use nonlinear geometry if True.
            differentiable_parameters: Explicit parameter(s) to differentiate
                through the linear solves.

        Returns:
                Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Final displacements,
                internal forces, flux, deformation gradient, and material state.

        """
        # Number of increments
        N = len(increments)

        # Determine differentiable dependencies for this solve call.
        if differentiable_parameters is None:
            differentiable_parameters = tuple()
        elif isinstance(differentiable_parameters, torch.Tensor):
            differentiable_parameters = (differentiable_parameters,)
        else:
            differentiable_parameters = tuple(differentiable_parameters)

        has_differentiable_dependency = any(
            param.requires_grad for param in differentiable_parameters
        )

        # Null space rigid body modes for AMG preconditioner
        B = self.compute_B()

        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()

        # Initialize variables to be computed
        u = torch.zeros(N, self.n_nod, self.n_dof_per_node)
        f = torch.zeros(N, self.n_nod, self.n_dof_per_node)
        flux = torch.zeros(N, self.n_int, self.n_elem, *self.n_flux)
        grad = torch.zeros(N, self.n_int, self.n_elem, *self.n_flux)
        grad[:, :, :, :, :] = self.initial_grad
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)

        if verbose and u.dtype != torch.float64:
            print(
                "WARNING: Detected single precision floating points. It is highly "
                "recommended to use torch-fem with double precision by setting "
                "'torch.set_default_dtype(torch.float64)'."
            )

        # Initialize global stiffness matrix
        self.K = torch.empty(0)

        # Initialize field variable increment
        du = torch.zeros(self.n_nod, self.n_dof_per_node).ravel()

        # Incremental loading
        for n in range(1, N):
            if verbose:
                print(f"Starting increment {n} ...")

            # Increment size
            inc = increments[n] - increments[n - 1]

            # Load increment
            F_ext = increments[n] * self._neumann.ravel()
            DU = inc * self._dirichlet.ravel()
            de0 = inc * self._external_gradient

            # Residual for Newton-Raphson iterations
            def eval_residual(
                du,
                i,
                u_prev=u[n - 1].detach(),
                grad_prev=grad[n - 1].detach(),
                flux_prev=flux[n - 1].detach(),
                state_prev=state[n - 1].detach(),
            ):
                # Enforce Dirichlet BCs on increment
                du_bc = du.clone()
                du_bc[con] = DU[con]

                # Element-wise integration
                k, f_i, _, _, _ = self.integrate_material(
                    u_prev,
                    grad_prev,
                    flux_prev,
                    state_prev,
                    du_bc,
                    de0,
                    i,
                    nlgeom,
                )

                # Assemble global stiffness matrix and internal force vector (if needed)
                if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                    self.K = self.assemble_matrix(k, con)
                F_int = self.assemble_rhs(f_i)

                # Compute baseline force from previous stress (du=0) to
                # stop gradient of the parameter-dependent scaling of the
                # accumulated stress.  This ensures dR/dp only reflects
                # the incremental stiffness contribution.
                if has_differentiable_dependency:
                    _, f_base, _, _, _ = self.integrate_material(
                        u_prev,
                        grad_prev,
                        flux_prev,
                        state_prev,
                        torch.zeros_like(du_bc),
                        torch.zeros_like(de0),
                        i,
                        nlgeom,
                    )
                    F_int_base = self.assemble_rhs(f_base)
                    F_int = (F_int - F_int_base) + F_int_base.detach()

                # Compute residual
                res = F_int - F_ext
                res[con] = 0.0

                return res, self.K

            # Solve for increment using Newton-Raphson method
            if use_cached_solve:
                cached_solve = self.cached_solve
            else:
                cached_solve = CachedSolve()

            du = newton_solve(
                eval_residual,
                du,
                B,
                max_iter,
                rtol,
                atol,
                stol,
                verbose,
                method,
                device,
                cached_solve,
                use_cached_solve,
                *differentiable_parameters,
            )

            # Evaluate converged state
            du_eval = du.clone()
            du_eval[con] = DU[con]
            _, f_i, grad[n], flux[n], state[n] = self.integrate_material(
                u[n - 1],
                grad[n - 1],
                flux[n - 1],
                state[n - 1],
                du_eval,
                de0,
                max_iter,
                nlgeom,
            )
            F_int = self.assemble_rhs(f_i)
            # Detach f[n] to avoid spurious gradient paths through
            # accumulated stress; for compliance c=f·u, gradients
            # should flow through u only (adjoint chain).
            f[n] = F_int.reshape((-1, self.n_dof_per_node)).detach()
            u[n] = u[n - 1] + du_eval.reshape((-1, self.n_dof_per_node))
            du = du_eval

        # Create output views without mutating tensors captured by eval_residual.
        out_flux = flux
        out_grad = grad
        out_state = state

        # Aggregate integration points as mean
        if aggregate_integration_points:
            out_grad = out_grad.mean(dim=1)
            out_flux = out_flux.mean(dim=1)
            out_state = out_state.mean(dim=1)

        # Squeeze outputs
        out_flux = out_flux.squeeze()
        out_grad = out_grad.squeeze()

        # In pure forward evaluations, detach output fields
        # (gradients may come from hyperelasticity)
        if not has_differentiable_dependency:
            out_flux = out_flux.detach()
            out_grad = out_grad.detach()
            out_state = out_state.detach()

        if return_intermediate:
            # Return all intermediate values
            return u, f, out_flux, out_grad, out_state
        else:
            # Return only the final values
            return u[-1], f[-1], out_flux[-1], out_grad[-1], out_state[-1]


class Mechanics(FEM, ABC):

    @property
    def n_dof_per_node(self) -> int:
        return self.nodes.shape[1]

    @property
    def initial_grad(self) -> Tensor:
        return torch.eye(self.n_flux[0])

    @property
    def forces(self) -> Tensor:
        return self._neumann

    @forces.setter
    def forces(self, value: Tensor):
        if not value.shape == (self.n_nod, self.n_dof_per_node):
            raise ValueError("Forces must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Forces must be a floating-point tensor.")
        self._neumann = value.to(self.nodes.device)

    @property
    def displacements(self) -> Tensor:
        return self._dirichlet

    @displacements.setter
    def displacements(self, value: Tensor):
        if not value.shape == (self.n_nod, self.n_dof_per_node):
            raise ValueError("Displacements must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Displacements must be a floating-point tensor.")
        self._dirichlet = value.to(self.nodes.device)

    @property
    def ext_strain(self) -> Tensor:
        return self._external_gradient

    @ext_strain.setter
    def ext_strain(self, value: Tensor):
        if not value.shape == (self.n_nod, self.n_dof_per_node, self.n_dim):
            raise ValueError("External strain must have the same shape as strains.")
        if not torch.is_floating_point(value):
            raise TypeError("External strain must be a floating-point tensor.")
        self._external_gradient = value.to(self.nodes.device)

    def k0(self) -> Tensor:
        """Compute element stiffness matrix for zero strain."""
        u = torch.zeros(self.n_nod, self.n_dof_per_node)
        grad = torch.zeros(self.n_int, self.n_elem, *self.n_flux)
        grad[:] = torch.eye(self.n_dim)
        flux = torch.zeros(self.n_int, self.n_elem, *self.n_flux)
        state = torch.zeros(self.n_int, self.n_elem, self.material.n_state)
        du = torch.zeros(self.n_nod, self.n_dof_per_node)
        de0 = torch.zeros(self.n_elem, *self.n_flux)
        self.K = torch.empty(0)
        k, _, _, _, _ = self.integrate_material(u, grad, flux, state, du, de0, 0, False)
        return k

    def integrate_material(
        self,
        u_prev: Tensor,
        grad_prev: Tensor,
        flux_prev: Tensor,
        state_prev: Tensor,
        du: Tensor,
        de0: Tensor,
        iter: int,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Perform numerical integrations for the element stiffness matrix and forces
        assuming a total Lagrangian formulation.

        Args:
            grad_prev: Deformation gradient at previous step [n_int, n_elem, *n_flux]
            flux_prev: Stress at previous step [n_int, n_elem, *n_flux]
            state_prev: Material state at previous step [n_int, n_elem, n_state]

        Returns:
            k, f, grad_new, flux_new, state_new
        """

        # Reshape displacement increment
        du = (
            du.view(-1, self.n_dof_per_node)[self.elements]
            .reshape(self.n_elem, -1, self.n_flux[0])
            .transpose(-1, -2)
        )

        # Initialize nodal force and stiffness
        N_nod = self.etype.nodes
        N_dof = self.n_dof_per_node
        f = torch.zeros(self.n_elem, N_dof * N_nod)
        k = torch.zeros((self.n_elem, N_dof * N_nod, N_dof * N_nod))

        # Initialize output for new state
        grad_new = torch.zeros_like(grad_prev)
        flux_new = torch.zeros_like(flux_prev)
        state_new = torch.zeros_like(state_prev)

        # Compute gradient operators
        _, B, detJ = self.eval_shape_functions(self.etype.ipoints)

        for i, w in enumerate(self.etype.iweights):
            # Compute displacement gradient increment (Batch, Spatial, Material)
            H_inc = du @ B[i].transpose(-1, -2)

            # Current deformation gradient for this Newton evaluation.
            F_new = grad_prev[i] + H_inc

            # Evaluate material response
            P, alpha, ddsdde = self.material.step(
                H_inc,
                grad_prev[i],
                flux_prev[i],
                state_prev[i],
                de0,
                self.char_lengths,
                iter,
            )

            # Store updated deformation gradient
            grad_new[i] = F_new

            # Compute new Cauchy stress
            if nlgeom:
                J = torch.det(F_new)[:, None, None]
                flux_new[i] = (F_new @ P) / J
            else:
                flux_new[i] = P

            # Store new state
            state_new[i] = alpha

            # Compute element internal forces
            force_contrib = self.compute_f(detJ[i], B[i], P)
            f += w * force_contrib.reshape(-1, N_dof * N_nod)

            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                BCB = torch.einsum("...Jp,...iJkL,...Lq->...piqk", B[i], ddsdde, B[i])
                BCB = BCB.reshape(-1, N_dof * N_nod, N_dof * N_nod)
                k += w * self.compute_k(detJ[i], BCB)

        return k, f, grad_new, flux_new, state_new

    def compute_m(self) -> Tensor:
        raise NotImplementedError


class Heat(FEM, ABC):
    @property
    def n_dof_per_node(self) -> int:
        return 1

    @property
    def n_flux(self) -> list[int]:
        """Shape of the heat flux tensor."""
        return [1, self.n_dim]

    @property
    def initial_grad(self) -> Tensor:
        return torch.zeros(1)

    @property
    def heat_flux(self) -> Tensor:
        return self._neumann

    @heat_flux.setter
    def heat_flux(self, value: Tensor):
        if not value.shape == (self.n_nod, 1):
            raise ValueError("Heat flux must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Heat flux must be a floating-point tensor.")
        self._neumann = value.to(self.nodes.device)

    @property
    def temperatures(self) -> Tensor:
        return self._dirichlet

    @temperatures.setter
    def temperatures(self, value: Tensor):
        if not value.shape == (self.n_nod, self.n_dof_per_node):
            raise ValueError("Temperatures must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Temperatures must be a floating-point tensor.")
        self._dirichlet = value.to(self.nodes.device)

    def k0(self) -> Tensor:
        """Compute element stiffness matrix for zero strain."""
        temp = torch.zeros(self.n_nod, self.n_dof_per_node)  # temperature
        temp_grad = torch.zeros(
            self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim
        )
        heat_flux = torch.zeros(
            self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim
        )  # heat flux
        state = torch.zeros(self.n_int, self.n_elem, self.material.n_state)
        dtemp = torch.zeros(self.n_nod, self.n_dof_per_node)  # temperature increment
        dtemp_grad0 = torch.zeros(
            self.n_elem, self.n_dof_per_node, self.n_dim
        )  # temperature gradient increment
        self.K = torch.empty(0)
        k, _, _, _, _ = self.integrate_material(
            temp, temp_grad, heat_flux, state, dtemp, dtemp_grad0, 0, False
        )
        return k

    def integrate_material(
        self,
        u_prev: Tensor,
        grad_prev: Tensor,
        flux_prev: Tensor,
        state_prev: Tensor,
        du: Tensor,
        de0: Tensor,
        iter: int,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix.

        Args:
            grad_prev: Previous temp. gradient [n_int, n_elem, n_dof_per_node, n_dim]
            flux_prev: Previous heat flux [n_int, n_elem, n_dof_per_node, n_dim]
            state_prev: Previous material state [n_int, n_elem, n_state]

        Returns:
            k, f, grad_new, flux_new, state_new
        """

        # Reshape temperature increment
        du = du.view(-1, self.n_dof_per_node)[self.elements].reshape(
            self.n_elem, -1, self.n_dof_per_node
        )

        # Initialize nodal heat fluxes and conductivity matrix
        N_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dof_per_node * N_nod)
        k = torch.zeros(
            (self.n_elem, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod)
        )

        # Initialize output for new state
        grad_new = torch.zeros_like(grad_prev)
        flux_new = torch.zeros_like(flux_prev)
        state_new = torch.zeros_like(state_prev)

        # Compute gradient operators
        _, B, detJ = self.eval_shape_functions(self.etype.ipoints)

        for i, w in enumerate(self.etype.iweights):

            # Compute temperature gradient increment
            temp_grad_inc = torch.einsum("...ij,...jk->...ki", B[i], du)
            # Update deformation gradient
            grad_new[i] = grad_prev[i] + temp_grad_inc

            # Evaluate material response
            flux_new[i], state_new[i], ddfddg = self.material.step(
                temp_grad_inc,
                grad_prev[i],
                flux_prev[i],
                state_prev[i],
                de0,
                self.char_lengths,
                iter,
            )

            # Compute element internal forces
            force_contrib = self.compute_f(detJ[i], B[i], flux_new[i])
            f += w * force_contrib.reshape(-1, self.n_dof_per_node * N_nod)

            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0:
                # Material stiffness
                BCB = torch.einsum("...ij,...iN,...jM->...NM", ddfddg, B[i], B[i])
                BCB = BCB.reshape(
                    -1, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod
                )
                k += w * self.compute_k(detJ[i], BCB)
        return k, f, grad_new, flux_new, state_new

    def time_integration(
        self,
        t_output: Tensor = torch.tensor([0.0, 1.0]),
        delta_t: float = 1.0e-1,
        max_iter: int = 100,
        verbose: bool = False,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        device: str | None = None,
        method: Literal["spsolve", "minres", "cg", "pardiso"] | None = None,
        aggregate_integration_points: bool = True,
        return_intermediate: bool = False,
        use_cached_solve: bool = False,
        differentiable_parameters: Tensor | Iterable[Tensor] | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        # initial step: we get heat fluxes and temperature gradients for initial
        # conditions enforce initial conditions as boundary conditions

        bc_constraints = self.constraints.clone()
        self.constraints[:] = True

        # solve for initial conditions
        temp_eq, _, heat_flux_eq, temp_grad_eq, alpha_eq = self.solve(
            aggregate_integration_points=False,
            use_cached_solve=use_cached_solve,
            differentiable_parameters=differentiable_parameters,
        )

        # assemble time_steps for evaluation
        start_time = 0.0
        end_time = t_output.max().item()
        t_eval = torch.clamp(t_output, min=0.0, max=end_time)
        increments = torch.arange(start_time, end_time, delta_t)

        increments = torch.cat((increments, t_eval))
        increments = increments.unique(sorted=True)

        dt = increments[1:] - increments[:-1]  # time step sizes

        N_output = len(increments)

        # release boundary conditions, restore
        self.constraints[:] = bc_constraints

        # null space rigid body modes for AMG preconditioner
        B = self.compute_B()

        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()

        # Initialize variables to be computed
        u = torch.zeros(N_output, self.n_nod, self.n_dof_per_node)
        f = torch.zeros(N_output, self.n_nod, self.n_dof_per_node)
        flux = torch.zeros(
            N_output, self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim
        )
        grad = torch.zeros(
            N_output,
            self.n_int,
            self.n_elem,
            self.n_dof_per_node,
            self.n_dim,
        )
        state = torch.zeros(N_output, self.n_int, self.n_elem, self.material.n_state)

        # fill initial conditions
        u[0] = temp_eq
        # f[0] = reaction_flux
        flux[0] = heat_flux_eq.view(
            self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim
        )
        grad[0] = temp_grad_eq.view(
            self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim
        )
        state[0] = alpha_eq

        # Initialize stiffness matrix and mass matrix
        self.K = torch.empty(0)
        self.M = torch.empty(0)

        # compute element mass matrices
        m = self.compute_m()

        # Initialize displacement increment
        du = torch.zeros(self.n_nod, self.n_dof_per_node).ravel()
        # Enforce initial BCs on u[0] explicitly, in case line_heat._dirichlet gives
        # updated BCs
        u[0].view(-1)[con] = self._dirichlet.view(-1)[con]

        for n in range(1, N_output):
            u_guess = u[n - 1].clone()
            dt_n = dt[n - 1]
            f_int_old = f[n - 1].clone()

            for it in range(max_iter):
                du = u_guess - u[n - 1]
                k, f_int, grad[n], flux[n], state[n] = self.integrate_material(
                    u[n - 1],
                    grad[n - 1],
                    flux[n - 1],
                    state[n - 1],
                    du,
                    self._external_gradient,
                    it,
                    False,
                )
                f_int = self.assemble_rhs(f_int)
                f_ext = self._neumann.ravel()

                # assemble stiffness and mass matrices
                if self.K.numel() == 0:
                    self.K = self.assemble_matrix(k, con)
                if self.M.numel() == 0:
                    self.M = self.assemble_matrix(m, con)

                f_inertia = self.M @ du

                residual = f_inertia.squeeze(-1) + 0.5 * dt_n * (
                    f_int_old.squeeze(-1) + f_int.squeeze(-1) + f_ext
                )

                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)

                # save initial residual
                if it == 0:
                    res_norm0 = res_norm

                # Print iteration information
                if verbose:
                    print(
                        f"Increment {n} | Iteration {it + 1} | Residual: {res_norm:.5e}"
                    )

                if res_norm < rtol * res_norm0 or res_norm < atol:
                    break

                # Use cached solve from previous increment if available.
                if it == 0 and use_cached_solve:
                    cached_solve = self.cached_solve
                else:
                    cached_solve = CachedSolve()

                # Keep cache tied to first Newton iteration only.
                update_cache = it == 0

                du = differentiable_sparse_solve(
                    self.M + 0.5 * dt_n * self.K,
                    -residual,
                    B,
                    stol,
                    device,
                    method,
                    None,
                    cached_solve,
                    update_cache,
                )

                u_guess = u_guess + du.reshape((-1, self.n_dof_per_node))

            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise RuntimeError("Newton-Raphson iteration did not converge.")

            u[n] = u_guess
            f[n] = f_int.reshape((-1, self.n_dof_per_node))

        # Create output views without mutating tensors captured by autograd.
        out_flux = flux
        out_grad = grad
        out_state = state

        # Aggregate integration points as mean
        if aggregate_integration_points:
            out_grad = out_grad.mean(dim=1)
            out_flux = out_flux.mean(dim=1)
            out_state = out_state.mean(dim=1)
        # Squeeze outputs
        out_flux = out_flux.squeeze()
        out_grad = out_grad.squeeze()
        if return_intermediate:
            # Return all intermediate values
            return u, f, out_flux, out_grad, out_state
        else:
            # Return only the final values
            return u[-1], f[-1], out_flux[-1], out_grad[-1], out_state[-1]
