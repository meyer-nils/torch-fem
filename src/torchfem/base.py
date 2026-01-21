from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal, Tuple

import torch
from torch import Tensor

from .elements import Element
from .materials import Material
from .sparse import CachedSolve, sparse_solve


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
        self.glob_idx_packed, inverse = torch.unique(packed, return_inverse=True)
        self.idx = self.idx.to(torch.int32)
        inverse = inverse.to(torch.int32)
        self.glob_idx = torch.stack(
            [
                torch.div(self.glob_idx_packed, 2**32, rounding_mode="floor"),
                self.glob_idx_packed % 2**32,
            ]
        ).to(torch.int32)
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
        u: Tensor,
        grad: Tensor,
        flux: Tensor,
        state: Tensor,
        n: int,
        iter: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor]:
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

    def eval_shape_functions(
        self, xi: Tensor, u: Tensor | float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gradient operator at integration points xi."""
        nodes = self.nodes + u
        nodes = nodes[self.elements, :]
        b = self.etype.B(xi)
        J = torch.einsum("...iN, ANj -> ...Aij", b, nodes)
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")
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

        Returns:
                Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Final displacements,
                internal forces, flux, deformation gradient, and material state.

        """
        # Number of increments
        N = len(increments)

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
            # Increment size
            inc = increments[n] - increments[n - 1]

            # Load increment
            F_ext = increments[n] * self._neumann.ravel()
            DU = inc * self._dirichlet.clone().ravel()
            de0 = inc * self._external_gradient

            # Newton-Raphson iterations
            for i in range(max_iter):
                du[con] = DU[con]

                # Element-wise integration
                k, f_i = self.integrate_material(
                    u, grad, flux, state, n, i, du, de0, nlgeom
                )

                # Assemble global stiffness matrix and internal force vector (if needed)
                if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                    self.K = self.assemble_matrix(k, con)
                F_int = self.assemble_rhs(f_i)

                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)

                # Save initial residual
                if i == 0:
                    res_norm0 = res_norm

                # Print iteration information
                if verbose:
                    print(
                        f"Increment {n} | Iteration {i + 1} | Residual: {res_norm:.5e}"
                    )

                # Check convergence
                if res_norm < rtol * res_norm0 or res_norm < atol:
                    break

                if torch.isnan(res_norm) or torch.isinf(res_norm):
                    raise Exception("Newton-Raphson iteration did not converge")

                # Use cached solve from previous iteration if available
                if i == 0 and use_cached_solve:
                    cached_solve = self.cached_solve
                else:
                    cached_solve = CachedSolve()

                # Only update cache on first iteration
                update_cache = i == 0

                # Solve for displacement increment
                du -= sparse_solve(
                    self.K,
                    residual,
                    B,
                    stol,
                    device,
                    method,
                    None,
                    cached_solve,
                    update_cache,
                )

            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise Exception("Newton-Raphson iteration did not converge.")

            # Update increment
            f[n] = F_int.reshape((-1, self.n_dof_per_node))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dof_per_node))

        # Aggregate integration points as mean
        if aggregate_integration_points:
            grad = grad.mean(dim=1)
            flux = flux.mean(dim=1)
            state = state.mean(dim=1)

        # Squeeze outputs
        flux = flux.squeeze()
        grad = grad.squeeze()

        if return_intermediate:
            # Return all intermediate values
            return u, f, flux, grad, state
        else:
            # Return only the final values
            return u[-1], f[-1], flux[-1], grad[-1], state[-1]


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
        grad = torch.zeros(2, self.n_int, self.n_elem, *self.n_flux)
        grad[:, :, :, :, :] = torch.eye(self.n_dim)
        flux = torch.zeros(2, self.n_int, self.n_elem, *self.n_flux)
        state = torch.zeros(2, self.n_int, self.n_elem, self.material.n_state)
        du = torch.zeros(self.n_nod, self.n_dof_per_node)
        de0 = torch.zeros(self.n_elem, *self.n_flux)
        self.K = torch.empty(0)
        k, _ = self.integrate_material(u, grad, flux, state, 1, 0, du, de0, False)
        return k

    def integrate_material(
        self,
        u: Tensor,
        grad: Tensor,
        flux: Tensor,
        state: Tensor,
        n: int,
        iter: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform numerical integrations for the element stiffness matrix and forces
        assuming a total Lagrangian formulation.
        """

        # Mechanical interpretation of variables
        F = grad
        stress = flux

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

        # Compute gradient operators
        _, B, detJ = self.eval_shape_functions(self.etype.ipoints)

        for i, w in enumerate(self.etype.iweights):
            # Compute displacement gradient increment (Batch, Spatial, Material)
            H_inc = du @ B[i].transpose(-1, -2)

            # Evaluate material response
            P, alpha, ddsdde = self.material.step(
                H_inc,
                F[n - 1, i],
                stress[n - 1, i],
                state[n - 1, i],
                de0,
                self.char_lengths,
                iter,
            )

            # Compute new Cauchy stress
            if nlgeom:
                J = torch.det(F[n, i])[:, None, None]
                stress[n, i] = (F[n, i] @ P) / J
            else:
                stress[n, i] = P

            # Compute new deformation gradient
            F[n, i] = F[n - 1, i] + H_inc

            # Compute new state
            state[n, i] = alpha

            # Compute element internal forces
            force_contrib = self.compute_f(detJ[i], B[i], P)
            f += w * force_contrib.reshape(-1, N_dof * N_nod)

            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                BCB = torch.einsum("...Jp,...iJkL,...Lq->...piqk", B[i], ddsdde, B[i])
                BCB = BCB.reshape(-1, N_dof * N_nod, N_dof * N_nod)
                k += w * self.compute_k(detJ[i], BCB)

        return k, f

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
            2, self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim
        )
        heat_flux = torch.zeros(
            2, self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim
        )  # heat flux
        state = torch.zeros(
            2, self.n_int, self.n_elem, self.material.n_state
        )  # state variables
        dtemp = torch.zeros(self.n_nod, self.n_dof_per_node)  # temperature increment
        dtemp_grad0 = torch.zeros(
            self.n_elem, self.n_dof_per_node, self.n_dim
        )  # temperature gradient increment
        self.K = torch.empty(0)
        k, _ = self.integrate_material(
            temp,
            temp_grad,
            heat_flux,
            state,
            1,
            0,
            dtemp,
            dtemp_grad0,
        )
        return k

    def integrate_material(
        self,
        u: Tensor,
        grad: Tensor,
        flux: Tensor,
        state: Tensor,
        n: int,
        iter: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix."""

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

        # Compute gradient operators
        _, B, detJ = self.eval_shape_functions(self.etype.ipoints)

        for i, w in enumerate(self.etype.iweights):

            # Compute temperature gradient increment
            temp_grad_inc = torch.einsum("...ij,...jk->...ki", B[i], du)
            # Update deformation gradient
            grad[n, i] = grad[n - 1, i] + temp_grad_inc

            # Evaluate material response
            flux[n, i], state[n, i], ddfddg = self.material.step(
                temp_grad_inc,
                grad[n - 1, i],
                flux[n - 1, i],
                state[n - 1, i],
                de0,
                self.char_lengths,
                iter,
            )

            # Compute element internal forces
            force_contrib = self.compute_f(detJ[i], B[i], flux[n, i].clone())
            f += w * force_contrib.reshape(-1, self.n_dof_per_node * N_nod)

            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0:
                # Material stiffness
                BCB = torch.einsum("...ij,...iN,...jM->...NM", ddfddg, B[i], B[i])
                BCB = BCB.reshape(
                    -1, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod
                )
                k += w * self.compute_k(detJ[i], BCB)
        return k, f

    def time_integration(
        self,
        t_output: Tensor = torch.tensor([0.0, 1.0]),
        delta_t: float = 1.0e-1,
        max_iter: int = 100,
        verbose: bool = False,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        use_cached_solve: bool = False,
        device: str | None = None,
        method: Literal["spsolve", "minres", "cg", "pardiso"] | None = None,
        aggregate_integration_points: bool = True,
        return_intermediate: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # in heat transfer there is no geometric nonlinearity
        nlgeom = False

        # initial step: we get heat fluxes and temperature gradients for initial
        # conditions enforce initial conditions as boundary conditions

        bc_constraints = self.constraints.clone()
        self.constraints[:] = True

        # solve for initial conditions
        temp_eq, reaction_flux, heat_flux_eq, temp_grad_eq, alpha_eq = self.solve(
            aggregate_integration_points=False
        )

        # assemble time_steps for evaluation
        start_time = 0.0
        end_time = t_output.max().item()
        t_eval = torch.clamp(t_output, min=0.0, max=end_time)
        increments = torch.arange(start_time, end_time, delta_t)

        increments = torch.cat((increments, t_eval))
        increments = increments.unique(sorted=True)

        dt = increments[1:] - increments[:-1]  # time step sizes

        N_inc = len(increments)  # number of temporal increments
        # N_output = len(t_output)  # number of output steps
        N_output = N_inc  # current design requires to explicitly store all increments.

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
        state = torch.zeros(
            N_output, self.n_int, self.n_int, self.n_elem, self.material.n_state
        )

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

        for n, t_n in enumerate(increments[1:], 1):
            u_guess = u[n - 1].clone()
            dt_n = dt[n - 1]
            f_int_old = f[n - 1].clone()

            for it in range(max_iter):
                du = u_guess - u[n - 1]
                k, f_int = self.integrate_material(
                    u,
                    grad,
                    flux,
                    state,
                    n,
                    it,
                    du,
                    self._external_gradient,
                    nlgeom,
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

                # Use cached solve from previous iteration if available
                if it == 0 and use_cached_solve:
                    cached_solve = self.cached_solve
                else:
                    cached_solve = CachedSolve()
                # only update cache on first iteration
                update_cache = it == 0
                du = sparse_solve(
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

                u_guess += du.reshape((-1, self.n_dof_per_node))

                # u_old = u_guess.clone()
                # f_int_old = f_int.clone()

            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise Exception("Newton-Raphson iteration did not converge.")

            u[n] = u_guess
            f[n] = f_int.reshape((-1, self.n_dof_per_node))

        # Aggregate integration points as mean
        if aggregate_integration_points:
            grad = grad.mean(dim=1)
            flux = flux.mean(dim=1)
            state = state.mean(dim=1)
        # Squeeze outputs
        flux = flux.squeeze()
        grad = grad.squeeze()
        if return_intermediate:
            # Return all intermediate values
            return u, f, flux, grad, state
        else:
            # Return only the final values
            return u[-1], f[-1], flux[-1], grad[-1], state[-1]
