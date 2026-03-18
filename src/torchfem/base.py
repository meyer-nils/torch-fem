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
        """Initialize a finite-element model.

        Args:
            nodes: Nodal coordinates with shape [n_nod, n_dim].
            elements: Connectivity with shape [n_elem, n_nodes_per_element].
            material: Material model. If not vectorized, it is vectorized over
                elements during initialization.
        """

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
        """Number of degrees of freedom per node."""
        raise NotImplementedError

    @property
    @abstractmethod
    def etype(self) -> type[Element]:
        """Finite-element type implementation used by this problem."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def char_lengths(self) -> Tensor:
        """Characteristic element lengths.

        Returns:
            Tensor with one characteristic length per element.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_grad(self) -> Tensor:
        """Initial gradient field value at integration points."""
        raise NotImplementedError

    @abstractmethod
    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        """Compute element stiffness contribution.

        Args:
            detJ: Jacobian determinant at the current integration point.
            BCB: Material tangent transformed by gradient operators.

        Returns:
            Element stiffness contribution tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        """Compute element internal force contribution.

        Args:
            detJ: Jacobian determinant at the current integration point.
            B: Gradient operator at the current integration point.
            S: Stress or flux-like constitutive quantity.

        Returns:
            Element internal nodal force contribution.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self, u: float | Tensor = 0.0, **kwargs):
        """Visualize the model and optionally a solution field.

        Args:
            u: Optional nodal field or scale factor, depending on subclass.
            **kwargs: Backend-specific plotting keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_m(self) -> Tensor:
        """Compute element mass matrix contributions."""
        raise NotImplementedError

    @abstractmethod
    def k0(self) -> Tensor:
        """Compute element stiffness for the reference configuration."""
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
        """Integrate constitutive response over all integration points.

        Args:
            u_prev: Nodal field at previous converged step.
            grad_prev: Previous gradient field at integration points.
            flux_prev: Previous flux or stress at integration points.
            state_prev: Previous material internal variables.
            du: Incremental nodal unknown for the current Newton evaluation.
            de0: Incremental external gradient-like loading term.
            iter: Newton iteration index.
            nlgeom: If True, evaluate with geometric nonlinearity.

        Returns:
            Tuple of element stiffness, element internal forces, updated
            gradients, updated fluxes, and updated material state.
        """
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
        """Evaluate shape functions and gradients at local coordinates.

        Args:
            xi: Local element coordinates where quantities are evaluated.

        Returns:
            Tuple of shape functions N, gradient operators B, and Jacobian
            determinants detJ.

        Raises:
            ValueError: If any element has non-positive Jacobian determinant.
        """
        nodes = self.nodes[self.elements, :]
        xi = xi.to(nodes.device)
        b = self.etype.B(xi)
        J = torch.einsum("...iN, ANj -> ...Aij", b, nodes)
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise ValueError("Negative Jacobian. Check element numbering.")
        B = torch.einsum("...Eij,...jN->...EiN", torch.linalg.inv(J), b)
        return self.etype.N(xi), B, detJ

    def compute_B(self) -> Tensor:
        """Build rigid-body null-space modes for linear solvers.

        Returns:
            Dense basis matrix with one column per rigid-body mode.
        """
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
        """Integrate a nodal scalar field over each element.

        Args:
            field: Nodal scalar values with shape [n_nod]. If None, integrates
                a unit field and therefore returns element volumes or areas.

        Returns:
            Per-element integral values with shape [n_elem].
        """

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
        """Assemble a global sparse matrix from element contributions.

        Args:
            k: Element matrix contributions.
            con: Flattened indices of constrained global degrees of freedom.

        Returns:
            Global sparse matrix with Dirichlet constraints enforced.
        """

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
        """Assemble a global right-hand-side vector from element values.

        Args:
            f: Element nodal vector contributions.

        Returns:
            Global vector with shape [n_dofs].
        """

        # Initialize global right hand side vector
        F = torch.zeros((self.n_dofs), device=f.device)

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
        """Solve the quasi-static finite-element problem by load increments.

        Args:
            increments: Monotonic load scale factors, typically [0, 1].
            max_iter: Maximum Newton iterations per load increment.
            rtol: Relative residual tolerance for Newton convergence.
            atol: Absolute residual tolerance for Newton convergence.
            stol: Tolerance used by iterative linear solvers.
            verbose: If True, prints per-increment progress.
            method: Linear solver backend name.
            device: Optional device hint for the linear solver backend.
            return_intermediate: If True, returns values for all increments.
            aggregate_integration_points: If True, averages flux, gradient, and
                state over integration points.
            use_cached_solve: If True, reuses cached linear solver data.
            nlgeom: If True, includes geometric nonlinearity.
            differentiable_parameters: Explicit parameter(s) to differentiate
                through implicit Newton/sparse solves. Accepts either a single
                tensor or an iterable of tensors.

        Returns:
            Tuple of displacement, internal force, flux, gradient, and material
            state. If return_intermediate is True, each tensor includes an
            increment dimension as the leading axis.

        """
        # Number of increments
        N = len(increments)

        # Determine differentiable dependencies for this solve call.
        if differentiable_parameters is None:
            differentiable_parameters = ()
        elif isinstance(differentiable_parameters, torch.Tensor):
            differentiable_parameters = (differentiable_parameters,)
        else:
            differentiable_parameters = tuple(differentiable_parameters)

        track_parameter_gradients = any(
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
            u_prev = u[n - 1].detach()
            grad_prev = grad[n - 1].detach()
            flux_prev = flux[n - 1].detach()
            state_prev = state[n - 1].detach()

            # Residual for Newton-Raphson iterations
            def eval_residual(du, i):
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
                if self.K.numel() == 0 or self.material.n_state != 0 or nlgeom:
                    self.K = self.assemble_matrix(k, con)
                F_int = self.assemble_rhs(f_i)

                # Compute baseline force from previous stress (du=0) to
                # stop gradient of the parameter-dependent scaling of the
                # accumulated stress.  This ensures dR/dp only reflects
                # the incremental stiffness contribution. This is only needed
                # during the adjoint backward replay (where du requires grad),
                # not during forward Newton iterations.
                if track_parameter_gradients and du.requires_grad:
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
            # Detach f[n] to avoid spurious gradient paths through accumulated stress
            f[n] = F_int.reshape((-1, self.n_dof_per_node)).detach()
            u[n] = u[n - 1] + du_eval.reshape((-1, self.n_dof_per_node))
            du = du_eval

        # Create output views without mutating tensors captured by eval_residual.
        out_flux = flux
        out_grad = grad
        out_state = state

        if aggregate_integration_points:
            out_grad = out_grad.mean(dim=1)
            out_flux = out_flux.mean(dim=1)
            out_state = out_state.mean(dim=1)

        out_flux = out_flux.squeeze()
        out_grad = out_grad.squeeze()

        if not track_parameter_gradients:
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
    """Base class for solid and structural mechanics formulations."""

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
        """Compute element stiffness matrix in the reference state."""
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
        Integrate mechanics material response over all integration points.

        Args:
            u_prev: Nodal displacement field at previous converged step.
            grad_prev: Deformation gradient at previous step
                [n_int, n_elem, *n_flux].
            flux_prev: Stress tensor at previous step [n_int, n_elem, *n_flux].
            state_prev: Internal variables at previous step
                [n_int, n_elem, n_state].
            du: Displacement increment used for the current Newton evaluation.
            de0: External strain-like increment per element.
            iter: Newton iteration index.
            nlgeom: If True, computes Cauchy stress from first Piola stress.

        Returns:
            k: Element stiffness contributions.
            f: Element internal nodal forces.
            grad_new: Updated deformation gradient.
            flux_new: Updated stress tensor.
            state_new: Updated internal material state.
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
        f = torch.zeros(self.n_elem, N_dof * N_nod, device=du.device)
        k = torch.zeros(
            (self.n_elem, N_dof * N_nod, N_dof * N_nod), device=du.device
        )

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
            if self.K.numel() == 0 or self.material.n_state != 0 or nlgeom:
                BCB = torch.einsum("...Jp,...iJkL,...Lq->...piqk", B[i], ddsdde, B[i])
                BCB = BCB.reshape(-1, N_dof * N_nod, N_dof * N_nod)
                k += w * self.compute_k(detJ[i], BCB)

        return k, f, grad_new, flux_new, state_new

    def compute_m(self) -> Tensor:
        raise NotImplementedError


class Heat(FEM, ABC):
    """Base class for steady and transient heat conduction formulations."""

    @property
    def n_dof_per_node(self) -> int:
        return 1

    @property
    def n_flux(self) -> list[int]:
        """Heat flux tensor shape per integration point."""
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
        """Compute element conductivity matrix in the reference state."""
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
        """Integrate thermal constitutive response over all integration points.

        Args:
            u_prev: Previous nodal temperature field.
            grad_prev: Previous temperature gradient
                [n_int, n_elem, n_dof_per_node, n_dim].
            flux_prev: Previous heat flux
                [n_int, n_elem, n_dof_per_node, n_dim].
            state_prev: Previous internal variables [n_int, n_elem, n_state].
            du: Temperature increment for the current Newton evaluation.
            de0: External temperature-gradient increment per element.
            iter: Newton iteration index.
            nlgeom: Unused for heat, kept for API compatibility.

        Returns:
            k: Element conductivity contributions.
            f: Element internal nodal heat-flux vector contributions.
            grad_new: Updated temperature gradient.
            flux_new: Updated heat flux.
            state_new: Updated internal material state.
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

        grad_new = []
        flux_new = []
        state_new = []

        # Compute gradient operators
        _, B, detJ = self.eval_shape_functions(self.etype.ipoints)

        for i, w in enumerate(self.etype.iweights):

            # Compute temperature gradient increment
            temp_grad_inc = torch.einsum("...ij,...jk->...ki", B[i], du)
            # Update deformation gradient
            grad_new.append(grad_prev[i] + temp_grad_inc)

            # Evaluate material response
            flux_i, state_i, ddfddg = self.material.step(
                temp_grad_inc,
                grad_prev[i],
                flux_prev[i],
                state_prev[i],
                de0,
                self.char_lengths,
                iter,
            )
            flux_new.append(flux_i)
            state_new.append(state_i)

            # Compute element internal forces
            force_contrib = self.compute_f(detJ[i], B[i], flux_i)
            f += w * force_contrib.reshape(-1, self.n_dof_per_node * N_nod)

            # Compute element stiffness matrix
            if self.K.numel() == 0 or self.material.n_state != 0:
                # Material stiffness
                BCB = torch.einsum("...ij,...iN,...jM->...NM", ddfddg, B[i], B[i])
                BCB = BCB.reshape(
                    -1, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod
                )
                k += w * self.compute_k(detJ[i], BCB)

        return (
            k,
            f,
            torch.stack(grad_new),
            torch.stack(flux_new),
            torch.stack(state_new),
        )

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
        """Integrate the heat equation in time with implicit increments.

        The routine first computes a consistent equilibrium state at the
        initial time under the current boundary conditions, then advances the
        solution over the requested output times.

        Args:
            t_output: Requested output times.
            delta_t: Maximum internal time step.
            max_iter: Maximum Newton iterations per time step.
            verbose: If True, prints per-step Newton residuals.
            rtol: Relative residual tolerance for Newton convergence.
            atol: Absolute residual tolerance for Newton convergence.
            stol: Tolerance used by iterative linear solvers.
            device: Optional device hint for the linear solver backend.
            method: Linear solver backend name.
            aggregate_integration_points: If True, averages flux, gradient, and
                state over integration points.
            return_intermediate: If True, returns all intermediate increments.
            use_cached_solve: If True, reuses cached linear solver data.
            differentiable_parameters: Explicit parameters that should receive
                gradients through implicit solves. Accepts either a single
                tensor or an iterable of tensors.

        Returns:
            Tuple of temperature, internal vector, heat flux, temperature
            gradient, and material state. If return_intermediate is True, each
            tensor includes a time-increment dimension as the leading axis.

        Raises:
            RuntimeError: If Newton iterations do not converge for a time step.
        """

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

        if aggregate_integration_points:
            out_grad = out_grad.mean(dim=1)
            out_flux = out_flux.mean(dim=1)
            out_state = out_state.mean(dim=1)

        out_flux = out_flux.squeeze()
        out_grad = out_grad.squeeze()

        if return_intermediate:
            # Return all intermediate values
            return u, f, out_flux, out_grad, out_state
        else:
            # Return only the final values
            return u[-1], f[-1], out_flux[-1], out_grad[-1], out_state[-1]
