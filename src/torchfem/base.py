from abc import ABC, abstractmethod
from typing import Literal, Tuple

import torch
from torch import Tensor

from .elements import Element
from .materials import Material
from .sparse import CachedSolve, sparse_solve


class FEMGeneral(ABC):
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

        # Initialize load variables
        self._forces = torch.zeros(self.n_nod, self.n_dof_per_node)
        self._displacements = torch.zeros(self.n_nod, self.n_dof_per_node)
        self._constraints = torch.zeros(
            self.n_nod, self.n_dof_per_node, dtype=torch.bool
        )

        # Compute mapping from local to global indices
        idx = (self.n_dof_per_node * self.elements).unsqueeze(-1) + torch.arange(
            self.n_dof_per_node
        )
        self.idx = idx.reshape(self.n_elem, -1).to(torch.int32)

        # Vectorize material
        if material.is_vectorized:
            self.material = material
        else:
            self.material = material.vectorize(self.n_elem)

        # Initialize types
        self.n_stress: int
        self.n_int: int
        self.ext_strain: Tensor
        self.etype: Element
        self.char_lengths: Tensor

        # Cached solve for sparse linear systems
        self.cached_solve = CachedSolve()

    @property
    def forces(self) -> Tensor:
        return self._forces

    @forces.setter
    def forces(self, value: Tensor):
        if not value.shape == (self.n_nod, self.n_dof_per_node):
            raise ValueError("Forces must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Forces must be a floating-point tensor.")
        self._forces = value.to(self.nodes.device)

    @property
    def displacements(self) -> Tensor:
        return self._displacements

    @displacements.setter
    def displacements(self, value: Tensor):
        if not value.shape == (self.n_nod, self.n_dof_per_node):
            raise ValueError("Displacements must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Displacements must be a floating-point tensor.")
        self._displacements = value.to(self.nodes.device)

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

    @abstractmethod
    def eval_shape_functions(
        self, xi: Tensor, u: Tensor | float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor]:
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

    def compute_B(self) -> Tensor:
        """Null space representing rigid body modes."""
        if self.n_dim == 3:
            B = torch.zeros((self.n_dim * self.n_nod, 6))
            B[0::3, 0] = 1
            B[1::3, 1] = 1
            B[2::3, 2] = 1
            B[1::3, 3] = -self.nodes[:, 2]
            B[2::3, 3] = self.nodes[:, 1]
            B[0::3, 4] = self.nodes[:, 2]
            B[2::3, 4] = -self.nodes[:, 0]
            B[0::3, 5] = -self.nodes[:, 1]
            B[1::3, 5] = self.nodes[:, 0]
        elif self.n_dim == 2:
            B = torch.zeros((self.n_dim * self.n_nod, 3))
            B[0::2, 0] = 1
            B[1::2, 1] = 1
            B[1::2, 2] = -self.nodes[:, 0]
            B[0::2, 2] = self.nodes[:, 1]
        elif self.n_dim == 1:
            B = torch.zeros((self.n_dim * self.n_nod, 1))
            B[0::1, 0] = 1
        else:
            raise ValueError("Unsupported spatial dimension.")
        return B

    @abstractmethod
    def k0(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def integrate_material(
        self,
        u: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        n: int,
        i: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def integrate_field(self, field: Tensor | None = None) -> Tensor:
        """Integrate scalar field over elements."""

        # Default field is ones to integrate volume
        if field is None:
            field = torch.ones(self.n_nod)

        # Integrate
        res = torch.zeros(len(self.elements))
        for w, xi in zip(self.etype.iweights, self.etype.ipoints):
            N, B, detJ = self.eval_shape_functions(xi)
            f = field[self.elements, None].squeeze() @ N
            res += w * f * detJ
        return res

    def assemble_stiffness(self, k: Tensor, con: Tensor) -> Tensor:
        """Assemble global stiffness matrix."""

        # Initialize sparse matrix
        size = (self.n_dofs, self.n_dofs)
        K = torch.empty(size, layout=torch.sparse_coo)

        # Build matrix in chunks to prevent excessive memory usage
        chunks = 4
        for idx, k_chunk in zip(torch.chunk(self.idx, chunks), torch.chunk(k, chunks)):
            # Ravel indices and values
            chunk_size = idx.shape[0]
            col = idx.unsqueeze(1).expand(chunk_size, self.idx.shape[1], -1).ravel()
            row = idx.unsqueeze(-1).expand(chunk_size, -1, self.idx.shape[1]).ravel()
            indices = torch.stack([row, col], dim=0)
            values = k_chunk.ravel()

            # Eliminate and replace constrained dofs
            ci = torch.isin(idx, con)
            mask_col = ci.unsqueeze(1).expand(chunk_size, self.idx.shape[1], -1).ravel()
            mask_row = (
                ci.unsqueeze(-1).expand(chunk_size, -1, self.idx.shape[1]).ravel()
            )
            mask = ~(mask_col | mask_row)
            diag_index = torch.stack((con, con), dim=0)
            diag_value = torch.ones_like(con, dtype=k.dtype)

            # Concatenate
            indices = torch.cat((indices[:, mask], diag_index), dim=1)
            values = torch.cat((values[mask], diag_value), dim=0)

            K += torch.sparse_coo_tensor(indices, values, size=size).coalesce()

        return K.coalesce()

    def assemble_force(self, f: Tensor) -> Tensor:
        """Assemble global force vector."""

        # Initialize force vector
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
                internal forces, stress, deformation gradient, and material state.

        """
        # Number of increments
        N = len(increments)

        # Null space rigid body modes for AMG preconditioner
        B = self.compute_B()

        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()

        # Initialize variables to be computed
        u = torch.zeros(
            N, self.n_nod, self.n_dof_per_node
        )  # generalized solution variable
        f = torch.zeros(N, self.n_nod, self.n_dof_per_node)  # generalized forces
        stress = torch.zeros(
            N, self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim
        )
        # generalized gradient of the solution variable
        defgrad = torch.zeros(
            N, self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim
        )
        # initialize with deformation gradient for mechanics, dirty hack for generality
        if self.n_dim == self.n_dof_per_node and self.physics_type == "mechanics":
            defgrad[:, :, :, :, :] = torch.eye(self.n_dim)
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)

        if verbose and u.dtype != torch.float64:
            print(
                "WARNING: Detected single precision floating points. It is highly "
                "recommended to use torch-fem with double precision by setting "
                "'torch.set_default_dtype(torch.float64)'."
            )

        # Initialize global stiffness matrix
        self.K = torch.empty(0)

        # Initialize displacement increment
        du = torch.zeros(self.n_nod, self.n_dof_per_node).ravel()

        # Incremental loading
        for n in range(1, N):
            # Increment size
            inc = increments[n] - increments[n - 1]

            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            de0 = inc * self.ext_strain

            # Newton-Raphson iterations
            for i in range(max_iter):
                du[con] = DU[con]

                # Element-wise integration
                k, f_i = self.integrate_material(
                    u, defgrad, stress, state, n, i, du, de0, nlgeom
                )

                # Assemble global stiffness matrix and internal force vector (if needed)
                if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                    self.K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_i)

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
            defgrad = defgrad.mean(dim=1)
            stress = stress.mean(dim=1)
            state = state.mean(dim=1)

        # Squeeze outputs
        stress = stress.squeeze()
        defgrad = defgrad.squeeze()

        if return_intermediate:
            # Return all intermediate values
            return u, f, stress, defgrad, state
        else:
            # Return only the final values
            return u[-1], f[-1], stress[-1], defgrad[-1], state[-1]


class FEM(FEMGeneral):
    physics_type = "mechanics"

    def k0(self) -> Tensor:
        """Compute element stiffness matrix for zero strain."""
        u = torch.zeros(self.n_nod, self.n_dof_per_node)
        F = torch.zeros(2, self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim)
        F[:, :, :, :, :] = torch.eye(self.n_dim)
        s = torch.zeros(2, self.n_int, self.n_elem, self.n_dof_per_node, self.n_dim)
        a = torch.zeros(2, self.n_int, self.n_elem, self.material.n_state)
        du = torch.zeros(self.n_nod, self.n_dof_per_node)
        de0 = torch.zeros(self.n_elem, self.n_dof_per_node, self.n_dim)
        self.K = torch.empty(0)
        k, _ = self.integrate_material(u, F, s, a, 1, 0, du, de0, False)
        return k

    def integrate_material(
        self,
        u: Tensor,
        F: Tensor,
        stress: Tensor,
        state: Tensor,
        n: int,
        iter: int,
        du: Tensor,
        de0: Tensor,
        nlgeom: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix."""
        # Compute updated configuration
        u_trial = u[n - 1] + du.view((-1, self.n_dof_per_node))

        # Reshape displacement increment
        du = du.view(-1, self.n_dof_per_node)[self.elements].reshape(
            self.n_elem, -1, self.n_dof_per_node
        )

        # Initialize nodal force and stiffness
        N_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dof_per_node * N_nod)
        k = torch.zeros(
            (self.n_elem, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod)
        )

        for i, (w, xi) in enumerate(zip(self.etype.iweights, self.etype.ipoints)):
            # Compute gradient operators
            _, B0, detJ0 = self.eval_shape_functions(xi)
            if nlgeom:
                # Compute updated gradient operators in deformed configuration
                _, B, detJ = self.eval_shape_functions(xi, u_trial)
            else:
                # Use initial gradient operators
                B = B0
                detJ = detJ0

            # Compute displacement gradient increment
            H_inc = torch.einsum("...ij,...jk->...ik", B0, du)

            # Update deformation gradient
            F[n, i] = F[n - 1, i] + H_inc

            # Evaluate material response
            stress[n, i], state[n, i], ddsdde = self.material.step(
                H_inc,
                F[n - 1, i],
                stress[n - 1, i],
                state[n - 1, i],
                de0,
                self.char_lengths,
                iter,
            )

            # Compute element internal forces
            force_contrib = self.compute_f(detJ, B, stress[n, i].clone())
            f += w * force_contrib.reshape(-1, self.n_dof_per_node * N_nod)

            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                # Material stiffness
                BCB = torch.einsum("...ijpq,...qk,...il->...ljkp", ddsdde, B, B)
                BCB = BCB.reshape(
                    -1, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod
                )
                k += w * self.compute_k(detJ, BCB)
            if nlgeom:
                # Geometric stiffness
                BSB = torch.einsum(
                    "...iq,...qk,...il->...lk", stress[n, i].clone(), B, B
                )
                zeros = torch.zeros_like(BSB)
                kg = torch.stack([BSB] + (self.n_dof_per_node - 1) * [zeros], dim=-1)
                kg = kg.reshape(-1, N_nod, self.n_dim * N_nod).unsqueeze(-2)
                zeros = torch.zeros_like(kg)
                kg = torch.stack([kg] + (self.n_dof_per_node - 1) * [zeros], dim=-2)
                kg = kg.reshape(
                    -1, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod
                )
                k += w * self.compute_k(detJ, kg)

        return k, f


class FEMHEat(FEM):
    physics_type = "thermal"

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
        temp: Tensor,
        temp_grad: Tensor,
        heat_flux: Tensor,
        state: Tensor,
        n: int,
        iter: int,
        dtemp: Tensor,
        dtemp_grad0: Tensor,
        nlgeom: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Perform numerical integrations for element stiffness matrix."""

        # Reshape temperature increment
        dtemp = dtemp.view(-1, self.n_dof_per_node)[self.elements].reshape(
            self.n_elem, -1, self.n_dof_per_node
        )

        # Initialize nodal heat fluxes and conductivity matrix
        N_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dof_per_node * N_nod)
        k = torch.zeros(
            (self.n_elem, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod)
        )

        for i, (w, xi) in enumerate(zip(self.etype.iweights, self.etype.ipoints)):
            # Compute gradient operators
            _, B0, detJ0 = self.eval_shape_functions(xi)

            B = B0
            detJ = detJ0

            # Compute temperature gradient increment
            temp_grad_inc = torch.einsum("...ij,...jk->...ki", B0, dtemp)

            # Update deformation gradient
            temp_grad[n, i] = temp_grad[n - 1, i] + temp_grad_inc

            # Evaluate material response
            heat_flux[n, i], state[n, i], ddheat_flux_ddtemp_grad = self.material.step(
                temp_grad_inc,
                temp_grad[n - 1, i],
                heat_flux[n - 1, i],
                state[n - 1, i],
                dtemp_grad0,
                self.char_lengths,
                iter,
            )

            # Compute element internal forces
            force_contrib = self.compute_f(detJ, B, heat_flux[n, i].clone())
            f += w * force_contrib.reshape(-1, self.n_dof_per_node * N_nod)

            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0:
                # Material stiffness
                BCB = torch.einsum(
                    "...ij,...iN,...jM->...NM", ddheat_flux_ddtemp_grad, B, B
                )
                BCB = BCB.reshape(
                    -1, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod
                )
                k += w * self.compute_k(detJ, BCB)

        return k, f
