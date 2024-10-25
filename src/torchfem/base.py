from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from .materials import Material
from .sparse import sparse_solve


class FEM(ABC):
    def __init__(self, nodes: torch.Tensor, elements: torch.Tensor, material: Material):
        """Initialize a general FEM problem."""

        # Store nodes and elements
        self.nodes = nodes
        self.elements = elements

        # Compute problem size
        self.n_dofs = torch.numel(self.nodes)
        self.n_nod = nodes.shape[0]
        self.n_dim = nodes.shape[1]
        self.n_elem = len(self.elements)

        if self.n_dim == 3:
            self.n_strains = 6
        elif self.n_dim == 2:
            self.n_strains = 3
        else:
            raise Exception("Only 2D and 3D problems are supported.")

        # Initialize load variables
        self.forces = torch.zeros_like(nodes)
        self.displacements = torch.zeros_like(nodes)
        self.constraints = torch.zeros_like(nodes, dtype=bool)
        self.ext_strain = torch.zeros(self.n_elem, self.n_strains)

        # Compute mapping from local to global indices (hard to read, but fast)
        self.idx = (
            (self.n_dim * self.elements).unsqueeze(-1) + torch.arange(self.n_dim)
        ).reshape(self.n_elem, -1)
        idx1 = self.idx.unsqueeze(1).expand(self.n_elem, self.idx.shape[1], -1)
        idx2 = self.idx.unsqueeze(-1).expand(self.n_elem, -1, self.idx.shape[1])
        self.indices = torch.stack([idx1, idx2], dim=0).reshape((2, -1))

    @abstractmethod
    def D(self, B):
        pass

    @abstractmethod
    def k(self, detJ, DCD):
        pass

    @abstractmethod
    def f(self, detJ, D, S):
        pass

    def k0(self):
        """Compute element stiffness matrix for zero strain."""
        de = torch.zeros(self.n_int, self.n_elem, self.n_strains)
        ds = torch.zeros(self.n_int, self.n_elem, self.n_strains)
        da = torch.zeros(self.n_int, self.n_elem, self.material.n_state)
        du = torch.zeros_like(self.nodes)
        dde0 = torch.zeros(self.n_elem, self.n_strains)
        k, _, _, _, _ = self.integrate(de, ds, da, du, dde0)
        return k

    def integrate(self, de, ds, da, du, dde0):
        """Perform numerical integrations for element stiffness matrix."""
        # Reshape variables
        nodes = self.nodes[self.elements, :]
        du = du.reshape((-1, self.n_dim))[self.elements, :].reshape(self.n_elem, -1)

        # Compute gradient operators
        b = self.etype.B(self.xi)
        J = torch.einsum("ijk,mkl->imjl", b, nodes)
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")
        B = torch.einsum("ijkl,ilm->ijkm", torch.linalg.inv(J), b)
        D = self.D(B)

        # Evaluate material response
        dde = torch.einsum("ijkl,jl->ijk", D, du) - dde0
        epsilon, sigma, state, ddsdde = self.material.step(dde, de, ds, da)

        # Compute element internal forces
        f = self.f(detJ, D, sigma)

        # Compute element stiffness matrix
        CD = torch.matmul(ddsdde, D)
        DCD = torch.matmul(CD.transpose(2, 3), D)
        k = self.k(detJ, DCD)

        return k, f, epsilon, sigma, state

    def assemble_stiffness(self, k, con):
        """Assemble global stiffness matrix."""

        # Initialize sparse matrix size
        size = (self.n_dofs, self.n_dofs)

        # Ravel indices and values
        indices = self.indices
        values = k.ravel()

        # Eliminate and replace constrained dofs
        mask = ~(torch.isin(indices[0, :], con) | torch.isin(indices[1, :], con))
        diag_index = torch.stack((con, con), dim=0)
        diag_value = torch.ones_like(con, dtype=k.dtype)

        # Concatenate
        indices = torch.cat((indices[:, mask], diag_index), dim=1)
        values = torch.cat((values[mask], diag_value), dim=0)

        return torch.sparse_coo_tensor(indices, values, size=size).coalesce()

    def assemble_force(self, f):
        """Assemble global force vector."""

        # Initialize force vector
        F = torch.zeros((self.n_dofs))

        # Ravel indices and values
        indices = self.idx.ravel()
        values = f.ravel()

        return F.index_add_(0, indices, values)

    def solve(
        self,
        increments=[0, 1],
        max_iter=10,
        tol=1e-4,
        verbose=False,
        return_intermediate=False,
        aggregate_integration_points=True,
    ):
        """Solve the FEM problem with the Newton-Raphson method."""
        # Number of increments
        N = len(increments)

        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()

        # Initialize variables to be computed
        epsilon = torch.zeros(N, self.n_int, self.n_elem, self.n_strains)
        sigma = torch.zeros(N, self.n_int, self.n_elem, self.n_strains)
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)
        f = torch.zeros(N, self.n_nod, self.n_dim)
        u = torch.zeros(N, self.n_nod, self.n_dim)

        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()

        # Incremental loading
        for i in tqdm(range(1, len(increments)), disable=not verbose, desc="Increment"):
            # Increment size
            inc = increments[i] - increments[i - 1]

            # Load increment
            F_ext = increments[i] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            DE = inc * self.ext_strain

            # Newton-Raphson iterations
            for _ in range(max_iter):
                du[con] = DU[con]

                # Element-wise integration
                k, f_int, epsilon_new, sigma_new, state_new = self.integrate(
                    epsilon[i - 1], sigma[i - 1], state[i - 1], du, DE
                )

                # Assemble global stiffness matrix and internal force vector
                K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_int)

                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = residual.abs().max() / F_int.abs().max()
                if res_norm < tol:
                    break

                # Solve for displacement increment
                du -= sparse_solve(K, residual)

            # Update increment
            epsilon[i] = epsilon_new
            sigma[i] = sigma_new
            state[i] = state_new
            f[i] = F_int.reshape((-1, self.n_dim))
            u[i] = u[i - 1] + du.reshape((-1, self.n_dim))

        # Aggregate integration points as mean
        if aggregate_integration_points:
            epsilon = epsilon.mean(dim=1)
            sigma = sigma.mean(dim=1)
            state = state.mean(dim=1)

        if return_intermediate:
            # Return all intermediate values
            return u, f, sigma, epsilon, state
        else:
            # Return only the final values
            return u[-1], f[-1], sigma[-1], epsilon[-1], state[-1]
