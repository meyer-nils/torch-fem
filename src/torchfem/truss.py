from typing import Literal, Tuple

import matplotlib.pyplot as plt
import pyvista
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from torch import Tensor

from .base import Mechanics
from .elements import Bar1, Bar2
from .materials import Material
from .sparse import CachedSolve, sparse_solve


class Truss(Mechanics):
    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize a truss FEM problem."""
        super().__init__(nodes, elements, material)

        # Set up areas
        self.areas = torch.ones((len(elements)))

        # Element type
        if len(elements[0]) == 2:
            self.etype = Bar1
        elif len(elements[0]) == 3:
            self.etype = Bar2
        else:
            raise ValueError("Element type not supported.")

        # Initialize characteristic lengths
        start_nodes = self.nodes[self.elements[:, 0]]
        end_nodes = self.nodes[self.elements[:, 1]]
        self.char_lengths = torch.linalg.norm(end_nodes - start_nodes, dim=-1)

        # Set element type specific sizes
        self.n_stress = 1
        self.n_int = len(self.etype.iweights)

    def __repr__(self) -> str:
        etype = self.etype.__class__.__name__
        return f"<torch-fem truss ({self.n_nod} nodes, {self.n_elem} {etype} elements)>"

    @property
    def external_gradient(self) -> Tensor:
        return torch.zeros(self.n_elem, 1, 1)

    def eval_shape_functions(
        self, xi: Tensor, u: Tensor | float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gradient operator at integration points xi."""
        nodes = self.nodes + u
        nodes = nodes[self.elements, :]
        # Direction of the element
        dx = nodes[:, 1] - nodes[:, 0]
        # Length of the element
        l0 = torch.linalg.norm(dx, dim=-1)
        # Cosine and sine of the element
        cs = dx / l0[:, None]

        J = 0.5 * torch.linalg.norm(dx, dim=1)[:, None, None]
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")

        b = self.etype.B(xi)
        B = torch.einsum("jkl,lm->jkm", torch.linalg.inv(J), b)
        B = torch.einsum("ijk,il->ijkl", B, cs).reshape(self.n_elem, -1)[:, None, :]

        return self.etype.N(xi), B, detJ

    def compute_k(self, detJ: Tensor, BCB: Tensor):
        """Element stiffness matrix."""
        return torch.einsum("...,...,...kl->...kl", self.areas, detJ, BCB)

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor):
        """Element internal force vector."""
        return torch.einsum("...,...,...ik,...ij->...kj", self.areas, detJ, B, S)

    def plot(self, u: float | Tensor = 0.0, **kwargs):
        if self.n_dim == 2:
            self.plot2d(u=u, **kwargs)
        elif self.n_dim == 3:
            self.plot3d(u=u, **kwargs)

    @torch.no_grad()
    def plot2d(
        self,
        u: float | Tensor = 0.0,
        element_property: Tensor | None = None,
        node_labels: bool = True,
        show_thickness: bool = False,
        thickness_threshold: float = 0.0,
        default_color: str = "black",
        cmap: str = "viridis",
        title: str | None = None,
        axes: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        ax: Axes | None = None,
    ):
        # Set figure size
        if ax is None:
            _, ax = plt.subplots()

        # Line widths from areas
        if show_thickness:
            a_max = torch.max(self.areas)
            linewidth = 8.0 * self.areas / a_max
        else:
            linewidth = 2.0 * torch.ones(self.n_elem)
            linewidth[self.areas < thickness_threshold] = 0.0

        # Line color from stress (if present)
        if element_property is not None:
            cm = plt.get_cmap(cmap)
            if vmin is None:
                vmin = min(float(element_property.min()), 0.0)
            if vmax is None:
                vmax = max(float(element_property.max()), 0.0)
            color = cm((element_property - vmin) / (vmax - vmin))
            sm = plt.cm.ScalarMappable(cmap=cm, norm=Normalize(vmin=vmin, vmax=vmax))
            plt.colorbar(sm, ax=ax, shrink=0.5)
        else:
            color = self.n_elem * [default_color]

        # Nodes
        pos = self.nodes + u
        ax.scatter(pos[:, 0], pos[:, 1], color=default_color, marker="o", zorder=10)
        if node_labels:
            for i, node in enumerate(pos):
                ax.annotate(
                    str(i),
                    (node[0].item() + 0.01, node[1].item() + 0.1),
                    color=default_color,
                )

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min())

        # Bars
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            x = [pos[n1][0], pos[n2][0]]
            y = [pos[n1][1], pos[n2][1]]
            ax.plot(x, y, linewidth=linewidth[j], c=color[j])

        # Forces
        for i, force in enumerate(self.forces):
            if torch.norm(force) > 0.0:
                s = 0.05 * size / torch.linalg.norm(force)  # scale
                plt.arrow(
                    float(pos[i][0]),
                    float(pos[i][1]),
                    s * force[0],
                    s * force[1],
                    width=0.05,
                    facecolor="gray",
                )

        # Constraints
        for i, constraint in enumerate(self.constraints):
            if constraint[0]:
                ax.plot(pos[i][0] - 0.1, pos[i][1], ">", color="gray")
            if constraint[1]:
                ax.plot(pos[i][0], pos[i][1] - 0.1, "^", color="gray")

        # Adjustments
        nmin = pos.min(dim=0).values
        nmax = pos.max(dim=0).values
        ax.set(
            xlim=(float(nmin[0]) - 0.5, float(nmax[0]) + 0.5),
            ylim=(float(nmin[1]) - 0.5, float(nmax[1]) + 0.5),
        )

        if title:
            ax.set_title(title)

        ax.set_aspect("equal", adjustable="box")
        if not axes:
            ax.set_axis_off()

    @torch.no_grad()
    def plot3d(
        self,
        u: float | Tensor = 0.0,
        element_property: dict[str, Tensor] | None = None,
        force_size_factor: float = 0.5,
        constraint_size_factor: float = 0.1,
        cmap: str = "viridis",
    ):
        pyvista.set_plot_theme("document")
        pyvista.set_jupyter_backend("client")
        pl = pyvista.Plotter()
        pl.enable_anti_aliasing("ssaa")

        # Nodes
        pos = self.nodes + u

        # Bounding box
        size = torch.linalg.norm(pos.max() - pos.min()).item()

        # Radii
        radii = torch.sqrt(self.areas / torch.pi).numpy()

        # Elements
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            tube = pyvista.Tube(
                pointa=pos[n1].numpy(), pointb=pos[n2].numpy(), radius=radii[j]
            )
            if element_property is not None:
                for key, value in element_property.items():
                    value = element_property[key].squeeze()
                    tube.cell_data[key] = value[j].numpy()
                pl.add_mesh(tube, scalars=key, cmap=cmap)
            else:
                pl.add_mesh(tube, color="gray")

        # Forces
        force_centers = []
        force_directions = []
        for i, force in enumerate(self.forces):
            if torch.norm(force) > 0.0:
                force_centers.append(pos[i])
                force_directions.append(force / torch.linalg.norm(force))
        pl.add_arrows(
            torch.stack(force_centers).numpy(),
            torch.stack(force_directions).numpy(),
            mag=force_size_factor * size,
            color="gray",
        )

        # Constraints
        for i, constraint in enumerate(self.constraints):
            if constraint.any():
                sphere = pyvista.Sphere(
                    radius=constraint_size_factor * size, center=pos[i].numpy()
                )
                pl.add_mesh(sphere, color="gray")

        pl.show(jupyter_backend="html")

    def k0(self) -> Tensor:
        """Compute element stiffness matrix for zero strain."""
        u = torch.zeros_like(self.nodes)
        F = torch.zeros(2, self.n_int, self.n_elem, self.n_stress, self.n_stress)
        F[:, :, :, :, :] = torch.eye(self.n_stress)
        s = torch.zeros(2, self.n_int, self.n_elem, self.n_stress, self.n_stress)
        a = torch.zeros(2, self.n_int, self.n_elem, self.material.n_state)
        du = torch.zeros_like(self.nodes)
        de0 = torch.zeros(self.n_elem, self.n_stress, self.n_stress)
        self.K = torch.empty(0)
        k, _ = self.integrate_material(u, F, s, a, 1, 0, du, de0, False)
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
        """Perform numerical integrations for element stiffness matrix."""

        # Interpretation of variables
        F = grad
        stress = flux

        # Compute updated configuration
        u_trial = u[n - 1] + du.view((-1, self.n_dim))

        # Reshape displacement increment
        du = du.view(-1, self.n_dim)[self.elements].reshape(
            self.n_elem, -1, self.n_stress
        )

        # Initialize nodal force and stiffness
        N_nod = self.etype.nodes
        f = torch.zeros(self.n_elem, self.n_dim * N_nod)
        k = torch.zeros((self.n_elem, self.n_dim * N_nod, self.n_dim * N_nod))

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
            f += w * force_contrib.reshape(-1, self.n_dim * N_nod)

            # Compute element stiffness matrix
            if self.K.numel() == 0 or not self.material.n_state == 0 or nlgeom:
                # Material stiffness
                BCB = torch.einsum("...ijpq,...qk,...il->...ljkp", ddsdde, B, B)
                BCB = BCB.reshape(-1, self.n_dim * N_nod, self.n_dim * N_nod)
                k += w * self.compute_k(detJ, BCB)
            if nlgeom:
                # Geometric stiffness
                BSB = torch.einsum(
                    "...iq,...qk,...il->...lk", stress[n, i].clone(), B, B
                )
                zeros = torch.zeros_like(BSB)
                kg = torch.stack([BSB] + (self.n_dim - 1) * [zeros], dim=-1)
                kg = kg.reshape(-1, N_nod, self.n_dim * N_nod).unsqueeze(-2)
                zeros = torch.zeros_like(kg)
                kg = torch.stack([kg] + (self.n_dim - 1) * [zeros], dim=-2)
                kg = kg.reshape(-1, self.n_dim * N_nod, self.n_dim * N_nod)
                k += w * self.compute_k(detJ, kg)

        return k, f

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
        u = torch.zeros(N, self.n_nod, self.n_dim)
        f = torch.zeros(N, self.n_nod, self.n_dim)
        stress = torch.zeros(N, self.n_int, self.n_elem, self.n_stress, self.n_stress)
        defgrad = torch.zeros(N, self.n_int, self.n_elem, self.n_stress, self.n_stress)
        defgrad[:, :, :, :, :] = torch.eye(self.n_stress)
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
        du = torch.zeros_like(self.nodes).ravel()

        # Incremental loading
        for n in range(1, N):
            # Increment size
            inc = increments[n] - increments[n - 1]

            # Load increment
            F_ext = increments[n] * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            de0 = inc * self.external_gradient

            # Newton-Raphson iterations
            for i in range(max_iter):
                du[con] = DU[con]

                # Element-wise integration
                k, f_i = self.integrate_material(
                    u, defgrad, stress, state, n, i, du, de0, nlgeom
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
            f[n] = F_int.reshape((-1, self.n_dim))
            u[n] = u[n - 1] + du.reshape((-1, self.n_dim))

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
            return u[-1], f[-1], stress[-1], defgrad[-1], state[-1]
