import torch
from tqdm import tqdm

from .elements import Hexa1, Hexa2, Tetra1, Tetra2
from .sparse import sparse_solve


class Solid:
    def __init__(self, nodes: torch.Tensor, elements: torch.Tensor, material):
        """Initialize the solid FEM problem."""

        # Store nodes and elements
        self.nodes = nodes
        self.elements = elements

        # Compute problem size
        self.n_dofs = torch.numel(self.nodes)
        self.n_nod = len(self.nodes)
        self.n_elem = len(self.elements)

        # Initialize load variables
        self.forces = torch.zeros_like(nodes)
        self.displacements = torch.zeros_like(nodes)
        self.constraints = torch.zeros_like(nodes, dtype=bool)
        self.ext_strain = torch.zeros(self.n_elem, 6)

        # Set element type depending on number of nodes per element
        if len(elements[0]) == 4:
            self.etype = Tetra1()
        elif len(elements[0]) == 8:
            self.etype = Hexa1()
        elif len(elements[0]) == 10:
            self.etype = Tetra2()
        elif len(elements[0]) == 20:
            self.etype = Hexa2()

        # Integration weights and points
        self.w = self.etype.iweights()
        self.xi = self.etype.ipoints()
        self.n_int = len(self.xi)

        # Vectorize material
        self.material = material.vectorize(self.n_int, self.n_elem)

        # Compute mapping from local to global indices (hard to read, but fast)
        N = self.n_elem
        self.idx = ((3 * self.elements).unsqueeze(-1) + torch.arange(3)).reshape(N, -1)
        idx1 = self.idx.unsqueeze(1).expand(N, self.idx.shape[1], -1)
        idx2 = self.idx.unsqueeze(-1).expand(N, -1, self.idx.shape[1])
        self.indices = torch.stack([idx1, idx2], dim=0).reshape((2, -1))

    def D(self, B):
        """Element gradient operator"""
        zeros = torch.zeros(self.n_int, self.n_elem, self.etype.nodes)
        shape = [self.n_int, self.n_elem, -1]
        D0 = torch.stack([B[:, :, 0, :], zeros, zeros], dim=-1).reshape(shape)
        D1 = torch.stack([zeros, B[:, :, 1, :], zeros], dim=-1).reshape(shape)
        D2 = torch.stack([zeros, zeros, B[:, :, 2, :]], dim=-1).reshape(shape)
        D3 = torch.stack([zeros, B[:, :, 2, :], B[:, :, 1, :]], dim=-1).reshape(shape)
        D4 = torch.stack([B[:, :, 2, :], zeros, B[:, :, 0, :]], dim=-1).reshape(shape)
        D5 = torch.stack([B[:, :, 1, :], B[:, :, 0, :], zeros], dim=-1).reshape(shape)
        return torch.stack([D0, D1, D2, D3, D4, D5], dim=2)

    def k0(self):
        """Compute element stiffness matrix for zero strain."""
        de = torch.zeros(self.n_int, self.n_elem, 6)
        ds = torch.zeros(self.n_int, self.n_elem, 6)
        da = torch.zeros(self.n_int, self.n_elem, self.material.n_state)
        du = torch.zeros_like(self.nodes)
        dde0 = torch.zeros(self.n_elem, 6)
        k, _, _, _, _ = self.integrate(de, ds, da, du, dde0)
        return k

    def integrate(self, de, ds, da, du, dde0):
        """Perform numerical integrations for element stiffness matrix."""
        # Reshape variables
        nodes = self.nodes[self.elements, :]
        du = du[self.elements, :].reshape(self.n_elem, -1)

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
        f = torch.einsum("i,ij,ijkl,ijk->ijl", self.w, detJ, D, sigma).sum(dim=0)

        # Compute element stiffness matrix
        CD = torch.matmul(ddsdde, D)
        DCD = torch.matmul(CD.transpose(2, 3), D)
        k = torch.einsum("i,ij,ijkl->ijkl", self.w, detJ, DCD).sum(dim=0)

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
        epsilon = torch.zeros(N, self.n_int, self.n_elem, 6)
        sigma = torch.zeros(N, self.n_int, self.n_elem, 6)
        state = torch.zeros(N, self.n_int, self.n_elem, self.material.n_state)
        f = torch.zeros(N, self.n_nod, 3)
        u = torch.zeros(N, self.n_nod, 3)

        # Initialize displacement increment
        du = torch.zeros_like(self.nodes).ravel()

        # Incremental loading
        for i in tqdm(range(1, len(increments)), disable=not verbose, desc="Increment"):
            # Increment size
            inc = increments[i] - increments[i - 1]

            # Load increment
            F_ext = inc * self.forces.ravel()
            DU = inc * self.displacements.clone().ravel()
            DE = inc * self.ext_strain

            # Newton-Raphson iterations
            for j in tqdm(
                range(max_iter),
                disable=not verbose,
                desc="Newton Iteration",
                leave=False,
            ):
                du[con] = DU[con]

                # Element-wise integration
                k, f_int, epsilon_new, sigma_new, state_new = self.integrate(
                    epsilon[i - 1], sigma[i - 1], state[i - 1], du.reshape((-1, 3)), DE
                )

                # Assemble global stiffness matrix and internal force vector
                K = self.assemble_stiffness(k, con)
                F_int = self.assemble_force(f_int)

                # Compute residual
                residual = F_int - F_ext
                residual[con] = 0.0
                res_norm = torch.max(residual) / torch.max(F_int)
                if res_norm < tol:
                    break

                # Solve for displacement increment
                du -= sparse_solve(K, residual)

            # Update increment
            epsilon[i] = epsilon_new
            sigma[i] = sigma_new
            state[i] = state_new
            f[i] = F_int.reshape((-1, 3))
            u[i] += du.reshape((-1, 3))

        if aggregate_integration_points:
            epsilon = epsilon.mean(dim=1)
            sigma = sigma.mean(dim=1)
            state = state.mean(dim=1)

        if return_intermediate:
            return u, f, sigma, epsilon, state
        else:
            return u[-1], f[-1], sigma[-1], epsilon[-1], state[-1]

    @torch.no_grad()
    def plot(
        self,
        u=0.0,
        node_property=None,
        element_property=None,
        show_edges=True,
        show_undeformed=False,
        contour=None,
        cmap="viridis",
    ):
        try:
            import pyvista
        except ImportError:
            raise Exception("Plotting 3D requires pyvista.")

        pyvista.set_plot_theme("document")
        pl = pyvista.Plotter()
        pl.enable_anti_aliasing("ssaa")

        # VTK cell types
        if isinstance(self.etype, Tetra1):
            cell_types = self.n_elem * [pyvista.CellType.TETRA]
        elif isinstance(self.etype, Tetra2):
            cell_types = self.n_elem * [pyvista.CellType.QUADRATIC_TETRA]
        elif isinstance(self.etype, Hexa1):
            cell_types = self.n_elem * [pyvista.CellType.HEXAHEDRON]
        elif isinstance(self.etype, Hexa2):
            cell_types = self.n_elem * [pyvista.CellType.QUADRATIC_HEXAHEDRON]

        # VTK element list
        elements = []
        for element in self.elements:
            elements += [len(element), *element]

        # Deformed node positions
        pos = self.nodes + u

        # Create unstructured mesh
        mesh = pyvista.UnstructuredGrid(elements, cell_types, pos.tolist())

        # Plot node properties
        if node_property:
            for key, val in node_property.items():
                mesh.point_data[key] = val

        # Plot cell properties
        if element_property:
            for key, val in element_property.items():
                mesh.cell_data[key] = val

        if contour:
            mesh = mesh.cell_data_to_point_data()
            mesh = mesh.contour(contour)

        if show_edges:
            if isinstance(self.etype, Tetra2) or isinstance(self.etype, Hexa2):
                # Trick to plot edges for quadratic elements
                # See: https://github.com/pyvista/pyvista/discussions/5777
                surface = mesh.separate_cells().extract_surface(nonlinear_subdivision=4)
                edges = surface.extract_feature_edges()
                pl.add_mesh(surface, cmap=cmap)
                actor = pl.add_mesh(edges, style="wireframe", color="black")
                actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            else:
                pl.add_mesh(mesh, cmap=cmap, show_edges=True)
        else:
            pl.add_mesh(mesh, cmap=cmap)

        if show_undeformed:
            undefo = pyvista.UnstructuredGrid(elements, cell_types, self.nodes.tolist())
            edges = (
                undefo.separate_cells()
                .extract_surface(nonlinear_subdivision=4)
                .extract_feature_edges()
            )
            pl.add_mesh(edges, style="wireframe", color="grey")

        pl.show(jupyter_backend="client")
