import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cached_property
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor

from .elements import Element
from .materials import Material
from .report import SolveReport, machine
from .sparse import (
    describe_method,
    differentiable_modal_eigsolve,
    differentiable_sparse_solve,
    newton_solve,
    resolve_method,
)


def skew(r: Tensor, dim: int) -> Tensor:
    """Columns of the `theta x r` operator: the basis vector `e_b` crossed into r.

    Padded to three components, so 2D rotates about z alone.
    """
    r = torch.cat([r, torch.zeros(len(r), 3 - dim)], dim=1)
    eye = torch.eye(3).expand(len(r), 3, 3)
    cross = torch.linalg.cross(eye, r[:, None, :].expand(-1, 3, -1), dim=-1)
    return cross.transpose(1, 2)


def near_null_space(nodes: Tensor, n_dof_per_node: int) -> Tensor:
    """The near-null space an algebraic multigrid setup needs for a mesh.

    Rigid body motions where the nodes carry them, and a constant field otherwise,
    as a heat model has. Rotational degrees of freedom enter the rotation modes, so
    a shell gets the same six modes as a solid.
    """
    dim = nodes.shape[1]
    if n_dof_per_node < dim:
        return torch.ones(n_dof_per_node * len(nodes), 1)
    axes = (0, 1, 2) if dim == 3 else (2,)
    modes = torch.zeros(n_dof_per_node * len(nodes), dim + len(axes))
    base = torch.arange(len(nodes)) * n_dof_per_node
    r = skew(nodes, dim)
    for a in range(dim):
        modes[base + a, a] = 1.0
        for b, axis in enumerate(axes):
            modes[base + a, dim + b] = r[:, a, axis]
    if n_dof_per_node > dim:
        for b in range(len(axes)):
            modes[base + dim + b, dim + b] = 1.0
    return modes


class FEM(ABC):
    """Abstract base class for all finite-element models.

    A model is defined by nodal coordinates, an element connectivity, and a
    material. Loads and boundary conditions are set through attributes of the
    concrete model classes, and the quasi-static solution is computed with
    `solve()`.

    Attributes:
        nodes: Nodal coordinates with shape [n_nod, n_dim].
        elements: Element connectivity with shape [n_elem, nodes_per_element].
        material: Vectorized material model (or None for laminate shells).
        constraints: Boolean mask of constrained degrees of freedom with shape
            [n_nod, n_dof_per_node].
        n_nod: Number of nodes.
        n_elem: Number of elements.
        n_dofs: Total number of degrees of freedom.
    """

    supports_nlgeom = True

    def __init__(self, nodes: Tensor, elements: Tensor, material: Material | None):
        """Initialize a finite-element model.

        Args:
            nodes: Nodal coordinates with shape [n_nod, n_dim].
            elements: Connectivity with shape [n_elem, n_nodes_per_element].
            material: Material model. If not vectorized, it is vectorized over
                elements during initialization. May be ``None`` for shells that
                use a laminate section instead.
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
        self.idx = idx.reshape(self.n_elem, -1).to(torch.int32)

        # Sparse assembly maps, built from the node adjacency. The matrix is
        # blocked by node, storing one column index per block rather than per
        # entry, and AmgX aggregates over those blocks. It takes none wider than
        # five, so a node splits into `s` blocks where its degrees of freedom
        # outrun that: a shell's six become two of three. A scalar problem has
        # nothing to block and stays compressed by row, for want of a torch
        # product over a block of one on CUDA.
        ndof = self.n_dof_per_node
        self.block_size = bs = max(d for d in range(1, 6) if ndof % d == 0)
        s = ndof // bs
        sub = torch.arange(s, dtype=torch.int32)
        dof = torch.arange(ndof, dtype=torch.int32)
        nod = torch.arange(self.n_nod)
        el = self.elements.contiguous()
        pair = (el.unsqueeze(-1) << 32) | el.unsqueeze(1)
        loop = (nod << 32) | nod  # a node with itself, so no row lacks a diagonal
        packed = torch.unique(torch.cat([pair.ravel(), loop]))
        deg = torch.bincount(packed >> 32, minlength=self.n_nod)
        node_crow = torch.cat([deg.new_zeros(1), deg.cumsum(0)]).to(torch.int32)
        node_col = (packed % 2**32).to(torch.int32)
        entry = torch.searchsorted(packed, pair, out_int32=True)
        # Where a node holds its own block, so the diagonal can be found again
        within = torch.searchsorted(packed, loop, out_int32=True) - node_crow[:-1]
        del pair, loop, packed

        # Each node row becomes `s` block rows repeating that node's columns
        length = (s * deg).repeat_interleave(s)
        crow = torch.cat([length.new_zeros(1), length.cumsum(0)]).to(torch.int32)
        cols = (node_col[:, None] * s + sub).ravel()
        shift = (s * node_crow[:-1]).repeat_interleave(s) - crow[:-1]
        pos = shift.repeat_interleave(length)
        pos += torch.arange(len(pos), dtype=torch.int32)
        self.crow = crow
        self.col = cols[pos]
        self.n_blocks = self.col.numel()
        # Where the diagonal of each block row sits, by node and block within it
        self.diag = (crow[:-1].view(-1, s) + within[:, None] * s + sub).ravel()
        del cols, shift, pos

        # Entry (p, i, q, j) of an element goes to the block holding row i of
        # node p and column j of node q, at (i, j) within it
        offset = (entry - node_crow[el][..., None])[:, :, None, :, None]
        start = crow[(s * el)[..., None] + dof // bs][..., None, None]
        block = start + offset * s + (dof // bs)
        self.k_map = (
            block * (bs * bs) + (dof % bs)[:, None, None] * bs + dof % bs
        ).ravel()
        # The block row each stored block sits in, which constrained rows need
        self.block_row = torch.repeat_interleave(
            torch.arange(crow.numel() - 1), crow.diff()
        )
        if self.nodes.is_cuda:
            torch.cuda.empty_cache()

        # Vectorize material
        self.material: Material | None
        if material is None or material.is_vectorized:
            self.material = material
        else:
            self.material = material.vectorize(self.n_elem)

    @property
    def n_state(self) -> int:
        """Number of internal state variables per integration point."""
        assert self.material is not None
        return self.material.n_state

    @property
    def volume_scale(self) -> Tensor:
        """Volume per unit element measure, i.e. thickness or cross section area."""
        return torch.ones(self.n_elem, device=self.nodes.device)

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

    def compute_m(self, detJ: Tensor, rho: Tensor) -> Tensor:
        """Compute element mass contribution.

        Args:
            detJ: Jacobian determinant at the current integration point.
            rho: Material density at the current integration point.

        Returns:
            Element mass contribution tensor.
        """
        raise NotImplementedError

    def k0(self) -> Tensor:
        """Compute the element matrix of the reference state.

        Returns:
            Element stiffness for a mechanics model and element conductivity for
            a thermal one, with shape [n_elem, n_dof_elem, n_dof_elem].
        """
        u = torch.zeros(self.n_nod, self.n_dof_per_node)
        grad = torch.zeros(self.n_int, self.n_elem, *self.n_flux)
        grad[:] = self.initial_grad
        flux = torch.zeros(self.n_int, self.n_elem, *self.n_flux)
        state = torch.zeros(self.n_int, self.n_elem, self.n_state)
        du = torch.zeros(self.n_nod, self.n_dof_per_node)
        de0 = torch.zeros(self.n_elem, *self.n_flux)
        self.K = torch.empty(0)
        k, _, _, _, _ = self.integrate_material(u, grad, flux, state, du, de0, 0, False)
        assert k is not None
        return k

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
        compute_stiffness: bool = True,
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
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
            compute_stiffness: If True, compute and return stiffness.

        Returns:
            Tuple of element stiffness (or None when skipped), element
            internal forces, updated gradients, updated fluxes, and updated
            material state.
        """
        raise NotImplementedError

    @property
    def constraints(self) -> Tensor:
        """Boolean mask of constrained DOFs with shape [n_nod, n_dof_per_node]."""
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
        B = torch.linalg.solve(J, b.unsqueeze(-3))
        return self.etype.N(xi), B, detJ

    def near_null_space(self) -> Tensor:
        """The near-null space an algebraic multigrid setup needs for this model."""
        return near_null_space(self.nodes, self.n_dof_per_node)

    @property
    def symmetric_tangent(self) -> bool:
        """Whether the material tangent has major symmetry."""
        return self.material is None or self.material.symmetric_tangent

    def integrate_shape_functions(self) -> Tensor:
        """Integrate each shape function over its element.

        Returns:
            Integrals of the nodal shape functions with shape
            [n_elem, nodes_per_element]. They sum to the element volume or area.
        """
        N, _, detJ = self.eval_shape_functions(self.etype.ipoints)
        return torch.einsum("i,in,ie->en", self.etype.iweights, N, detJ)

    def integrate_field(self, field: Tensor | None = None) -> Tensor:
        """Integrate a nodal scalar field over each element.

        The measure is that of the mesh and excludes `volume_scale`, so a planar
        model integrates over areas and a truss over lengths. Scaling to a volume
        is left to the caller, which keeps this constant where a thickness or a
        cross section is a design variable.

        Args:
            field: Nodal scalar values with shape [n_nod]. If None, integrates
                a unit field and therefore returns the measure of each element.

        Returns:
            Per-element integral values with shape [n_elem].
        """
        w = self.integrate_shape_functions()
        if field is None:
            return w.sum(dim=1)
        return (w * field[self.elements]).sum(dim=1)

    def integrate_mass(self) -> Tensor:
        """Integrate mass matrix.

        Returns:
            Element mass matrix tensor with shape [n_elem, n_dof_elem, n_dof_elem].
        """
        assert self.material is not None
        N_nod = self.etype.nodes
        N_dof = self.n_dof_per_node
        m = torch.zeros((self.n_elem, N_dof * N_nod, N_dof * N_nod))

        N, _, detJ = self.eval_shape_functions(self.etype.ipoints)
        I_dof = torch.eye(N_dof)

        for i, w in enumerate(self.etype.iweights):
            m_i = self.compute_m(detJ[i], self.material.rho)
            m_scalar = torch.einsum("N,M,E->ENM", N[i], N[i], m_i)
            m_block = torch.einsum("Enm,ij->Enimj", m_scalar, I_dof)
            m += w * m_block.reshape(self.n_elem, N_dof * N_nod, N_dof * N_nod)

        return m

    def assemble_matrix(self, k: Tensor, con: Tensor) -> Tensor:
        """Assemble a global sparse matrix from element contributions.

        Args:
            k: Element matrix contributions.
            con: Flattened indices of constrained global degrees of freedom.

        Returns:
            Global matrix, blocked by node where the blocking is worth keeping
            and compressed by row otherwise, with Dirichlet constraints enforced.
        """

        # Fill in stiffness matrix values at appropriate indices
        bs = self.block_size
        val = torch.zeros(self.n_blocks * bs * bs)
        val.index_add_(0, self.k_map, k.ravel())

        constrained = torch.zeros(self.n_dofs, dtype=torch.bool)
        constrained[con] = True
        size = (self.n_dofs, self.n_dofs)

        with torch.sparse.check_sparse_tensor_invariants(False):
            if bs == 1:
                # Apply Dirichlet conditions. CSR stores no row per entry.
                row = torch.repeat_interleave(constrained, self.crow.diff())
                val[row | constrained[self.col]] = 0.0
                val[self.diag[con]] = 1.0
                return torch.sparse_csr_tensor(self.crow, self.col, val, size=size)

            # A constrained degree of freedom clears its row and its column,
            # over the blocks that hold either.
            val = val.view(self.n_blocks, bs, bs)
            mask = constrained.view(-1, bs)
            val[mask[self.block_row]] = 0.0
            val.transpose(1, 2)[mask[self.col]] = 0.0
            block, within = con // bs, con % bs
            val[self.diag[block], within, within] = 1.0
            return torch.sparse_bsr_tensor(self.crow, self.col, val, size=size)

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

    def _scatter(self, conn: Tensor, contrib: Tensor) -> Tensor:
        """Scatter per-node load contributions of elements or facets to the nodes."""
        f = torch.zeros(self.n_nod, contrib.shape[-1], device=self.nodes.device)
        return f.index_add_(0, conn.ravel(), contrib.flatten(0, 1))

    def _boundary_facets(self, mask: Tensor) -> Tensor:
        """Select boundary facets whose nodes all lie in a nodal mask.

        A facet is on the boundary if it belongs to exactly one element, so facets
        inside the selection are dropped rather than loaded twice.
        """
        device = self.elements.device
        table = self.etype.facets.to(device)
        facets = self.elements[:, table].reshape(-1, table.shape[1])
        facets = facets[mask[facets].all(dim=1)]
        _, inv, count = torch.unique(
            facets.sort(dim=1).values, dim=0, return_inverse=True, return_counts=True
        )
        # Keep the first occurrence of each facet, so its node winding is preserved
        first = torch.full((len(count),), len(inv), device=device)
        first.scatter_reduce_(
            0, inv, torch.arange(len(inv), device=device), reduce="amin"
        )
        return facets[first[count == 1]]

    def _integrate_facet_load(
        self, conn: Tensor, ftype: type[Element], load: Tensor
    ) -> Tensor:
        """Consistent nodal loads from a distributed load on the given facets."""
        xi = ftype.ipoints
        # The facet Jacobian is not square, so the measure is sqrt(det(J J^T))
        J = torch.einsum("iaN,eNj->ieaj", ftype.B(xi), self.nodes[conn])
        detJ = torch.sqrt(torch.linalg.det(J @ J.transpose(-1, -2)))
        if load.dim() == 0 and self.n_dof_per_node > 1:
            # A scalar is a pressure acting along the outward normal
            if J.shape[-2] == 2:  # face of a volume element or a shell element
                n = torch.linalg.cross(J[..., 0, :], J[..., 1, :], dim=-1)
            else:  # edge of a planar element
                n = torch.stack([J[..., 0, 1], -J[..., 0, 0]], dim=-1)
            val = load * torch.nn.functional.normalize(n, dim=-1)
        else:
            # A uniform load is broadcast to one value per facet
            per_facet = load if load.dim() == 2 else load.reshape(1, -1)
            val = per_facet.expand(len(conn), -1).expand(len(xi), -1, -1)
        contrib = torch.einsum(
            "i,in,ie,iek->enk", ftype.iweights, ftype.N(xi), detJ, val
        )
        return self._scatter(conn, contrib)

    def integrate_body_load(self, load: float | Tensor) -> Tensor:
        """Consistent nodal loads from a load per unit volume, e.g. gravity.

        Args:
            load: Load per unit volume as a float, with shape [k] if uniform, or with
                shape [n_elem, k] to vary it per element. `k` is the number of loaded
                degrees of freedom per node, i.e. `n_dim` for a force and 1 for a heat
                source.

        Returns:
            Nodal loads with shape [n_nod, k], to be added to `forces` or `heat_flux`.
        """
        load = torch.as_tensor(load, dtype=self.nodes.dtype)
        # A uniform load is broadcast to one value per element
        per_elem = load if load.dim() == 2 else load.reshape(1, -1)
        w = self.integrate_shape_functions() * self.volume_scale[:, None]
        contrib = torch.einsum("en,ek->enk", w, per_elem.expand(self.n_elem, -1))
        return self._scatter(self.elements, contrib)

    def integrate_surface_load(self, mask: Tensor, load: float | Tensor) -> Tensor:
        """Consistent nodal loads from a load per unit area on a surface.

        The surface is made up of the element faces whose nodes all lie in `mask`
        and that are on the boundary of the mesh.

        Args:
            mask: Boolean nodal mask with shape [n_nod] selecting the surface.
            load: Load per unit area. A float is a pressure acting along the outward
                normal, while shape [k] or [n_face, k] is a traction in global
                coordinates.

        Returns:
            Nodal loads with shape [n_nod, k], to be added to `forces` or `heat_flux`.
        """
        if self.etype.iso_dim != 3:
            raise NotImplementedError(
                f"{type(self).__name__} has no surfaces to load. Use "
                "integrate_line_load(...) or integrate_body_load(...) instead."
            )
        return self._integrate_facet_load(
            self._boundary_facets(mask),
            self.etype.facet_type,
            torch.as_tensor(load, dtype=self.nodes.dtype),
        )

    def integrate_line_load(self, mask: Tensor, load: float | Tensor) -> Tensor:
        """Consistent nodal loads from a load per unit length on a line.

        The line is made up of the element edges whose nodes all lie in `mask` and
        that are on the boundary of the mesh.

        Args:
            mask: Boolean nodal mask with shape [n_nod] selecting the line.
            load: Load per unit length with shape [k] or [n_edge, k]. For a planar
                model a float is a pressure acting along the outward normal.

        Returns:
            Nodal loads with shape [n_nod, k], to be added to `forces` or `heat_flux`.
        """
        if self.etype.iso_dim != 2:
            raise NotImplementedError(
                f"{type(self).__name__} has no edges to load. Use "
                "integrate_surface_load(...) or integrate_body_load(...) instead."
            )
        load = torch.as_tensor(load, dtype=self.nodes.dtype)
        if load.dim() == 0 and self.n_dim == 3:
            raise ValueError(
                "A line in 3D has no unique normal, so a scalar load is ambiguous. "
                "Pass a load vector instead."
            )
        return self._integrate_facet_load(
            self._boundary_facets(mask), self.etype.facet_type, load
        )

    def _report(
        self,
        verbose: bool,
        title: str,
        dtype: torch.dtype,
        method: str,
        preconditioner: str | None,
        device: str | None,
        newton: str,
        **kwargs: str,
    ) -> SolveReport | None:
        """Open a report on this model and its linear solver, if verbose."""
        if not verbose:
            return None
        device = device or self.nodes.device.type
        header = {
            "model": f"{type(self).__name__} | {self.n_elem:,} elem | "
            f"{self.n_dofs:,} dof | {str(dtype).removeprefix('torch.')}",
            "machine": machine(device),
            "solver": describe_method(method, device, preconditioner),
            "newton": newton,
        }
        if dtype != torch.float64:
            header["warning"] = (
                "⚠ single precision, prefer torch.set_default_dtype(torch.float64)"
            )
        return SolveReport(f"torch-fem | {title}", header, **kwargs)

    def solve(
        self,
        increments: Tensor | None = None,
        max_iter: int = 10,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        cutback_factor: float = 0.5,
        growth_factor: float = 1.1,
        max_cutbacks: int = 10,
        verbose: bool = False,
        method: Literal["direct", "cg", "bicgstab"] | None = None,
        preconditioner: Literal["amg", "jacobi", "none"] | None = None,
        device: str | None = None,
        return_intermediate: bool = False,
        aggregate_integration_points: bool = True,
        nlgeom: bool = False,
        alpha: float = 0.0,
        differentiable_parameters: Tensor | Iterable[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Solve the quasi-static finite-element problem by load increments.

        Args:
            increments: Load scale factors, typically [0, 1]. They may rise and
                fall, so a load cycle is expressed as a sequence like
                [0, 1, 0]. Results are always returned at exactly these values.
                If a Newton solve does not converge, the increment is subdivided
                internally and retried, and the substep is grown again after
                each success.
            max_iter: Maximum Newton iterations before an increment is cut back.
            rtol: Relative residual tolerance for Newton convergence.
            atol: Absolute residual tolerance for Newton convergence.
            stol: Tolerance used by iterative linear solvers.
            cutback_factor: Factor applied to the substep size after a Newton
                solve failed to converge.
            growth_factor: Factor applied to the substep size after a Newton
                solve converged, capped at the requested increment.
            max_cutbacks: Number of successive cutbacks accepted within an
                increment before the solve is given up.
            verbose: If True, reports the solver configuration and a table of
                per-increment progress, updated in place inside notebooks.
            method: Linear solver method, chosen by size and tangent symmetry
                when omitted.
            preconditioner: Preconditioner for an iterative method, chosen by
                device and available backends when omitted.
            device: Optional device hint for the linear solver backend.
            return_intermediate: If True, returns values for all increments.
            aggregate_integration_points: If True, averages flux, gradient, and
                state over integration points.
            nlgeom: If True, includes geometric nonlinearity.
            alpha: Damping factor for viscous stabilization. Dissipated
                energy is accumulated in `self.stabilization_energy`.
            differentiable_parameters: Explicit parameter(s) to differentiate
                through implicit Newton/sparse solves. Accepts either a single
                tensor or an iterable of tensors.

        Returns:
            Tuple of displacement, internal force, flux, gradient, and material
                state. If return_intermediate is True, each tensor includes an
                increment dimension as the leading axis.
        """
        if nlgeom and not self.supports_nlgeom:
            raise NotImplementedError(
                f"Geometric nonlinearity is not implemented for {type(self).__name__}."
            )

        increments = torch.tensor([0.0, 1.0]) if increments is None else increments

        # Number of increments
        N = len(increments)

        # Mass matrix and dissipated energy for viscous stabilization
        m = self.integrate_mass() if alpha > 0.0 else None
        self.stabilization_energy = torch.zeros(N)

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
        B = self.near_null_space()

        # Indexes of constrained and unconstrained degrees of freedom
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()

        # Initialize variables to be computed
        u = torch.zeros(N, self.n_nod, self.n_dof_per_node)
        f = torch.zeros(N, self.n_nod, self.n_dof_per_node)
        flux = torch.zeros(N, self.n_int, self.n_elem, *self.n_flux)
        grad = torch.zeros(N, self.n_int, self.n_elem, *self.n_flux)
        grad[:, :, :, :, :] = self.initial_grad
        state = torch.zeros(N, self.n_int, self.n_elem, self.n_state)

        newton = (
            f"rtol {rtol:.0e} | atol {atol:.0e} | <={max_iter} it"
            + (" | nlgeom" if nlgeom else "")
            + (f" | stabilized alpha={alpha:g}" if alpha > 0.0 else "")
        )
        # Resolved once here, from what the model knows about its own tangent,
        # rather than per linear solve.
        solve_method = resolve_method(self.n_dofs, method, self.symmetric_tangent)
        report = self._report(
            verbose, "solve", u.dtype, solve_method, preconditioner, device, newton
        )

        # Initialize global stiffness matrix
        self.K = torch.empty(0)

        # Initialize field variable increment
        du = torch.zeros(self.n_nod, self.n_dof_per_node).ravel()

        def make_eval_residual(F_ext, DU, de0, k_visc):
            # Bind this increment's loads at definition time. A plain closure
            # over the loop variables would late-bind them, so the adjoint
            # backward replay would see the last increment's loads.
            def eval_residual(du, i, u_prev, grad_prev, flux_prev, state_prev):
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

                # Viscous stabilization (k is None when self.K is reused as-is)
                if k_visc is not None:
                    du_e = du_bc.view(-1, self.n_dof_per_node)[self.elements].flatten(1)
                    f_i = f_i + torch.einsum("...ij,...j->...i", k_visc, du_e)
                    if k is not None:
                        k = k + k_visc

                # Assemble global stiffness matrix and internal force vector (if needed)
                if k is not None:
                    self.K = self.assemble_matrix(k, con)
                F_int = self.assemble_rhs(f_i)

                # Compute residual
                res = F_int - F_ext
                res[con] = 0.0

                return res, self.K

            return eval_residual

        # Running state, advanced by substeps and stored at requested increments
        u_cur = u[0].clone()
        f_cur = f[0].clone()
        grad_cur = grad[0].clone()
        flux_cur = flux[0].clone()
        state_cur = state[0].clone()
        energy = torch.zeros(())

        # Pseudo time, the fraction of an increment attempted per substep, and
        # the substep the cached viscous tangent belongs to, all carried across
        # increments. A fraction rescales to each increment's own span.
        lam = float(increments[0])
        step_frac = 1.0
        k_step = 0.0

        # Incremental loading with automatic cutback
        for n in range(1, N):
            target = float(increments[n])
            if report is not None:
                report.begin(n, target)

            span = target - lam
            direction = math.copysign(1.0, span)
            step_size = step_frac * abs(span)
            min_step = abs(span) * cutback_factor**max_cutbacks

            while abs(target - lam) > 1e-12 * max(1.0, abs(target)):
                # Never step past the requested increment
                step = direction * min(step_size, abs(target - lam))

                # Load at the end of the substep, and the substep's increments
                F_ext = (lam + step) * self._neumann.ravel()
                DU = step * self._dirichlet.ravel()
                de0 = step * self._external_gradient

                # Element viscous stiffness alpha/dt * M for this substep. A
                # linear model caches K, so it must be rebuilt when it changes.
                k_visc = None
                if m is not None:
                    k_visc = alpha / abs(step) * m
                    if abs(step) != k_step:
                        self.K = torch.empty(0)
                    k_step = abs(step)

                # Previous state passed to the Newton solver. The adjoint
                # backward differentiates the residual w.r.t. this state, which
                # chains sensitivities across substeps. Clones are required when
                # tracking gradients, because the solver saves these tensors for
                # backward while the running state is replaced below.
                if track_parameter_gradients:
                    u_prev = u_cur.clone()
                    grad_prev = grad_cur.clone()
                    flux_prev = flux_cur.clone()
                    state_prev = state_cur.clone()
                else:
                    u_prev = u_cur.detach()
                    grad_prev = grad_cur.detach()
                    flux_prev = flux_cur.detach()
                    state_prev = state_cur.detach()

                # Solve for increment using Newton-Raphson method
                try:
                    du = newton_solve(
                        make_eval_residual(F_ext, DU, de0, k_visc),
                        du.detach(),
                        B,
                        max_iter,
                        rtol,
                        atol,
                        stol,
                        report,
                        solve_method,
                        preconditioner,
                        device,
                        u_prev,
                        grad_prev,
                        flux_prev,
                        state_prev,
                        *differentiable_parameters,
                    )
                except RuntimeError as err:
                    # Cut the substep back and retry from the same state
                    step_size = cutback_factor * abs(step)
                    if step_size < min_step:
                        raise RuntimeError(
                            f"Newton-Raphson did not converge in increment {n} "
                            f"after {max_cutbacks} cutbacks."
                        ) from err
                    if report is not None:
                        report.cutback()
                    continue

                # Evaluate converged state. Tangent not needed (compute_stiffness=False)
                du_eval = du.clone()
                du_eval[con] = DU[con]
                _, f_i, grad_cur, flux_cur, state_cur = self.integrate_material(
                    u_cur,
                    grad_cur,
                    flux_cur,
                    state_cur,
                    du_eval,
                    de0,
                    max_iter,
                    nlgeom,
                    compute_stiffness=False,
                )
                F_int = self.assemble_rhs(f_i)

                # Viscous forces balance the loads and their work is dissipated
                if k_visc is not None:
                    du_e = du_eval.view(-1, self.n_dof_per_node)[self.elements]
                    f_v = torch.einsum("...ij,...j->...i", k_visc, du_e.flatten(1))
                    F_v = self.assemble_rhs(f_v)
                    F_int = F_int + F_v
                    energy = energy + torch.dot(du_eval, F_v).detach()

                f_cur = F_int.reshape((-1, self.n_dof_per_node))
                u_cur = u_cur + du_eval.reshape((-1, self.n_dof_per_node))
                du = du_eval

                # Accept the substep and grow the next one. Growth applies to
                # the size the solver asked for, not to `step`, which is clipped
                # to land on the increment and would shrink the substep for good.
                lam += step
                if report is not None and not math.isclose(step_size, abs(span)):
                    report.growth()  # not already spanning the whole increment
                step_size = min(growth_factor * step_size, abs(span))

            # Carry the achieved fraction into the next increment
            step_frac = min(step_size / abs(span), 1.0) if span != 0.0 else 1.0

            # Store the results at the requested increment
            u[n] = u_cur
            f[n] = f_cur
            grad[n] = grad_cur
            flux[n] = flux_cur
            state[n] = state_cur
            self.stabilization_energy[n] = energy

            if report is not None:
                report.end()

        if report is not None:
            report.close()

        # Create output views without mutating tensors captured by eval_residual.
        out_u = u
        out_f = f
        out_flux = flux
        out_grad = grad
        out_state = state

        if aggregate_integration_points:
            out_grad = out_grad.mean(dim=1)
            out_flux = out_flux.mean(dim=1)
            out_state = out_state.mean(dim=1)

        out_flux = out_flux.squeeze((-2, -1))
        out_grad = out_grad.squeeze((-2, -1))

        if not track_parameter_gradients:
            out_u = out_u.detach()
            out_f = out_f.detach()
            out_flux = out_flux.detach()
            out_grad = out_grad.detach()
            out_state = out_state.detach()

        if return_intermediate:
            # Return all intermediate values
            return out_u, out_f, out_flux, out_grad, out_state
        else:
            # Return only the final values
            return out_u[-1], out_f[-1], out_flux[-1], out_grad[-1], out_state[-1]


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
        """Applied external nodal forces with shape [n_nod, n_dof_per_node]."""
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
        """Prescribed nodal displacements with shape [n_nod, n_dof_per_node].

        Values take effect only where `constraints` is True.
        """
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
        """External strain increment per element with shape [n_elem, d, d].

        Used to impose macroscopic strains, e.g. in homogenization.
        """
        return self._external_gradient

    @ext_strain.setter
    def ext_strain(self, value: Tensor):
        if not value.shape == (self.n_elem, self.n_dof_per_node, self.n_dim):
            raise ValueError("External strain must have the same shape as strains.")
        if not torch.is_floating_point(value):
            raise TypeError("External strain must be a floating-point tensor.")
        self._external_gradient = value.to(self.nodes.device)

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
        compute_stiffness: bool = True,
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
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
            compute_stiffness: If True, compute and return element stiffness k.
                When only forces are needed, pass False to avoid allocating the
                large k tensor.

        Returns:
            k: Element stiffness contributions (None when compute_stiffness is False).
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
        need_k = compute_stiffness and (
            self.K.numel() == 0 or self.n_state != 0 or nlgeom
        )
        k = (
            torch.zeros((self.n_elem, N_dof * N_nod, N_dof * N_nod), device=du.device)
            if need_k
            else None
        )

        # Initialize output for new state
        grad_new = torch.zeros_like(grad_prev)
        flux_new = torch.zeros_like(flux_prev)
        state_new = torch.zeros_like(state_prev)

        assert self.material is not None

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
            if need_k:
                assert k is not None
                BCB = torch.einsum("...Jp,...iJkL,...Lq->...piqk", B[i], ddsdde, B[i])
                BCB = BCB.reshape(-1, N_dof * N_nod, N_dof * N_nod)
                k += self.compute_k(detJ[i], BCB).mul_(w)

        return k, f, grad_new, flux_new, state_new

    def compute_m(self, detJ: Tensor, rho: Tensor) -> Tensor:
        raise NotImplementedError

    def solve_modes(self, n_modes: int) -> tuple[Tensor, Tensor]:
        """Compute the natural frequencies and mode shapes.

        Solves the generalized eigenvalue problem

        $$\\mathbf{K}\\boldsymbol{\\phi} = \\omega^2 \\mathbf{M}\\boldsymbol{\\phi}$$

        Args:
            n_modes: Number of eigenpairs to compute.

        Returns:
            Tuple ``(omega_sq, modes)`` where ``omega_sq`` has shape
                ``[n_modes]`` (squared angular frequencies, differentiable) and
                ``modes`` has shape ``[n_modes, n_nod, n_dof_per_node]`` (detached).
        """
        con = torch.nonzero(self.constraints.ravel(), as_tuple=False).ravel()
        free_indices = torch.nonzero(~self.constraints.ravel(), as_tuple=False).ravel()

        k = self.k0()
        K = self.assemble_matrix(k, con)

        m = self.integrate_mass()
        M = self.assemble_matrix(m, con)

        # free_indices restricts eigsh to the free-DOF subspace to avoid
        # spurious eigenvalues from the K_ii = M_ii = 1 penalty on constrained
        # DOFs.  Gradients still flow through the full K and M.
        omega_sq, phis = differentiable_modal_eigsolve(
            K, M, n_modes, free_indices=free_indices
        )

        modes = phis.T.reshape(n_modes, self.n_nod, self.n_dof_per_node)
        return omega_sq, modes


class Heat(FEM, ABC):
    """Base class for steady and transient heat conduction formulations."""

    supports_nlgeom = False

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
        """Applied external nodal heat sources with shape [n_nod, 1]."""
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
        """Prescribed nodal temperatures with shape [n_nod, 1].

        Values take effect only where `constraints` is True.
        """
        return self._dirichlet

    @temperatures.setter
    def temperatures(self, value: Tensor):
        if not value.shape == (self.n_nod, self.n_dof_per_node):
            raise ValueError("Temperatures must have the same shape as nodes.")
        if not torch.is_floating_point(value):
            raise TypeError("Temperatures must be a floating-point tensor.")
        self._dirichlet = value.to(self.nodes.device)

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
        compute_stiffness: bool = True,
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
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
        need_k = compute_stiffness and (self.K.numel() == 0 or self.n_state != 0)
        f = torch.zeros(
            self.n_elem,
            self.n_dof_per_node * N_nod,
            device=du.device,
            dtype=du.dtype,
        )
        k = (
            torch.zeros(
                (
                    self.n_elem,
                    self.n_dof_per_node * N_nod,
                    self.n_dof_per_node * N_nod,
                ),
                device=du.device,
                dtype=du.dtype,
            )
            if need_k
            else None
        )

        assert self.material is not None

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
            if need_k:
                assert k is not None
                BCB = torch.einsum("...ij,...iN,...jM->...NM", ddfddg, B[i], B[i])
                BCB = BCB.reshape(
                    -1, self.n_dof_per_node * N_nod, self.n_dof_per_node * N_nod
                )
                k += self.compute_k(detJ[i], BCB).mul_(w)

        return (
            k,
            f,
            torch.stack(grad_new),
            torch.stack(flux_new),
            torch.stack(state_new),
        )

    def time_integration(
        self,
        t_output: Tensor | None = None,
        delta_t: float = 1.0e-1,
        max_iter: int = 100,
        verbose: bool = False,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        device: str | None = None,
        method: Literal["direct", "cg", "bicgstab"] | None = None,
        preconditioner: Literal["amg", "jacobi", "none"] | None = None,
        aggregate_integration_points: bool = True,
        differentiable_parameters: Tensor | Iterable[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Integrate the heat equation in time with implicit increments.

        Computes an equilibrium state at t=0 under the current boundary
        conditions, then advances to each requested output time using internal
        steps of at most `delta_t`.

        Args:
            t_output: Requested output times. Results are returned at exactly
                these times.
            delta_t: Maximum internal time step.
            max_iter: Maximum Newton iterations per time step.
            verbose: If True, reports the solver configuration and a table of
                per-time-step progress, updated in place inside notebooks.
            rtol: Relative residual tolerance for Newton convergence.
            atol: Absolute residual tolerance for Newton convergence.
            stol: Tolerance used by iterative linear solvers.
            device: Optional device hint for the linear solver backend.
            method: Linear solver method, chosen by size and tangent symmetry
                when omitted.
            preconditioner: Preconditioner for an iterative method, chosen by
                device and available backends when omitted.
            aggregate_integration_points: If True, averages flux, gradient, and
                state over integration points.
            differentiable_parameters: Explicit parameters that should receive
                gradients through implicit solves. Accepts either a single
                tensor or an iterable of tensors.

        Returns:
            Tuple of temperature, internal vector, heat flux, temperature
            gradient, and material state, each with a leading axis of length
            `len(t_output)`.

        Raises:
            ValueError: If `t_output` is empty, negative, or not increasing.
            RuntimeError: If Newton iterations do not converge for a time step.
        """

        # Validate before self.constraints is modified below.
        t_output = torch.tensor([0.0, 1.0]) if t_output is None else t_output

        if t_output.numel() == 0:
            raise ValueError("t_output must contain at least one time.")
        if t_output.min() < 0.0:
            raise ValueError("t_output must not contain negative times.")
        if (t_output[1:] <= t_output[:-1]).any():
            raise ValueError("t_output must be strictly increasing.")

        # initial step: we get heat fluxes and temperature gradients for initial
        # conditions enforce initial conditions as boundary conditions

        bc_constraints = self.constraints.clone()
        self.constraints[:] = True

        # solve for initial conditions
        temp_eq, f_int_eq, heat_flux_eq, temp_grad_eq, alpha_eq = self.solve(
            aggregate_integration_points=False,
            differentiable_parameters=differentiable_parameters,
        )

        # Knots bound the intervals to subdivide; integration starts at t=0
        # even when it is not requested as an output time.
        knots = t_output
        if knots[0] > 0.0:
            knots = torch.cat((knots.new_zeros(1), knots))

        chunks = [knots[0:1]]
        # Row of the internal grid holding each output time.
        output_rows = [] if t_output[0] > 0.0 else [0]
        row = 0
        for t_start, t_end in pairwise(knots):
            # The tolerance keeps float error in an interval that is an exact
            # multiple of delta_t from adding a spurious substep.
            ratio = ((t_end - t_start) / delta_t).item()
            n_sub = max(1, math.ceil(ratio - 1e-9 * max(1.0, ratio)))
            sub = torch.linspace(t_start.item(), t_end.item(), n_sub + 1)[1:]
            # Restore the exact knot; linspace can miss it by an ulp.
            sub[-1] = t_end
            chunks.append(sub)
            row += n_sub
            output_rows.append(row)

        increments = torch.cat(chunks)
        t_rows = torch.tensor(output_rows)

        dt = increments[1:] - increments[:-1]  # time step sizes

        N_output = len(increments)

        # release boundary conditions, restore
        self.constraints[:] = bc_constraints

        # null space rigid body modes for AMG preconditioner
        B = self.near_null_space()

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
        state = torch.zeros(N_output, self.n_int, self.n_elem, self.n_state)

        # fill initial conditions
        u[0] = temp_eq
        f[0] = f_int_eq
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
        m = self.integrate_mass()

        # Initialize displacement increment
        du = torch.zeros(self.n_nod, self.n_dof_per_node).ravel()

        newton = (
            f"rtol {rtol:.0e} | atol {atol:.0e} | <={max_iter} it | dt <= {delta_t:g}"
        )
        # The transient tangent adds the mass matrix, which is symmetric, so the
        # material alone decides, as in `solve`.
        solve_method = resolve_method(self.n_dofs, method, self.symmetric_tangent)
        report = self._report(
            verbose,
            "time integration",
            u.dtype,
            solve_method,
            preconditioner,
            device,
            newton,
            label="Time step",
            value="Time",
            unit="time steps",
        )

        # Enforce initial BCs on u[0] explicitly, in case line_heat._dirichlet gives
        # updated BCs
        u[0].view(-1)[con] = self._dirichlet.view(-1)[con]

        for n in range(1, N_output):
            u_guess = u[n - 1].clone()
            dt_n = dt[n - 1]
            f_int_old = f[n - 1].clone()

            if report is not None:
                report.begin(n, float(increments[n]))

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

                # assemble stiffness and mass matrices, as COO: the sum below
                # and its accumulated gradient need MKL for CSR.
                if k is not None:
                    self.K = self.assemble_matrix(k, con).to_sparse_coo()
                if self.M.numel() == 0:
                    self.M = self.assemble_matrix(m, con).to_sparse_coo()

                f_inertia = self.M @ du

                residual = f_inertia.squeeze(-1) + 0.5 * dt_n * (
                    f_int_old.squeeze(-1) + f_int.squeeze(-1) - 2.0 * f_ext
                )

                residual[con] = 0.0
                res_norm = torch.linalg.norm(residual)

                # save initial residual
                if it == 0:
                    res_norm0 = res_norm

                # Report iteration information
                if report is not None:
                    report.iteration(it, res_norm)

                if res_norm < rtol * res_norm0 or res_norm < atol:
                    break

                du = differentiable_sparse_solve(
                    self.M + 0.5 * dt_n * self.K,
                    -residual,
                    B,
                    stol,
                    device,
                    solve_method,
                    preconditioner,
                )

                u_guess = u_guess + du.reshape((-1, self.n_dof_per_node))

            if res_norm > rtol * res_norm0 and res_norm > atol:
                raise RuntimeError("Newton-Raphson iteration did not converge.")

            u[n] = u_guess
            f[n] = f_int.reshape((-1, self.n_dof_per_node))

            if report is not None:
                report.end()

        if report is not None:
            report.close()

        # Create output views without mutating tensors captured by autograd.
        out_u = u[t_rows]
        out_f = f[t_rows]
        out_flux = flux[t_rows]
        out_grad = grad[t_rows]
        out_state = state[t_rows]

        if aggregate_integration_points:
            out_grad = out_grad.mean(dim=1)
            out_flux = out_flux.mean(dim=1)
            out_state = out_state.mean(dim=1)

        out_flux = out_flux.squeeze((-2, -1))
        out_grad = out_grad.squeeze((-2, -1))

        return out_u, out_f, out_flux, out_grad, out_state
