import math
import typing
import warnings
from collections.abc import Iterable, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import pyvista
import torch
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from pyvista import DataSet
from torch import Tensor

from .base import FEM, Heat
from .report import SolveReport, machine
from .sparse import describe_method, newton_solve

# An empty constraint set, so `assemble_matrix` leaves the part matrix raw.
EMPTY = torch.empty(0, dtype=torch.int64)


def _mv(A: Tensor, x: Tensor) -> Tensor:
    """Multiply a sparse matrix with a dense vector."""
    return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)


def _split(flat: Tensor, shapes: list[torch.Size]) -> list[Tensor]:
    """Split a concatenated flat tensor back into one tensor per part."""
    out = []
    start = 0
    for shape in shapes:
        size = math.prod(shape)
        out.append(flat[start : start + size].reshape(shape))
        start += size
    return out


class ReferencePoint:
    """A free node with rigid-body degrees of freedom and no stiffness.

    Coupling nodes to it makes them follow its rigid-body motion, which is how a
    moment reaches a solid mesh whose nodes carry translations only. The position
    sets the dimension: three translations and three rotations in 3D, two and one
    in 2D.

    Attributes:
        nodes: Position with shape [1, n_dim].
        forces: Applied forces and moments with shape [1, n_dofs].
        displacements: Prescribed displacements and rotations with shape
            [1, n_dofs].
        constraints: Boolean mask of constrained DOFs with shape [1, n_dofs].
    """

    n_nod = 1

    def __init__(self, position: Tensor | Sequence[float]):
        """Initialize a reference point at `position`, of shape [2] or [3]."""
        self.nodes = torch.as_tensor(position).reshape(1, -1)
        dim = self.nodes.shape[1]
        self.n_dof_per_node = dim + (3 if dim == 3 else 1)
        self.n_dofs = self.n_dof_per_node
        self._neumann = torch.zeros(1, self.n_dofs)
        self._dirichlet = torch.zeros(1, self.n_dofs)
        self._constraints = torch.zeros(1, self.n_dofs, dtype=torch.bool)

    forces = property(lambda self: self._neumann)
    displacements = property(lambda self: self._dirichlet)
    constraints = property(lambda self: self._constraints)

    def __repr__(self) -> str:
        at = ", ".join(f"{x:g}" for x in self.nodes[0].tolist())
        return f"<torch-fem reference point at ({at})>"


class ReferencePointHeat:
    """A free node with one temperature and no heat capacity.

    Coupling nodes to it holds them at one common temperature, so a surface can
    be driven by a single heat source and its total heat flow read back.

    Attributes:
        nodes: Position with shape [1, n_dim].
        heat_flux: Applied heat source with shape [1, 1].
        temperatures: Prescribed temperature with shape [1, 1].
        constraints: Boolean mask of constrained DOFs with shape [1, 1].
    """

    n_nod = 1
    n_dof_per_node = 1
    n_dofs = 1

    def __init__(self, position: Tensor | Sequence[float]):
        """Initialize a thermal reference point at `position`, of shape [2] or [3]."""
        self.nodes = torch.as_tensor(position).reshape(1, -1)
        self._neumann = torch.zeros(1, 1)
        self._dirichlet = torch.zeros(1, 1)
        self._constraints = torch.zeros(1, 1, dtype=torch.bool)

    heat_flux = property(lambda self: self._neumann)
    temperatures = property(lambda self: self._dirichlet)
    constraints = property(lambda self: self._constraints)

    def __repr__(self) -> str:
        at = ", ".join(f"{x:g}" for x in self.nodes[0].tolist())
        return f"<torch-fem thermal reference point at ({at})>"


Part = FEM | ReferencePoint | ReferencePointHeat
Point = ReferencePoint | ReferencePointHeat


class Assembly:
    """Several models coupled by kinematic constraints.

    Parts keep their own nodes, elements, materials and boundary conditions. The
    assembly stacks their degrees of freedom into one global vector, in the order
    the parts are given, and couples them with `coupling(...)`: translations
    follow `u_secondary = u_primary + theta_primary x (x_secondary - x_primary)`
    and rotations follow `theta_secondary = theta_primary`. A primary without
    rotations leaves a pure translation link, which joins a coincident mechanical
    interface and holds two thermal meshes at one temperature.

    Constraints are enforced by eliminating the secondary degrees of freedom, so
    the solved system stays symmetric and positive definite and needs no penalty
    parameter. The parts must share one physics and one spatial dimension.

    Assemblies solve on the reference configuration: `nlgeom` is not supported,
    since shells do not implement it and the rotation of a coupling is
    linearized.

    Attributes:
        parts: The coupled models, in global degree-of-freedom order.
        n_dofs: Total number of degrees of freedom, before elimination.
    """

    def __init__(self, parts: Sequence[Part]):
        """Initialize an assembly from its parts.

        Args:
            parts: Models and points to couple. They must share one physics,
                either all mechanical or all thermal.

        Raises:
            ValueError: If a part appears twice, or the parts mix physics or
                spatial dimensions.
        """
        if len({id(part) for part in parts}) != len(parts):
            raise ValueError("A part may appear only once in an assembly.")

        dims = {part.nodes.shape[1] for part in parts}
        if len(dims) != 1:
            raise ValueError("All parts of an assembly share one spatial dimension.")
        self.dim = dims.pop()

        # Displacements and temperatures share no equilibrium, so an assembly
        # solves one physics at a time.
        thermal = [isinstance(part, (Heat, ReferencePointHeat)) for part in parts]
        if any(thermal) and not all(thermal):
            raise ValueError("An assembly is either mechanical or thermal, not both.")

        self.parts = list(parts)
        sizes = [part.n_dofs for part in self.parts]
        self.offsets = [sum(sizes[:i]) for i in range(len(sizes))]
        self.n_dofs = sum(sizes)
        self._at = {id(part): offset for part, offset in zip(self.parts, self.offsets)}

        # Eliminated DOFs and the `(secondary, primary, coefficient)` row entries,
        # seeded empty so an assembly without constraints needs no special case.
        self._eliminated: list[Tensor] = [EMPTY]
        self._rows: list[tuple[Tensor, Tensor, Tensor]] = [
            (EMPTY, EMPTY, torch.empty(0))
        ]
        # The two ends of each coupling as (part, nodes), kept for `plot(...)`
        self._links: list[tuple[tuple[Part, Tensor], tuple[Part, Tensor]]] = []

    def __repr__(self) -> str:
        return f"<torch-fem assembly ({len(self.parts)} parts, {self.n_dofs} dofs)>"

    def coupling(
        self,
        secondary: Part,
        mask: Tensor,
        primary: Part,
        primary_mask: Tensor | None = None,
        dofs: Iterable[int] | None = None,
    ):
        """Couple nodes of a part to the nearest nodes of another part.

        Each secondary node follows the rigid-body motion of the primary node
        closest to it: a single-node primary such as a reference point drives
        them all, a primary surface pairs a coincident interface node for node,
        and a solid's through-thickness nodes pair with the shell node whose
        rotation carries their offset.

        Args:
            secondary: Part whose degrees of freedom are eliminated.
            mask: Boolean nodal mask with shape [n_nod] selecting the coupled
                nodes of `secondary`.
            primary: Part driving the motion.
            primary_mask: Boolean nodal mask selecting the candidate nodes of
                `primary`. Defaults to all of them, which is the single node of
                a `ReferencePoint`.
            dofs: Degrees of freedom of `secondary` to couple, as indices into
                its degrees of freedom per node. Defaults to all of them, so
                `dofs=[2]` couples only the displacement along z.
        """
        if id(secondary) not in self._at or id(primary) not in self._at:
            raise ValueError("Both parts must belong to this assembly.")

        n_sec = secondary.n_dof_per_node
        n_pri = primary.n_dof_per_node
        dofs = list(range(n_sec)) if dofs is None else list(dofs)
        if any(dof < 0 or dof >= n_sec for dof in dofs):
            raise ValueError(f"dofs must be indices in [0, {n_sec}) for this part.")
        if any(dof >= self.dim for dof in dofs) and n_pri <= self.dim:
            raise ValueError(
                "Rotational DOFs cannot be coupled to a primary part that has none."
            )

        # Pair each secondary node with the closest primary node
        secondary_nodes = torch.nonzero(mask).ravel()
        if primary_mask is None:
            primary_nodes = torch.arange(primary.n_nod)
        else:
            primary_nodes = torch.nonzero(primary_mask).ravel()
        x_secondary = secondary.nodes[secondary_nodes]
        distance = torch.cdist(x_secondary, primary.nodes[primary_nodes])
        primary_nodes = primary_nodes[distance.argmin(dim=1)]

        self._links.append(((secondary, secondary_nodes), (primary, primary_nodes)))

        secondary_base = self._at[id(secondary)] + secondary_nodes * n_sec
        primary_base = self._at[id(primary)] + primary_nodes * n_pri
        ones = torch.ones(len(secondary_nodes))

        skew = self._skew(x_secondary - primary.nodes[primary_nodes])
        axes = (0, 1, 2) if self.dim == 3 else (2,)

        for dof in dofs:
            self._eliminated.append(secondary_base + dof)
            self._rows.append((secondary_base + dof, primary_base + dof, ones))
            if dof < self.dim and n_pri > self.dim:
                for b, axis in enumerate(axes):
                    rot = primary_base + self.dim + b
                    self._rows.append((secondary_base + dof, rot, skew[:, dof, axis]))

    def _skew(self, r: Tensor) -> Tensor:
        """Columns of the `theta x r` operator: the basis vector `e_b` crossed into r.

        Padded to three components, so 2D rotates about z alone.
        """
        r = torch.cat([r, torch.zeros(len(r), 3 - self.dim)], dim=1)
        eye = torch.eye(3).expand(len(r), 3, 3)
        cross = torch.linalg.cross(eye, r[:, None, :].expand(-1, 3, -1), dim=-1)
        return cross.transpose(1, 2)

    def _build_T(self) -> tuple[Tensor, Tensor]:
        """Build the map from retained to all degrees of freedom.

        Returns:
            The sparse map `T` with shape [n_dofs, n_retained], such that
            `u = T q`, and the retained global degree-of-freedom indices.

        Raises:
            ValueError: If a degree of freedom is eliminated twice, or is both
                eliminated and used as a primary.
        """
        eliminated = torch.cat(self._eliminated)
        if len(torch.unique(eliminated)) != len(eliminated):
            raise ValueError("A DOF is eliminated by more than one constraint.")

        keep = torch.ones(self.n_dofs, dtype=torch.bool)
        keep[eliminated] = False
        retained = torch.nonzero(keep).ravel()
        column = torch.full((self.n_dofs,), -1, dtype=torch.int64)
        column[retained] = torch.arange(len(retained))

        secondary, primary, coeffs = (torch.cat(x) for x in zip(*self._rows))
        if bool((column[primary] < 0).any()):
            raise ValueError(
                "A DOF is both eliminated and used as a primary. Chain the "
                "constraints to an independent part instead."
            )

        rows = torch.cat([retained, secondary])
        cols = torch.cat([column[retained], column[primary]])
        values = torch.cat([torch.ones(len(retained)), coeffs])
        with torch.sparse.check_sparse_tensor_invariants(False):
            T = torch.sparse_coo_tensor(
                torch.stack([rows, cols]), values, (self.n_dofs, len(retained))
            ).coalesce()
        return T, retained

    def _stiffness(
        self, blocks: list[tuple[int, Tensor]], T: Tensor, con: Tensor
    ) -> Tensor:
        """Assemble the part tangents, eliminate the secondary DOFs, and constrain.

        Each part's indices are sorted and the blocks sit on disjoint ascending
        ranges of the global numbering, so their concatenation is already
        coalesced.
        """
        n = T.shape[1]
        idx = torch.cat([K._indices() + offset for offset, K in blocks], dim=1)
        val = torch.cat([K._values() for _, K in blocks])

        with (
            torch.sparse.check_sparse_tensor_invariants(False),
            warnings.catch_warnings(),
        ):
            # The sparse-sparse product routes through CSR, whose beta notice is noise
            warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta")
            K = torch.sparse_coo_tensor(
                idx, val, (self.n_dofs, self.n_dofs), is_coalesced=True
            )
            K = torch.sparse.mm(T.transpose(0, 1), torch.sparse.mm(K, T)).coalesce()

            # A part without stiffness, e.g. a reference point, contributes no
            # diagonal, so add explicit zeros before writing the unit entries.
            diagonal = torch.arange(n)
            idx = torch.cat([K._indices(), torch.stack([diagonal, diagonal])], dim=1)
            val = torch.cat([K._values(), torch.zeros(n)])
            K = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

            idx, val = K._indices(), K._values().clone()
            is_constrained = torch.zeros(n, dtype=torch.bool)
            is_constrained[con] = True
            val[is_constrained[idx[0]] | is_constrained[idx[1]]] = 0.0
            on_diagonal = idx[0] == idx[1]
            position = torch.zeros(n, dtype=torch.int64)
            position[idx[0][on_diagonal]] = torch.nonzero(on_diagonal).ravel()
            val[position[con]] = 1.0
            return torch.sparse_coo_tensor(idx, val, (n, n), is_coalesced=True)

    def _rigid_modes(self) -> Tensor:
        """The near-null space an algebraic multigrid setup needs, over all DOFs.

        Rigid body motions for a mechanical assembly and a constant field for a
        thermal one. `FEM.compute_B(...)` builds this for a single part, but
        returns a different number of modes per part, which cannot be stacked.
        """
        if self.parts[0].n_dof_per_node < self.dim:
            return torch.ones(self.n_dofs, 1)

        axes = (0, 1, 2) if self.dim == 3 else (2,)
        modes = torch.zeros(self.n_dofs, self.dim + len(axes))
        for part, offset in zip(self.parts, self.offsets):
            base = offset + torch.arange(part.n_nod) * part.n_dof_per_node
            skew = self._skew(part.nodes)
            for a in range(self.dim):
                modes[base + a, a] = 1.0
                for b, axis in enumerate(axes):
                    modes[base + a, self.dim + b] = skew[:, a, axis]
            if part.n_dof_per_node > self.dim:
                for b in range(len(axes)):
                    modes[base + self.dim + b, self.dim + b] = 1.0
        return modes

    def _report(
        self, verbose: bool, method: str | None, device: str | None, newton: str
    ) -> SolveReport | None:
        """Open a report on this assembly and its linear solver, if verbose."""
        if not verbose:
            return None
        device = device or self.parts[0].nodes.device.type
        n_elem = sum(getattr(part, "n_elem", 0) for part in self.parts)
        dtype = str(torch.get_default_dtype()).removeprefix("torch.")
        header = {
            "model": f"Assembly | {len(self.parts)} parts | {n_elem:,} elem | "
            f"{self.n_dofs:,} dof | {dtype}",
            "machine": machine(device),
            "solver": describe_method(self.n_dofs, device, method),
            "newton": newton,
        }
        return SolveReport("torch-fem | solve", header)

    def solve(
        self,
        increments: Tensor | None = None,
        max_iter: int = 10,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        stol: float = 1e-10,
        verbose: bool = False,
        method: Literal["spsolve", "minres", "cg", "pardiso", "amgx"] | None = None,
        device: str | None = None,
        return_intermediate: bool = False,
        aggregate_integration_points: bool = True,
        differentiable_parameters: Tensor | Iterable[Tensor] | None = None,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Solve the constrained quasi-static problem by load increments.

        Args:
            increments: Load scale factors, typically [0, 1]. Unlike
                `FEM.solve(...)`, an increment that does not converge is not
                subdivided, so a nonlinear problem needs its own increments.
            max_iter: Maximum Newton iterations per increment.
            rtol: Relative residual tolerance for Newton convergence.
            atol: Absolute residual tolerance for Newton convergence.
            stol: Tolerance used by iterative linear solvers.
            verbose: If True, reports solver configuration and progress.
            method: Linear solver backend name.
            device: Optional device hint for the linear solver backend.
            return_intermediate: If True, returns values for all increments.
            aggregate_integration_points: If True, averages flux, gradient, and
                state over integration points.
            differentiable_parameters: Explicit parameter(s) to differentiate
                through the implicit Newton solve.

        Returns:
            Tuple of displacement, internal force, flux, gradient, and material
                state, each a list with one entry per part. A reference point
                contributes empty flux, gradient and state. A retained DOF
                carries what the constraints transmit into it, so its force is
                the reaction where it is constrained and the coupling load at a
                reference point, while an eliminated one carries the part's own
                internal force.

        Raises:
            ValueError: If a constrained degree of freedom is eliminated.
        """
        increments = torch.tensor([0.0, 1.0]) if increments is None else increments
        N = len(increments)

        if differentiable_parameters is None:
            differentiable_parameters = ()
        elif isinstance(differentiable_parameters, Tensor):
            differentiable_parameters = (differentiable_parameters,)
        else:
            differentiable_parameters = tuple(differentiable_parameters)
        track = any(param.requires_grad for param in differentiable_parameters)

        T, retained = self._build_T()
        Tt = T.transpose(0, 1)

        # Global boundary conditions, gathered from the parts in DOF order
        neumann = torch.cat([part._neumann.ravel() for part in self.parts])
        dirichlet = torch.cat([part._dirichlet.ravel() for part in self.parts])
        constrained = torch.cat([part._constraints.ravel() for part in self.parts])
        con = torch.nonzero(constrained[retained]).ravel()
        if len(con) != int(constrained.sum()):
            raise ValueError(
                "A constrained DOF is eliminated by a constraint. Constrain the "
                "primary part instead."
            )

        # Per-part results, and the shapes that split the flat state back up
        u = [torch.zeros(N, part.n_nod, part.n_dof_per_node) for part in self.parts]
        f = [torch.zeros(N, part.n_nod, part.n_dof_per_node) for part in self.parts]
        grad, flux, state = [], [], []
        for part in self.parts:
            if not isinstance(part, FEM):
                grad.append(torch.zeros(N, 0))
                flux.append(torch.zeros(N, 0))
                state.append(torch.zeros(N, 0))
                continue
            field = (N, part.n_int, part.n_elem, *part.n_flux)
            grad.append(torch.zeros(field))
            grad[-1][:] = part.initial_grad
            flux.append(torch.zeros(field))
            state.append(torch.zeros(N, part.n_int, part.n_elem, part.n_state))
        shapes = [[x[0].shape for x in q] for q in (u, grad, flux, state)]

        newton = f"rtol {rtol:.0e} | atol {atol:.0e} | <={max_iter} it"
        report = self._report(verbose, method, device, newton)

        # Each part caches its own tangent block, reused when a linear material
        # reports no new element stiffness.
        for part in self.parts:
            if isinstance(part, FEM):
                part.K = torch.empty(0)

        def constrain(dq: Tensor, DU: Tensor) -> Tensor:
            """Enforce the Dirichlet increment on the retained DOFs."""
            dq = dq.clone()
            dq[con] = DU[con]
            return dq

        def integrate(prev, du, step, iteration, tangent=True):
            """Integrate every part over the global increment `du`.

            Returns the part tangent blocks, the global internal force, and the
            new gradient, flux and state as one list per quantity.
            """
            u_p, grad_p, flux_p, state_p = (
                _split(value, shape) for value, shape in zip(prev, shapes)
            )
            blocks: list[tuple[int, Tensor]] = []
            F_int, updated = [], []
            for j, part in enumerate(self.parts):
                if not isinstance(part, FEM):
                    F_int.append(torch.zeros(part.n_dofs))
                    updated.append((grad_p[j], flux_p[j], state_p[j]))
                    continue
                offset = self.offsets[j]
                k, f_e, *new = part.integrate_material(
                    u_p[j],
                    grad_p[j],
                    flux_p[j],
                    state_p[j],
                    du[offset : offset + part.n_dofs],
                    step * part._external_gradient,
                    iteration,
                    False,  # nlgeom, see the note in the class docstring
                    compute_stiffness=tangent,
                )
                if k is not None:
                    part.K = part.assemble_matrix(k, EMPTY)
                blocks.append((offset, part.K))
                F_int.append(part.assemble_rhs(f_e))
                updated.append(tuple(new))
            return blocks, torch.cat(F_int), list(zip(*updated))

        def make_eval_residual(F_ext, DU, step):
            # Bind this increment's loads at definition time, so the adjoint
            # backward replays the increment it belongs to.
            def eval_residual(dq, iteration, *prev):
                du = _mv(T, constrain(dq, DU))
                blocks, F_int, _ = integrate(prev, du, step, iteration)
                res = _mv(Tt, F_int - F_ext)
                res[con] = 0.0
                return res, self._stiffness(blocks, T, con)

            return eval_residual

        # Iterative solvers build their preconditioner from the near-null space.
        # `T` is the identity on the retained rows, so restricting the modes to
        # them is the `q` of `u = T q` that reproduces each one.
        B = self._rigid_modes()[retained]

        dq = torch.zeros(len(retained))
        carry = (
            torch.zeros(self.n_dofs),
            *(torch.cat([x[0].ravel() for x in q]) for q in (grad, flux, state)),
        )

        for n in range(1, N):
            level = float(increments[n])
            step = level - float(increments[n - 1])
            if report is not None:
                report.begin(n, level)

            F_ext = level * neumann
            DU = (step * dirichlet)[retained]

            # The solver saves these for backward while the running state is
            # replaced below, so they are cloned when gradients are tracked.
            prev = tuple(x.clone() if track else x.detach() for x in carry)

            dq = newton_solve(
                make_eval_residual(F_ext, DU, step),
                dq.detach(),
                B,
                max_iter,
                rtol,
                atol,
                stol,
                report,
                method,
                device,
                None,
                False,
                *prev,
                *differentiable_parameters,
            )

            # Evaluate the converged state, whose tangent is not needed
            dq = constrain(dq, DU)
            du = _mv(T, dq)
            _, F_int, updated = integrate(prev, du, step, max_iter, tangent=False)

            # A retained DOF also carries what the constraints transmit into it,
            # which is the only force a reference point ever sees.
            f_cur = F_int.index_put((retained,), _mv(Tt, F_int))
            carry = (
                prev[0] + du,
                *(torch.cat([x.ravel() for x in q]) for q in updated),
            )

            # Store the results at the requested increment
            for j, part in enumerate(self.parts):
                block = slice(self.offsets[j], self.offsets[j] + part.n_dofs)
                u[j][n] = carry[0][block].reshape(shapes[0][j])
                f[j][n] = f_cur[block].reshape(shapes[0][j])
                grad[j][n], flux[j][n], state[j][n] = (q[j] for q in updated)

            if report is not None:
                report.end()

        if report is not None:
            report.close()

        if aggregate_integration_points:
            # A reference point has no integration points to average over
            grad = [x.mean(dim=1) if x.dim() > 2 else x for x in grad]
            flux = [x.mean(dim=1) if x.dim() > 2 else x for x in flux]
            state = [x.mean(dim=1) if x.dim() > 2 else x for x in state]

        flux = [x.squeeze((-2, -1)) if x.dim() > 2 else x for x in flux]
        grad = [x.squeeze((-2, -1)) if x.dim() > 2 else x for x in grad]

        out = [u, f, flux, grad, state]
        if not track:
            out = [[x.detach() for x in q] for q in out]
        if not return_intermediate:
            out = [[x[-1] for x in q] for q in out]
        return out[0], out[1], out[2], out[3], out[4]

    def _resolve(
        self, u: list[Tensor] | float, kwargs: dict
    ) -> tuple[list[tuple[FEM, dict]], list[tuple[Point, Tensor]], list[Tensor], float]:
        """Everything a plot draws, moved by the displacement it is given.

        A list is the parts' shares of an argument, inside a dict too, so `u`
        and `node_property={"u": u}` split alike. Returns the arguments each
        mesh part is plotted with, the points with their positions, one [2n, n_dim]
        tensor per coupling holding the secondary nodes followed by the primary
        ones they pair with, and the extent of the whole assembly.
        """
        per_part, moved = [], {}
        for j, part in enumerate(self.parts):
            arguments = {}
            for name, value in kwargs.items():
                if isinstance(value, dict):
                    value = {
                        k: v[j] if isinstance(v, list) else v for k, v in value.items()
                    }
                arguments[name] = value[j] if isinstance(value, list) else value

            displacement = u[j] if isinstance(u, list) else u
            if isinstance(displacement, Tensor):
                # Translations only: a shell also carries rotations, and a
                # temperature is no displacement at all
                moves = part.n_dof_per_node >= self.dim
                displacement = displacement[:, : self.dim] if moves else 0.0
            arguments["u"] = displacement
            if isinstance(part, FEM):
                per_part.append((part, arguments))
            moved[id(part)] = part.nodes + displacement

        points = [(p, moved[id(p)][0]) for p in self.parts if not isinstance(p, FEM)]
        links = [
            torch.cat([moved[id(part)][nodes] for part, nodes in ends])
            for ends in self._links
        ]
        span = torch.cat(list(moved.values()))
        size = torch.linalg.norm(span.max(dim=0).values - span.min(dim=0).values)
        return per_part, points, links, float(size)

    def plot(self, u: list[Tensor] | float = 0.0, **kwargs):
        """Plot the assembly in 2D (matplotlib) or 3D (PyVista).

        Dispatches to `plot2d` or `plot3d` based on the spatial dimension.

        Args:
            u: Nodal displacements per part, e.g. the `u` of `solve(...)`.
                Defaults to 0.0 (undeformed).
            **kwargs: Forwarded to `plot2d` or `plot3d`, and from there to
                every part's own `plot(...)`. A list is spread over the parts,
                inside a dict too, so `node_property={"u": u}` splits like `u`.
                `bcs=True` renders the boundary conditions of the parts and of
                the reference points alike.
        """
        if self.dim == 2:
            self.plot2d(u=u, **kwargs)
        else:
            self.plot3d(u=u, **kwargs)

    def plot2d(
        self,
        u: list[Tensor] | float = 0.0,
        ax: Axes | None = None,
        bcs: bool = False,
        **kwargs,
    ):
        """Plot the parts with matplotlib, marking the points and couplings.

        Args:
            u: Nodal displacements per part. Defaults to 0.0 (undeformed).
            ax: Matplotlib axes to draw into. Defaults to a new figure.
            bcs: If True, renders the boundary conditions of every part and of
                the reference points, whose rotations are drawn as circular
                arrows.
            **kwargs: Forwarded to each part's `plot(...)`.
        """
        from .plot_utils import arrows2d, dots2d, markers2d

        if ax is None:
            _, ax = plt.subplots()
        per_part, points, links, size = self._resolve(u, kwargs)

        for part, arguments in per_part:
            part.plot(ax=ax, bcs=bcs, **arguments)

        deformed = isinstance(u, list)
        width = 0.01 * size
        for point, center in points:
            ax.plot(*center.tolist(), "o", color="gray", zorder=3)
            # A temperature draws no glyph
            if not bcs or isinstance(point, ReferencePointHeat):
                continue
            node = center[None]
            prescribed = torch.where(point.constraints, point.displacements, 0.0)
            # A prescribed rotation cannot be drawn to scale, so it keeps its mark
            fixed = point.constraints.clone()
            fixed[:, :2] &= deformed | (prescribed[:, :2] == 0.0)
            arrows2d(ax, node, point.forces[:, :2], width, span=0.1 * size)
            if not deformed:
                arrows2d(ax, node, prescribed[:, :2], width)
            tip = node if deformed else node + prescribed[:, :2]
            if prescribed[:, :2].any():
                dots2d(ax, tip)
            markers2d(ax, node, fixed[:, :2])
            if fixed[:, 2:].any() or point.forces[:, 2:].any():
                ax.plot(
                    *center.tolist(),
                    marker="$\\circlearrowleft$",
                    markersize=18,
                    color="gray",
                    clip_on=False,
                )

        for ends in links:
            n = len(ends) // 2
            segments = [[a.tolist(), b.tolist()] for a, b in zip(ends[:n], ends[n:])]
            ax.add_collection(LineCollection(segments, colors="gray", zorder=2))
            dots2d(ax, ends)

        # A part fixes the limits to its own mesh, which a point may sit outside
        at = [center for _, center in points] + links
        if at:
            ax.update_datalim(torch.cat([x.reshape(-1, 2) for x in at]))
            ax.set_autoscale_on(True)
            ax.autoscale_view()

    def plot3d(
        self,
        u: list[Tensor] | float = 0.0,
        plotter: pyvista.Plotter | None = None,
        bcs: bool = False,
        **kwargs,
    ):
        """Plot the parts with PyVista, marking the points and couplings.

        Points are drawn as spheres and each coupling as a tube joining the
        paired nodes, both sized like the boundary condition markers of a part.

        Args:
            u: Nodal displacements per part. Defaults to 0.0 (undeformed).
            plotter: PyVista plotter to draw into. If None, one is created and
                shown once the whole assembly is drawn.
            bcs: If True, renders the boundary conditions of every part and of
                the reference points, whose moments and rotations are drawn
                with a doubled head.
            **kwargs: Forwarded to each part's `plot(...)`.
        """
        from .plot_utils import arrows, cones, dots

        pl = pyvista.Plotter() if plotter is None else plotter
        per_part, points, links, size = self._resolve(u, kwargs)
        # Glyphs follow the element size, with a floor so that they stay visible
        # on a fine mesh, whose elements are far smaller than the model
        elements = [part.char_lengths for part in self.parts if isinstance(part, FEM)]
        scale = max(0.5 * float(torch.cat(elements).mean()), 0.02 * size)

        for part, arguments in per_part:
            part.plot(plotter=pl, bcs=bcs, **arguments)

        deformed = isinstance(u, list)
        span = 0.1 * size
        for point, center in points:
            sphere = pyvista.Sphere(radius=0.3 * scale, center=center.tolist())
            pl.add_mesh(sphere, color="gray")
            # A temperature draws no glyph
            if not bcs or isinstance(point, ReferencePointHeat):
                continue
            node = center[None]
            prescribed = torch.where(point.constraints, point.displacements, 0.0)
            # A prescribed rotation cannot be drawn to scale, so it keeps its cone
            fixed = point.constraints.clone()
            fixed[:, : self.dim] &= deformed | (prescribed[:, : self.dim] == 0.0)
            arrows(pl, node, point.forces[:, : self.dim], span=span)
            arrows(pl, node, point.forces[:, self.dim :], span=span, doubled=True)
            if not deformed:
                arrows(pl, node, prescribed[:, : self.dim])
            tip = node if deformed else node + prescribed[:, : self.dim]
            if prescribed[:, : self.dim].any():
                dots(pl, tip, 0.3 * scale)
            cones(pl, node, fixed[:, : self.dim], scale)
            cones(pl, node, fixed[:, self.dim :], scale, doubled=True)

        for ends in links:
            # A VTK line cell is its point count followed by its point indices
            n = len(ends) // 2
            cells = [i for k in range(n) for i in (2, k, k + n)]
            link = pyvista.PolyData(ends.cpu().numpy(), lines=cells)
            tube = typing.cast(DataSet, link.tube(radius=0.1 * scale))
            pl.add_mesh(tube, color="gray")

        if plotter is None:
            from .plot_utils import show_html

            show_html(pl)
