import pyvista
import torch
from torch import Tensor

from .base import FEM
from .elements import Hexa1, Hexa2, Tetra1, Tetra2
from .materials import Material


class Solid(FEM):
    def __init__(self, nodes: Tensor, elements: Tensor, material: Material):
        """Initialize the solid FEM problem."""

        super().__init__(nodes, elements, material)

        # Set element type depending on number of nodes per element
        if len(elements[0]) == 4:
            self.etype = Tetra1()
        elif len(elements[0]) == 8:
            self.etype = Hexa1()
        elif len(elements[0]) == 10:
            self.etype = Tetra2()
        elif len(elements[0]) == 20:
            self.etype = Hexa2()
        else:
            raise ValueError("Element type not supported.")

        # Set element type specific sizes
        self.n_stress = 3
        self.n_int = len(self.etype.iweights())

        # Initialize external strain
        self.ext_strain = torch.zeros(self.n_elem, 3, 3)

    def eval_shape_functions(self, xi: Tensor, u: Tensor | float = 0.0) -> Tensor:
        """Gradient operator at integration points xi."""
        nodes = self.nodes + u
        nodes = nodes[self.elements, :]
        b = self.etype.B(xi)
        J = torch.einsum("jk,mkl->mjl", b, nodes)
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise Exception("Negative Jacobian. Check element numbering.")
        B = torch.einsum("jkl,lm->jkm", torch.linalg.inv(J), b)
        return self.etype.N(xi), B, detJ

    def compute_k(self, detJ: Tensor, BCB: Tensor) -> Tensor:
        """Element stiffness matrix"""
        return torch.einsum("j,jkl->jkl", detJ, BCB)

    def compute_f(self, detJ: Tensor, B: Tensor, S: Tensor) -> Tensor:
        """Element internal force vector."""
        return torch.einsum("...,...ik,...ij->...kj", detJ, B, S)

    @torch.no_grad()
    def plot(
        self,
        u: float | Tensor = 0.0,
        node_property: dict[str, Tensor] | None = None,
        element_property: dict[str, Tensor] | None = None,
        orientations: Tensor | None = None,
        show_edges: bool = True,
        show_undeformed: bool = False,
        contour: tuple[str, list[float]] | None = None,
        plotter: pyvista.Plotter | None = None,
        threshold_condition: torch.Tensor | None = None,
        **kwargs,
    ):
        """Plot the mesh with optional node and element properties.

        Args:
            u (float or torch.Tensor, optional):
                Displacement field. Defaults to 0.0.
            node_property (dict[str, torch.Tensor], optional):
                Nodal property to plot. Defaults to None.
            element_property (dict[str, torch.Tensor], optional):
                Element property to plot. Defaults to None.
            orientations (torch.Tensor, optional):
                Element orientations. Defaults to None.
            show_edges (bool, optional):
                Show edges. Defaults to True.
            show_undeformed (bool, optional):
                Show undeformed mesh. Defaults to False.
            contour (tuple[str, list[float]], optional):
                Contour plot. Defaults to None.
            plotter (pyvista.Plotter, optional):
                PyVista plotter. Defaults to None.
            threshold_condition (torch.Tensor, optional):
                Threshold condition to recover subshape. Defaults to None.
            **kwargs:
                Additional keyword arguments passed to pyvista.Plotter.add_mesh.
        """

        pyvista.set_plot_theme("document")
        pl = pyvista.Plotter() if plotter is None else plotter
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
        el = len(self.elements[0]) * torch.ones(self.n_elem, dtype=self.elements.dtype)
        elements = torch.cat([el[:, None], self.elements], dim=1).view(-1).tolist()

        # Deformed node positions
        pos = self.nodes + u

        # Create unstructured mesh
        mesh = pyvista.UnstructuredGrid(elements, cell_types, pos.tolist())

        # Plot node properties
        if node_property:
            for key, val in node_property.items():
                mesh.point_data[key] = val.cpu().numpy()

        # Plot cell properties
        if element_property:
            for key, val in element_property.items():
                mesh.cell_data[key] = val.cpu().numpy()

        if threshold_condition is None:
            threshold_condition = torch.ones(self.n_elem, dtype=torch.bool)

        # Apply threshold to recover subshape
        mesh = mesh.extract_cells(threshold_condition.cpu().numpy())

        # Plot orientations
        if orientations is not None:
            ecenters = pos[self.elements].mean(dim=1)
            for j, color in enumerate(["red", "green", "blue"]):
                directions = orientations[:, j, :]
                pl.add_arrows(
                    ecenters.cpu().numpy()[threshold_condition],
                    directions.cpu().numpy()[threshold_condition],
                    mag=0.5,
                    color=color,
                    show_scalar_bar=False,
                )

        # Plot mesh
        if contour:
            scalars, values = contour
            pl.add_mesh(mesh.outline(), color="black")
            pl.add_mesh(mesh.contour(values, scalars=scalars), **kwargs)
        else:
            if show_edges:
                if isinstance(self.etype, Tetra2) or isinstance(self.etype, Hexa2):
                    # Trick to plot edges for quadratic elements
                    # See: https://github.com/pyvista/pyvista/discussions/5777
                    surface = mesh.separate_cells().extract_surface(
                        nonlinear_subdivision=4
                    )
                    edges = surface.extract_feature_edges()
                    pl.add_mesh(surface, **kwargs)
                    actor = pl.add_mesh(edges, style="wireframe", color="black")
                    actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
                else:
                    pl.add_mesh(mesh, show_edges=True, **kwargs)
            else:
                pl.add_mesh(mesh, **kwargs)

        if show_undeformed:
            undefo = pyvista.UnstructuredGrid(elements, cell_types, self.nodes.tolist())
            edges = (
                undefo.separate_cells()
                .extract_surface(nonlinear_subdivision=4)
                .extract_feature_edges()
            )
            pl.add_mesh(edges, style="wireframe", color="grey")

        if plotter is None:
            pl.show(jupyter_backend="html")
