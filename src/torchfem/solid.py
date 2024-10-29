import torch

from .base import FEM
from .elements import Hexa1, Hexa2, Tetra1, Tetra2
from .materials import Material


class Solid(FEM):
    def __init__(self, nodes: torch.Tensor, elements: torch.Tensor, material: Material):
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
        self.n_strains = 6
        self.n_int = len(self.etype.iweights())

        # Initialize external strain
        self.ext_strain = torch.zeros(self.n_elem, self.n_strains)

    def D(self, B: torch.Tensor, _):
        """Element gradient operator"""
        zeros = torch.zeros(self.n_elem, self.etype.nodes)
        shape = [self.n_elem, -1]
        D0 = torch.stack([B[:, 0, :], zeros, zeros], dim=-1).reshape(shape)
        D1 = torch.stack([zeros, B[:, 1, :], zeros], dim=-1).reshape(shape)
        D2 = torch.stack([zeros, zeros, B[:, 2, :]], dim=-1).reshape(shape)
        D3 = torch.stack([zeros, B[:, 2, :], B[:, 1, :]], dim=-1).reshape(shape)
        D4 = torch.stack([B[:, 2, :], zeros, B[:, 0, :]], dim=-1).reshape(shape)
        D5 = torch.stack([B[:, 1, :], B[:, 0, :], zeros], dim=-1).reshape(shape)
        return torch.stack([D0, D1, D2, D3, D4, D5], dim=1)

    def compute_k(self, detJ: torch.Tensor, DCD: torch.Tensor):
        """Element stiffness matrix"""
        return torch.einsum("j,jkl->jkl", detJ, DCD)

    def compute_f(self, detJ: torch.Tensor, D: torch.Tensor, S: torch.Tensor):
        """Element internal force vector."""
        return torch.einsum("j,jkl,jk->jl", detJ, D, S)

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
