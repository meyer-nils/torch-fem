from abc import ABC, abstractmethod
from math import sqrt
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

# Registry of all concrete Element subclasses
ELEMENT_REGISTRY: list[type["Element"]] = []


class ClassPropertyDescriptor:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, cls):
        return self.fget(cls)


def classproperty(func):
    return ClassPropertyDescriptor(func)


class Element(ABC):
    """Abstract base class for isoparametric finite elements.

    Concrete element types define interpolation and integration on a reference
    (isoparametric) domain.

    Attributes:
        iso_volume (float): Measure of the reference element
            (length/area/volume).
        iso_dim (int): Reference-space dimension.
        nodes (int): Number of nodes per element.
        meshio_type (Literal): Mesh cell type used for meshio I/O.
    """

    iso_volume: float
    iso_dim: int
    nodes: int
    meshio_type: Literal[
        "line",
        "triangle",
        "triangle6",
        "quad",
        "quad8",
        "tetra",
        "tetra10",
        "hexahedron",
        "hexahedron20",
    ]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register concrete subclasses (exclude the abstract base itself)
        if cls is not Element:
            ELEMENT_REGISTRY.append(cls)

    @classmethod
    @abstractmethod
    def N(cls, xi: Tensor) -> Tensor:
        """Evaluate shape functions at reference coordinates.

        Args:
            xi (Tensor): Reference coordinates.
                *Shape:* `(iso_dim,)` or `(n_points, iso_dim)`.

        Returns:
            Tensor: Shape function values.
                *Shape:* `(nodes,)` or `(n_points, nodes)`.
        """
        pass

    @classmethod
    @abstractmethod
    def B(cls, xi: Tensor) -> Tensor:
        """Evaluate reference-space derivatives of shape functions.

        Args:
            xi (Tensor): Reference coordinates.
                *Shape:* `(iso_dim,)` or `(n_points, iso_dim)`.

        Returns:
            Tensor: Derivatives `dN/dxi`.
                *Shape:* `(iso_dim, nodes)` or `(n_points, iso_dim, nodes)`.
        """
        pass

    @classproperty
    @abstractmethod
    def iso_coords(cls) -> Tensor:
        """Return reference coordinates of element nodes.

        Returns:
            Tensor: Node coordinates in reference space.
                *Shape:* `(nodes, iso_dim)`.
        """
        pass

    @classproperty
    @abstractmethod
    def ipoints(cls) -> Tensor:
        """Return integration points in reference coordinates.

        Returns:
            Tensor: Integration points.
                *Shape:* `(n_ip, iso_dim)`.
        """
        pass

    @classproperty
    @abstractmethod
    def iweights(cls) -> Tensor:
        """Return integration weights associated with `ipoints`.

        Returns:
            Tensor: Integration weights.
                *Shape:* `(n_ip,)`.
        """
        pass


class Bar1(Element):
    """Two-node linear line element.

    Notes:
        Node ordering:

            0 ---- 1
    """

    iso_volume = 2.0
    iso_dim = 1
    nodes = 2
    meshio_type = "line"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor([[-1.0], [1.0]])

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = 1 - xi[..., 0]
        N_2 = 1 + xi[..., 0]
        return 1 / 2 * torch.stack([N_1, N_2], dim=-1)

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        if xi.dim() == 1:
            return torch.tensor([[-0.5, 0.5]])
        else:
            N = xi.shape[0]
            return torch.tensor([[-0.5, 0.5]]).repeat(N, 1, 1)

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor([2.0])

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor([[0.0]])

    @classmethod
    def plot(cls, n_points: int = 100):
        import matplotlib.pyplot as plt

        # Compute shape functions at evenly spaced points in reference space
        xi = torch.linspace(-1.0, 1.0, n_points).unsqueeze(-1)
        N = cls.N(xi)

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4))
        for i in range(cls.nodes):
            ax.plot(xi, N[:, i], linewidth=2.0, label=f"$N_{i}$")
        ax.set_xlabel("$\\xi$")
        ax.set_ylabel("$N_i(\\xi)$")
        ax.grid(alpha=0.3)
        ax.legend()

        # Save plot to docs/images directory
        root = Path(__file__).resolve().parents[2] / "docs" / "images"
        save_path = root / "Bar1_shape_functions.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")


class Bar2(Bar1):
    """Three-node quadratic line element.

    Notes:
        Node ordering:

            0 -- 2 -- 1
    """

    nodes = 3
    meshio_type = "line"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor([[-1.0], [1.0], [0.0]])

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = 1 / 2 * xi[..., 0] * (xi[..., 0] - 1)
        N_2 = 1 / 2 * xi[..., 0] * (xi[..., 0] + 1)
        N_3 = 1 - xi[..., 0] ** 2
        return torch.stack([N_1, N_2, N_3], dim=-1)

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        return torch.stack(
            [
                torch.stack(
                    [
                        0.5 * (2 * xi[..., 0] - 1),
                        0.5 * (2 * xi[..., 0] + 1),
                        -2 * xi[..., 0],
                    ],
                    dim=-1,
                )
            ],
            dim=xi.dim() - 1,
        )

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor([1.0, 1.0])

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor([[-1.0 / sqrt(3.0)], [1.0 / sqrt(3.0)]])

    @classmethod
    def plot(cls, n_points: int = 100):
        import matplotlib.pyplot as plt

        # Compute shape functions at evenly spaced points in reference space
        xi = torch.linspace(-1.0, 1.0, n_points).unsqueeze(-1)
        N = cls.N(xi)

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4))
        for i in range(cls.nodes):
            ax.plot(xi, N[:, i], linewidth=2.0, label=f"$N_{i}$")
        ax.set_xlabel("$\\xi$")
        ax.set_ylabel("$N_i(\\xi)$")
        ax.grid(alpha=0.3)
        ax.legend()

        # Save plot to docs/images directory
        root = Path(__file__).resolve().parents[2] / "docs" / "images"
        save_path = root / "Bar2_shape_functions.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")


class Tria1(Element):
    r"""Three-node linear triangle element.

    Notes:
        Node ordering:

            2
            | \
            |   \
            0 --- 1
    """

    iso_volume = 0.5
    iso_dim = 2
    nodes = 3
    meshio_type = "triangle"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = 1.0 - xi[..., 0] - xi[..., 1]
        N_2 = xi[..., 0]
        N_3 = xi[..., 1]
        return torch.stack([N_1, N_2, N_3], dim=-1)

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        if xi.dim() == 1:
            return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        else:
            N = xi.shape[0]
            return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]).repeat(N, 1, 1)

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor([0.5])

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor([[1.0 / 3.0, 1.0 / 3.0]])

    @classmethod
    def plot(cls, n_points: int = 30):
        import matplotlib.pyplot as plt

        # Sample inside triangular reference domain (ξ₁ ≥ 0, ξ₂ ≥ 0, ξ₁+ξ₂ ≤ 1)
        t = np.linspace(0.0, 1.0, n_points)
        xi1, xi2 = np.meshgrid(t, t)
        mask = (xi1 + xi2) <= 1.0
        xi1f, xi2f = xi1[mask], xi2[mask]
        xi = torch.tensor(np.stack([xi1f, xi2f], axis=-1), dtype=torch.float32)
        N = cls.N(xi).detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(10, 4), subplot_kw={"projection": "3d"})
        for i, ax in enumerate(axes):
            ax.plot_trisurf(xi1f, xi2f, N[:, i], color=f"C{i}", alpha=0.9)
            ax.set_xlabel("$\\xi_1$")
            ax.set_ylabel("$\\xi_2$")
            ax.set_title(f"$N_{i}$")

        fig.tight_layout()
        root = Path(__file__).resolve().parents[2] / "docs" / "images"
        fig.savefig(root / "Tria1_shape_functions.png", dpi=200, bbox_inches="tight")


class Tria2(Tria1):
    r"""Six-node quadratic triangle element with midside nodes.

    Notes:
        Node ordering:

            2
            | \
            5   4
            |     \
            0 - 3 - 1
    """

    nodes = 6
    meshio_type = "triangle6"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.0, 0.5],
            ]
        )

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = (1 - xi[..., 0] - xi[..., 1]) * (1 - 2 * xi[..., 0] - 2 * xi[..., 1])
        N_2 = xi[..., 0] * (2 * xi[..., 0] - 1)
        N_3 = xi[..., 1] * (2 * xi[..., 1] - 1)
        N_4 = 4 * xi[..., 0] * (1 - xi[..., 0] - xi[..., 1])
        N_5 = 4 * xi[..., 0] * xi[..., 1]
        N_6 = 4 * xi[..., 1] * (1 - xi[..., 0] - xi[..., 1])
        return torch.stack([N_1, N_2, N_3, N_4, N_5, N_6], dim=-1)

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        zeros = torch.zeros_like(xi[..., 0])
        return torch.stack(
            [
                torch.stack(
                    [
                        4 * xi[..., 0] + 4 * xi[..., 1] - 3,
                        4 * xi[..., 0] - 1,
                        zeros,
                        -4 * (2 * xi[..., 0] + xi[..., 1] - 1),
                        4 * xi[..., 1],
                        -4 * xi[..., 1],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        4 * xi[..., 0] + 4 * xi[..., 1] - 3,
                        zeros,
                        4 * xi[..., 1] - 1,
                        -4 * xi[..., 0],
                        4 * xi[..., 0],
                        -4 * (xi[..., 0] + 2 * xi[..., 1] - 1),
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]])

    @classmethod
    def plot(cls, n_points: int = 30):
        import matplotlib.pyplot as plt

        # Sample inside triangular reference domain (ξ₁ ≥ 0, ξ₂ ≥ 0, ξ₁+ξ₂ ≤ 1)
        t = np.linspace(0.0, 1.0, n_points)
        xi1, xi2 = np.meshgrid(t, t)
        mask = (xi1 + xi2) <= 1.0
        xi1f, xi2f = xi1[mask], xi2[mask]
        xi = torch.tensor(np.stack([xi1f, xi2f], axis=-1), dtype=torch.float32)
        N = cls.N(xi).detach().cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(10, 8), subplot_kw={"projection": "3d"})
        for i, ax in enumerate(axes.ravel()):
            ax.plot_trisurf(xi1f, xi2f, N[:, i], color=f"C{i}", alpha=0.9)
            ax.set_xlabel("$\\xi_1$")
            ax.set_ylabel("$\\xi_2$")
            ax.set_title(f"$N_{i}$")

        fig.tight_layout()
        root = Path(__file__).resolve().parents[2] / "docs" / "images"
        fig.savefig(root / "Tria2_shape_functions.png", dpi=200, bbox_inches="tight")


class Quad1(Element):
    """Four-node bilinear quadrilateral element.

    Notes:
        Node ordering:

            3 ---- 2
            |      |
            |      |
            0 ---- 1
    """

    iso_volume = 4.0
    iso_dim = 2
    nodes = 4
    meshio_type = "quad"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1])
        N_2 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1])
        N_3 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1])
        N_4 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1])
        return 0.25 * torch.stack([N_1, N_2, N_3, N_4], dim=-1)

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        return 0.25 * torch.stack(
            [
                torch.stack(
                    [
                        -(1 - xi[..., 1]),
                        (1 - xi[..., 1]),
                        (1.0 + xi[..., 1]),
                        -(1.0 + xi[..., 1]),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(1 - xi[..., 0]),
                        -(1 + xi[..., 0]),
                        (1.0 + xi[..., 0]),
                        (1.0 - xi[..., 0]),
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor([1, 1, 1, 1])

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
                for xi_2 in [-1, 1]
                for xi_1 in [-1, 1]
            ]
        )

    @classmethod
    def plot(cls, n_points: int = 30):
        import matplotlib.pyplot as plt

        # Sample on the square reference domain (ξ₁, ξ₂ ∈ [-1, 1])
        t = np.linspace(-1.0, 1.0, n_points)
        xi1, xi2 = np.meshgrid(t, t)
        xi = torch.tensor(
            np.stack([xi1.ravel(), xi2.ravel()], axis=-1), dtype=torch.float32
        )
        N = cls.N(xi).detach().cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={"projection": "3d"})
        for i, ax in enumerate(axes.ravel()):
            ax.plot_surface(
                xi1,
                xi2,
                N[:, i].reshape(n_points, n_points),
                color=f"C{i}",
                alpha=0.9,
                linewidth=0,
            )
            ax.set_xlabel("$\\xi_1$")
            ax.set_ylabel("$\\xi_2$")
            ax.set_title(f"$N_{i}$")

        fig.tight_layout()
        root = Path(__file__).resolve().parents[2] / "docs" / "images"
        fig.savefig(root / "Quad1_shape_functions.png", dpi=200, bbox_inches="tight")


class Quad2(Quad1):
    """Eight-node quadratic quadrilateral element with midside nodes.

    Notes:
        Node ordering:

            3 -- 6 -- 2
            |         |
            7         5
            |         |
            0 -- 4 -- 1
    """

    nodes = 8
    meshio_type = "quad8"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor(
            [
                # 4 corner nodes
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                # 4 midside nodes
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = -(1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 0] + xi[..., 1])
        N_2 = -(1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 - xi[..., 0] + xi[..., 1])
        N_3 = -(1 + xi[..., 0]) * (1 + xi[..., 1]) * (1 - xi[..., 0] - xi[..., 1])
        N_4 = -(1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 + xi[..., 0] - xi[..., 1])
        N_5 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 0])
        N_6 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 1])
        N_7 = 2 * (1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 + xi[..., 0])
        N_8 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 + xi[..., 1])
        return 0.25 * torch.stack([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8], dim=-1)

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        return 0.25 * torch.stack(
            [
                torch.stack(
                    [
                        -(xi[..., 1] - 1) * (2 * xi[..., 0] + xi[..., 1]),
                        -(xi[..., 1] - 1) * (2 * xi[..., 0] - xi[..., 1]),
                        +(xi[..., 1] + 1) * (2 * xi[..., 0] + xi[..., 1]),
                        +(xi[..., 1] + 1) * (2 * xi[..., 0] - xi[..., 1]),
                        +4 * xi[..., 0] * (xi[..., 1] - 1),
                        +2 - 2 * xi[..., 1] ** 2,
                        -4 * xi[..., 0] * (xi[..., 1] + 1),
                        -2 + 2 * xi[..., 1] ** 2,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(xi[..., 0] - 1) * (xi[..., 0] + 2 * xi[..., 1]),
                        -(xi[..., 0] + 1) * (xi[..., 0] - 2 * xi[..., 1]),
                        +(xi[..., 0] + 1) * (xi[..., 0] + 2 * xi[..., 1]),
                        +(xi[..., 0] - 1) * (xi[..., 0] - 2 * xi[..., 1]),
                        -2 + 2 * xi[..., 0] ** 2,
                        -4 * xi[..., 1] * (xi[..., 0] + 1),
                        +2 - 2 * xi[..., 0] ** 2,
                        +4 * (xi[..., 0] - 1) * xi[..., 1],
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor([1, 1, 1, 1])

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
                for xi_2 in [-1, 1]
                for xi_1 in [-1, 1]
            ]
        )

    @classmethod
    def plot(cls, n_points: int = 30):
        import matplotlib.pyplot as plt

        # Sample on the square reference domain (ξ₁, ξ₂ ∈ [-1, 1])
        t = np.linspace(-1.0, 1.0, n_points)
        xi1, xi2 = np.meshgrid(t, t)
        xi = torch.tensor(
            np.stack([xi1.ravel(), xi2.ravel()], axis=-1), dtype=torch.float32
        )
        N = cls.N(xi).detach().cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(14, 8), subplot_kw={"projection": "3d"})
        for i, ax in enumerate(axes.ravel()):
            ax.plot_surface(
                xi1,
                xi2,
                N[:, i].reshape(n_points, n_points),
                color=f"C{i}",
                alpha=0.9,
                linewidth=0,
            )
            ax.set_xlabel("$\\xi_1$")
            ax.set_ylabel("$\\xi_2$")
            ax.set_title(f"$N_{i}$")

        fig.tight_layout()
        root = Path(__file__).resolve().parents[2] / "docs" / "images"
        fig.savefig(root / "Quad2_shape_functions.png", dpi=200, bbox_inches="tight")


class Tetra1(Element):
    r"""Four-node linear tetrahedral element.

    Notes:
        Node ordering:

                3
               /|\
              / | \
             /  |  \
            0---|---1
             \  |  /
              \ | /
               \|/
                2
    """

    iso_volume = 1.0 / 6.0
    iso_dim = 3
    nodes = 4
    meshio_type = "tetra"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = 1.0 - xi[..., 0] - xi[..., 1] - xi[..., 2]
        N_2 = xi[..., 0]
        N_3 = xi[..., 1]
        N_4 = xi[..., 2]
        return torch.stack([N_1, N_2, N_3, N_4], dim=-1)

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        if xi.dim() == 1:
            return torch.tensor(
                [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]]
            )
        else:
            N = xi.shape[0]
            return torch.tensor(
                [[-1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]]
            ).repeat(N, 1, 1)

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor([1.0 / 6.0])

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor([[0.25, 0.25, 0.25]])


class Tetra2(Tetra1):
    r"""Ten-node quadratic tetrahedral element with midside nodes.

    Notes:
        Node ordering:

                3
               /|\
             7/ | \8
             /  |  \
            0---4---1
             \  |  /
            6 \ | / 5
               \|/
                2
    """

    nodes = 10
    meshio_type = "tetra10"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor(
            [
                # 4 corner nodes
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                # 6 midside nodes
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ]
        )

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = (1.0 - xi[..., 0] - xi[..., 1] - xi[..., 2]) * (
            2 * (1.0 - xi[..., 0] - xi[..., 1] - xi[..., 2]) - 1
        )
        N_2 = xi[..., 0] * (2 * xi[..., 0] - 1)
        N_3 = xi[..., 1] * (2 * xi[..., 1] - 1)
        N_4 = xi[..., 2] * (2 * xi[..., 2] - 1)
        N_5 = 4 * xi[..., 0] * (1 - xi[..., 0] - xi[..., 1] - xi[..., 2])
        N_6 = 4 * xi[..., 0] * xi[..., 1]
        N_7 = 4 * xi[..., 1] * (1 - xi[..., 0] - xi[..., 1] - xi[..., 2])
        N_8 = 4 * xi[..., 2] * (1 - xi[..., 0] - xi[..., 1] - xi[..., 2])
        N_9 = 4 * xi[..., 0] * xi[..., 2]
        N_10 = 4 * xi[..., 1] * xi[..., 2]
        return torch.stack([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8, N_9, N_10], dim=-1)

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        zeros = torch.zeros_like(xi[..., 0])
        return torch.stack(
            [
                torch.stack(
                    [
                        4 * (xi[..., 0] + xi[..., 1] + xi[..., 2]) - 3,
                        4 * xi[..., 0] - 1,
                        zeros,
                        zeros,
                        -4 * (2 * xi[..., 0] + xi[..., 1] + xi[..., 2] - 1),
                        4 * xi[..., 1],
                        -4 * xi[..., 1],
                        -4 * xi[..., 2],
                        4 * xi[..., 2],
                        zeros,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        4 * (xi[..., 0] + xi[..., 1] + xi[..., 2]) - 3,
                        zeros,
                        4 * xi[..., 1] - 1,
                        zeros,
                        -4 * xi[..., 0],
                        4 * xi[..., 0],
                        -4 * (xi[..., 0] + 2 * xi[..., 1] + xi[..., 2] - 1),
                        -4 * xi[..., 2],
                        zeros,
                        4 * xi[..., 2],
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        4 * (xi[..., 0] + xi[..., 1] + xi[..., 2]) - 3,
                        zeros,
                        zeros,
                        4 * xi[..., 2] - 1,
                        -4 * xi[..., 0],
                        zeros,
                        -4 * xi[..., 1],
                        -4 * (xi[..., 0] + xi[..., 1] + 2 * xi[..., 2] - 1),
                        4 * xi[..., 0],
                        4 * xi[..., 1],
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor([0.041666667, 0.041666667, 0.041666667, 0.041666667])

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor(
            [
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660],
            ]
        )


class Hexa1(Element):
    r"""Eight-node trilinear hexahedral element.

    Notes:
        Node ordering:

              7 ---- 6
             /|     /|
            4 ---- 5 |
            | 3 --|- 2
            |/     |/
            0 ---- 1
    """

    iso_volume = 8.0
    iso_dim = 3
    nodes = 8
    meshio_type = "hexahedron"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
        )

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0), xi_3 / sqrt(3.0)]
                for xi_3 in [-1.0, 1.0]
                for xi_2 in [-1.0, 1.0]
                for xi_1 in [-1.0, 1.0]
            ]
        )

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 - xi[..., 2])
        N_2 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 - xi[..., 2])
        N_3 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 - xi[..., 2])
        N_4 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 - xi[..., 2])
        N_5 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 + xi[..., 2])
        N_6 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1]) * (1.0 + xi[..., 2])
        N_7 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 + xi[..., 2])
        N_8 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1]) * (1.0 + xi[..., 2])
        return 0.125 * torch.stack([N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8], dim=-1)

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        return 0.125 * torch.stack(
            [
                torch.stack(
                    [
                        -(1.0 - xi[..., 1]) * (1.0 - xi[..., 2]),
                        (1.0 - xi[..., 1]) * (1.0 - xi[..., 2]),
                        (1.0 + xi[..., 1]) * (1.0 - xi[..., 2]),
                        -(1.0 + xi[..., 1]) * (1.0 - xi[..., 2]),
                        -(1.0 - xi[..., 1]) * (1.0 + xi[..., 2]),
                        (1.0 - xi[..., 1]) * (1.0 + xi[..., 2]),
                        (1.0 + xi[..., 1]) * (1.0 + xi[..., 2]),
                        -(1.0 + xi[..., 1]) * (1.0 + xi[..., 2]),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(1.0 - xi[..., 0]) * (1.0 - xi[..., 2]),
                        -(1.0 + xi[..., 0]) * (1.0 - xi[..., 2]),
                        (1.0 + xi[..., 0]) * (1.0 - xi[..., 2]),
                        (1.0 - xi[..., 0]) * (1.0 - xi[..., 2]),
                        -(1.0 - xi[..., 0]) * (1.0 + xi[..., 2]),
                        -(1.0 + xi[..., 0]) * (1.0 + xi[..., 2]),
                        (1.0 + xi[..., 0]) * (1.0 + xi[..., 2]),
                        (1.0 - xi[..., 0]) * (1.0 + xi[..., 2]),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        -(1.0 - xi[..., 0]) * (1.0 - xi[..., 1]),
                        -(1.0 + xi[..., 0]) * (1.0 - xi[..., 1]),
                        -(1.0 + xi[..., 0]) * (1.0 + xi[..., 1]),
                        -(1.0 - xi[..., 0]) * (1.0 + xi[..., 1]),
                        (1.0 - xi[..., 0]) * (1.0 - xi[..., 1]),
                        (1.0 + xi[..., 0]) * (1.0 - xi[..., 1]),
                        (1.0 + xi[..., 0]) * (1.0 + xi[..., 1]),
                        (1.0 - xi[..., 0]) * (1.0 + xi[..., 1]),
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )


class Hexa2(Hexa1):
    r"""Twenty-node quadratic serendipity hexahedral element.

    Notes:
        Node ordering:

                7 -- 14 --  6
               /|          /|
             15 19       13 18
             /  |        /  |
            4 -- 12 -- 5    |
            |   3 -- 10 |-- 2
            16 /        17 /
            | 11        | 9
            |/          |/
            0 --  8 --  1
    """

    iso_dim = 3
    nodes = 20
    meshio_type = "hexahedron20"

    @classproperty
    def iso_coords(cls) -> Tensor:
        return torch.tensor(
            [
                # 8 corner nodes
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                # 12 midside nodes
                [0.0, -1.0, -1.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, -1.0],
                [-1.0, 0.0, -1.0],
                [0.0, -1.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]
        )

    @classmethod
    def N(cls, xi: Tensor) -> Tensor:
        N_1 = (
            (1 - xi[..., 0])
            * (1 - xi[..., 1])
            * (1 - xi[..., 2])
            * (-xi[..., 0] - xi[..., 1] - xi[..., 2] - 2)
        )
        N_2 = (
            (1 + xi[..., 0])
            * (1 - xi[..., 1])
            * (1 - xi[..., 2])
            * (+xi[..., 0] - xi[..., 1] - xi[..., 2] - 2)
        )
        N_3 = (
            (1 + xi[..., 0])
            * (1 + xi[..., 1])
            * (1 - xi[..., 2])
            * (+xi[..., 0] + xi[..., 1] - xi[..., 2] - 2)
        )
        N_4 = (
            (1 - xi[..., 0])
            * (1 + xi[..., 1])
            * (1 - xi[..., 2])
            * (-xi[..., 0] + xi[..., 1] - xi[..., 2] - 2)
        )

        N_5 = (
            (1 - xi[..., 0])
            * (1 - xi[..., 1])
            * (1 + xi[..., 2])
            * (-xi[..., 0] - xi[..., 1] + xi[..., 2] - 2)
        )
        N_6 = (
            (1 + xi[..., 0])
            * (1 - xi[..., 1])
            * (1 + xi[..., 2])
            * (+xi[..., 0] - xi[..., 1] + xi[..., 2] - 2)
        )
        N_7 = (
            (1 + xi[..., 0])
            * (1 + xi[..., 1])
            * (1 + xi[..., 2])
            * (+xi[..., 0] + xi[..., 1] + xi[..., 2] - 2)
        )
        N_8 = (
            (1 - xi[..., 0])
            * (1 + xi[..., 1])
            * (1 + xi[..., 2])
            * (-xi[..., 0] + xi[..., 1] + xi[..., 2] - 2)
        )
        N_9 = 2 * (1 - xi[..., 0] ** 2) * (1 - xi[..., 1]) * (1 - xi[..., 2])
        N_10 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1] ** 2) * (1 - xi[..., 2])
        N_11 = 2 * (1 - xi[..., 0] ** 2) * (1 + xi[..., 1]) * (1 - xi[..., 2])
        N_12 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1] ** 2) * (1 - xi[..., 2])
        N_13 = 2 * (1 - xi[..., 0] ** 2) * (1 - xi[..., 1]) * (1 + xi[..., 2])
        N_14 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1] ** 2) * (1 + xi[..., 2])
        N_15 = 2 * (1 - xi[..., 0] ** 2) * (1 + xi[..., 1]) * (1 + xi[..., 2])
        N_16 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1] ** 2) * (1 + xi[..., 2])
        N_17 = 2 * (1 - xi[..., 0]) * (1 - xi[..., 1]) * (1 - xi[..., 2] ** 2)
        N_18 = 2 * (1 + xi[..., 0]) * (1 - xi[..., 1]) * (1 - xi[..., 2] ** 2)
        N_19 = 2 * (1 + xi[..., 0]) * (1 + xi[..., 1]) * (1 - xi[..., 2] ** 2)
        N_20 = 2 * (1 - xi[..., 0]) * (1 + xi[..., 1]) * (1 - xi[..., 2] ** 2)
        return 0.125 * torch.stack(
            [
                N_1,
                N_2,
                N_3,
                N_4,
                N_5,
                N_6,
                N_7,
                N_8,
                N_9,
                N_10,
                N_11,
                N_12,
                N_13,
                N_14,
                N_15,
                N_16,
                N_17,
                N_18,
                N_19,
                N_20,
            ],
            dim=-1,
        )

    @classmethod
    def B(cls, xi: Tensor) -> Tensor:
        return 0.125 * torch.stack(
            [
                torch.stack(
                    [
                        +(xi[..., 1] - 1)
                        * (xi[..., 2] - 1)
                        * (+2 * xi[..., 0] + xi[..., 1] + xi[..., 2] + 1),
                        -(xi[..., 1] - 1)
                        * (xi[..., 2] - 1)
                        * (-2 * xi[..., 0] + xi[..., 1] + xi[..., 2] + 1),
                        -(xi[..., 1] + 1)
                        * (xi[..., 2] - 1)
                        * (+2 * xi[..., 0] + xi[..., 1] - xi[..., 2] - 1),
                        +(xi[..., 1] + 1)
                        * (xi[..., 2] - 1)
                        * (-2 * xi[..., 0] + xi[..., 1] - xi[..., 2] - 1),
                        -(xi[..., 1] - 1)
                        * (xi[..., 2] + 1)
                        * (+2 * xi[..., 0] + xi[..., 1] - xi[..., 2] + 1),
                        +(xi[..., 1] - 1)
                        * (xi[..., 2] + 1)
                        * (-2 * xi[..., 0] + xi[..., 1] - xi[..., 2] + 1),
                        +(xi[..., 1] + 1)
                        * (xi[..., 2] + 1)
                        * (+2 * xi[..., 0] + xi[..., 1] + xi[..., 2] - 1),
                        -(xi[..., 1] + 1)
                        * (xi[..., 2] + 1)
                        * (-2 * xi[..., 0] + xi[..., 1] + xi[..., 2] - 1),
                        -4 * xi[..., 0] * (xi[..., 1] - 1) * (xi[..., 2] - 1),
                        +2 * (xi[..., 1] ** 2 - 1) * (xi[..., 2] - 1),
                        +4 * xi[..., 0] * (xi[..., 1] + 1) * (xi[..., 2] - 1),
                        -2 * (xi[..., 1] ** 2 - 1) * (xi[..., 2] - 1),
                        +4 * xi[..., 0] * (xi[..., 1] - 1) * (xi[..., 2] + 1),
                        -2 * (xi[..., 1] ** 2 - 1) * (xi[..., 2] + 1),
                        -4 * xi[..., 0] * (xi[..., 1] + 1) * (xi[..., 2] + 1),
                        +2 * (xi[..., 1] ** 2 - 1) * (xi[..., 2] + 1),
                        -2 * (xi[..., 1] - 1) * (xi[..., 2] ** 2 - 1),
                        +2 * (xi[..., 1] - 1) * (xi[..., 2] ** 2 - 1),
                        -2 * (xi[..., 1] + 1) * (xi[..., 2] ** 2 - 1),
                        +2 * (xi[..., 1] + 1) * (xi[..., 2] ** 2 - 1),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        +(xi[..., 0] - 1)
                        * (xi[..., 2] - 1)
                        * (xi[..., 0] + 2 * xi[..., 1] + xi[..., 2] + 1),
                        +(xi[..., 0] + 1)
                        * (xi[..., 2] - 1)
                        * (xi[..., 0] - 2 * xi[..., 1] - xi[..., 2] - 1),
                        -(xi[..., 0] + 1)
                        * (xi[..., 2] - 1)
                        * (xi[..., 0] + 2 * xi[..., 1] - xi[..., 2] - 1),
                        -(xi[..., 0] - 1)
                        * (xi[..., 2] - 1)
                        * (xi[..., 0] - 2 * xi[..., 1] + xi[..., 2] + 1),
                        -(xi[..., 0] - 1)
                        * (xi[..., 2] + 1)
                        * (xi[..., 0] + 2 * xi[..., 1] - xi[..., 2] + 1),
                        -(xi[..., 0] + 1)
                        * (xi[..., 2] + 1)
                        * (xi[..., 0] - 2 * xi[..., 1] + xi[..., 2] - 1),
                        +(xi[..., 0] + 1)
                        * (xi[..., 2] + 1)
                        * (xi[..., 0] + 2 * xi[..., 1] + xi[..., 2] - 1),
                        +(xi[..., 0] - 1)
                        * (xi[..., 2] + 1)
                        * (xi[..., 0] - 2 * xi[..., 1] - xi[..., 2] + 1),
                        -2 * (xi[..., 0] ** 2 - 1) * (xi[..., 2] - 1),
                        +4 * xi[..., 1] * (xi[..., 0] + 1) * (xi[..., 2] - 1),
                        +2 * (xi[..., 0] ** 2 - 1) * (xi[..., 2] - 1),
                        -4 * xi[..., 1] * (xi[..., 0] - 1) * (xi[..., 2] - 1),
                        +2 * (xi[..., 0] ** 2 - 1) * (xi[..., 2] + 1),
                        -4 * xi[..., 1] * (xi[..., 0] + 1) * (xi[..., 2] + 1),
                        -2 * (xi[..., 0] ** 2 - 1) * (xi[..., 2] + 1),
                        +4 * xi[..., 1] * (xi[..., 0] - 1) * (xi[..., 2] + 1),
                        -2 * (xi[..., 0] - 1) * (xi[..., 2] ** 2 - 1),
                        +2 * (xi[..., 0] + 1) * (xi[..., 2] ** 2 - 1),
                        -2 * (xi[..., 0] + 1) * (xi[..., 2] ** 2 - 1),
                        +2 * (xi[..., 0] - 1) * (xi[..., 2] ** 2 - 1),
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        +(xi[..., 0] - 1)
                        * (xi[..., 1] - 1)
                        * (xi[..., 0] + xi[..., 1] + 2 * xi[..., 2] + 1),
                        +(xi[..., 0] + 1)
                        * (xi[..., 1] - 1)
                        * (xi[..., 0] - xi[..., 1] - 2 * xi[..., 2] - 1),
                        -(xi[..., 0] + 1)
                        * (xi[..., 1] + 1)
                        * (xi[..., 0] + xi[..., 1] - 2 * xi[..., 2] - 1),
                        -(xi[..., 0] - 1)
                        * (xi[..., 1] + 1)
                        * (xi[..., 0] - xi[..., 1] + 2 * xi[..., 2] + 1),
                        -(xi[..., 0] - 1)
                        * (xi[..., 1] - 1)
                        * (xi[..., 0] + xi[..., 1] - 2 * xi[..., 2] + 1),
                        -(xi[..., 0] + 1)
                        * (xi[..., 1] - 1)
                        * (xi[..., 0] - xi[..., 1] + 2 * xi[..., 2] - 1),
                        +(xi[..., 0] + 1)
                        * (xi[..., 1] + 1)
                        * (xi[..., 0] + xi[..., 1] + 2 * xi[..., 2] - 1),
                        +(xi[..., 0] - 1)
                        * (xi[..., 1] + 1)
                        * (xi[..., 0] - xi[..., 1] - 2 * xi[..., 2] + 1),
                        -2 * (xi[..., 0] ** 2 - 1) * (xi[..., 1] - 1),
                        +2 * (xi[..., 0] + 1) * (xi[..., 1] ** 2 - 1),
                        +2 * (xi[..., 0] ** 2 - 1) * (xi[..., 1] + 1),
                        -2 * (xi[..., 0] - 1) * (xi[..., 1] ** 2 - 1),
                        +2 * (xi[..., 0] ** 2 - 1) * (xi[..., 1] - 1),
                        -2 * (xi[..., 0] + 1) * (xi[..., 1] ** 2 - 1),
                        -2 * (xi[..., 0] ** 2 - 1) * (xi[..., 1] + 1),
                        +2 * (xi[..., 0] - 1) * (xi[..., 1] ** 2 - 1),
                        -4 * (xi[..., 0] - 1) * (xi[..., 1] - 1) * xi[..., 2],
                        +4 * (xi[..., 0] + 1) * (xi[..., 1] - 1) * xi[..., 2],
                        -4 * (xi[..., 0] + 1) * (xi[..., 1] + 1) * xi[..., 2],
                        +4 * (xi[..., 0] - 1) * (xi[..., 1] + 1) * xi[..., 2],
                    ],
                    dim=-1,
                ),
            ],
            dim=xi.dim() - 1,
        )

    @classproperty
    def iweights(cls) -> Tensor:
        return torch.tensor(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )

    @classproperty
    def ipoints(cls) -> Tensor:
        return torch.tensor(
            [
                [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0), xi_3 / sqrt(3.0)]
                for xi_3 in [-1.0, 1.0]
                for xi_2 in [-1.0, 1.0]
                for xi_1 in [-1.0, 1.0]
            ]
        )


def linear_to_quadratic(nodes: Tensor, elements: Tensor) -> tuple[Tensor, Tensor]:
    """Convert supported linear meshes to quadratic meshes.

    Supported topologies are:

    - 2-node bars to 3-node bars
    - 3-node triangles to 6-node triangles
    - 4-node quadrilaterals to 8-node quadrilaterals
    - 4-node tetrahedra to 10-node tetrahedra
    - 8-node hexahedra to 20-node hexahedra

    New nodes are created at edge midpoints and appended to `nodes`.

    Args:
        nodes (Tensor): Nodal coordinates.
            *Shape:* `(n_nodes, dim)`.
        elements (Tensor): Connectivity of linear elements.
            *Shape:* `(n_elem, n_nodes_per_elem)`.

    Returns:
        new_nodes (Tensor): Extended nodal coordinates with edge midpoints.
            *Shape:* `(n_nodes + n_unique_edges, dim)`.
        new_elements (Tensor): Quadratic element connectivity.
            *Shape:* `(n_elem, n_quadratic_nodes_per_elem)`.
    """

    if elements.shape[1] == 2:
        # Bar1 element
        edges = torch.tensor([[0, 1]])
    elif elements.shape[1] == 3 and nodes.shape[1] == 2:
        # Tri1 element
        edges = torch.tensor([[0, 1], [1, 2], [2, 0]])
    elif elements.shape[1] == 4 and nodes.shape[1] == 2:
        # Quad1 element
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    elif elements.shape[1] == 4 and nodes.shape[1] == 3:
        # Tetra1 element
        edges = torch.tensor([[0, 1], [1, 2], [0, 2], [3, 0], [1, 3], [2, 3]])
    elif elements.shape[1] == 8 and nodes.shape[1] == 3:
        # Hexa1 element
        edges = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
        )
    else:
        raise Exception(
            "The element type is not supported for conversion to quadratic."
            "Maybe the element is already quadratic?"
        )

    # Vectorize edges for all elements and sort each pair with increasing node IDs
    edge_vector = elements[:, edges].reshape(-1, 2)
    edge_vector, _ = torch.sort(edge_vector, dim=1)

    # Find unique edges and get the inverse mapping
    if edge_vector.device.type == "cuda":
        # Default torch implementation for CUDA
        unique_edges, inv_indices = torch.unique(
            edge_vector, dim=0, return_inverse=True
        )
    else:
        # Numpy implementation for CPU (breaks gradient flow, but should not be needed)
        edge_vector_np = edge_vector.numpy()
        dt = np.dtype((np.void, edge_vector_np.dtype.itemsize * edge_vector.shape[1]))
        b = np.ascontiguousarray(edge_vector).view(dt)
        _, unique_indices, inv_indices_np = np.unique(
            b, return_index=True, return_inverse=True
        )
        unique_edges = edge_vector[unique_indices]
        inv_indices = torch.as_tensor(inv_indices_np)

    # Compute nodes
    mid_node_coords = (nodes[unique_edges[:, 0]] + nodes[unique_edges[:, 1]]) / 2.0
    new_nodes = torch.cat([nodes, mid_node_coords], dim=0)

    # Compute elements
    mid_node_map = inv_indices.view(elements.shape[0], -1) + nodes.shape[0]
    new_elements = torch.cat([elements, mid_node_map], dim=1)

    return new_nodes, new_elements
