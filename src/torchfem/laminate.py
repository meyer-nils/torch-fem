"""Laminate definition for layered shell elements.

A `Laminate` is a stacking sequence of (possibly rotated) plane-stress layers.
It owns the through-thickness integration stations and precomputes the
transverse shear stiffness and mass integrals; the `Shell` integrates the
section response (the ABD equivalent) over the stations during the analysis, so
it stays valid for nonlinear, state-bearing layers.

Following classical lamination theory, layers stack from ``z = -h/2`` to
``z = +h/2`` about the mid-plane and angles are measured from the element's
first local axis, counter-clockwise as in `planar_rotation`. `Shell` places the
reference surface within that stack through its `offset`.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from .materials import MechanicsMaterial
from .rotations import planar_rotation


class Laminate:
    """A stacking sequence of plane-stress layers for shell elements.

    Args:
        materials: Plane-stress mechanics materials, one per layer.
        thicknesses: Per-layer thicknesses. Each entry may be a scalar (constant
            over the mesh) or a tensor of shape `(n_elem,)`.
        angles: Per-layer orientation angles in radians, measured from the
            element's first local axis. Each entry may be a scalar or a tensor
            of shape `(n_elem,)`.
        n_simpson: Number of Simpson integration points used *per layer* through
            the thickness. Must be an odd integer (default `3`).
        symmetric: If `True`, the given layers are the half-stack (outer surface
            to mid-plane) and are mirrored to form the full laminate.

    Notes:
        - Layers may carry internal state (e.g. an elastoplastic metal ply); the
          state width is the per-layer maximum. Stations are integrated during
          the analysis, so nonlinear, state-bearing layers need no ABD matrices.
        - `Shell` binds the laminate to its mesh with `vectorize(...)`, and the
          bound section at `shell.section` rotates its `materials` by their
          `angles` and reads its stations, transverse shear stiffness and mass
          from its `thicknesses` whenever they are needed. A layer's material,
          angle or thickness may be reassigned there and may carry a gradient,
          as a design variable of an orientation or free-size optimization.
        - The stack is centered on its mid-plane. `Shell(offset=...)` moves the
          reference surface, adding the membrane-bending coupling it implies.
    """

    n_elem: int
    offset: Tensor

    @property
    def symmetric_tangent(self) -> bool:
        """Whether every layer has a symmetric tangent. See `Material`."""
        return all(material.symmetric_tangent for material in self.materials)

    @property
    def finite_strain(self) -> bool:
        """Whether any layer is formulated for finite strain. See `Material`."""
        return any(material.finite_strain for material in self.materials)

    def __init__(
        self,
        materials: Sequence[MechanicsMaterial],
        thicknesses: Sequence[float] | Sequence[Tensor] | Tensor,
        angles: Sequence[float] | Sequence[Tensor] | Tensor,
        n_simpson: int = 3,
        symmetric: bool = False,
    ):
        if not (len(materials) == len(thicknesses) == len(angles)):
            raise ValueError(
                "materials, thicknesses, and angles must have equal length."
            )
        if len(materials) == 0:
            raise ValueError("A laminate must contain at least one layer.")
        for material in materials:
            if not isinstance(material, MechanicsMaterial) or material.dim != 2:
                raise ValueError(
                    "A laminate layer must be a 2D mechanics material, not "
                    f"{type(material).__name__}."
                )
        if n_simpson % 2 == 0:
            raise ValueError("n_simpson must be an odd integer.")

        dtype = torch.get_default_dtype()
        self.materials = list(materials)
        self.thicknesses = [torch.as_tensor(t, dtype=dtype) for t in thicknesses]
        self.angles = [torch.as_tensor(a, dtype=dtype) for a in angles]

        # Mirror the half-stack about the mid-plane for a symmetric laminate.
        # Mirroring the canonical lists (not the union-typed inputs) keeps the
        # element types clean and also handles tensor inputs.
        if symmetric:
            self.materials += self.materials[::-1]
            self.thicknesses += self.thicknesses[::-1]
            self.angles += self.angles[::-1]

        self.n_layers = len(self.materials)
        self.n_simpson = n_simpson
        self.n_z = self.n_layers * n_simpson

        # The layer of each station, its position within the layer from 0 to 1,
        # and its Simpson weight within the layer, which sum to 1 per layer
        self.layer = torch.arange(self.n_layers).repeat_interleave(n_simpson)
        self.zeta = torch.linspace(0.0, 1.0, n_simpson).repeat(self.n_layers)
        weights = torch.ones(n_simpson)
        weights[1:-1:2] = 4.0
        weights[2:-2:2] = 2.0
        self.weights = weights.repeat(self.n_layers) / (3.0 * (n_simpson - 1))

        # State width is the per-layer maximum; each layer's `step` touches only
        # the slots it needs, so mixing elastic and state-bearing layers is free.
        self.n_state = max(m.n_state for m in self.materials)

    def __repr__(self) -> str:
        return (
            f"<torch-fem laminate ({self.n_layers} layers, "
            f"{self.n_z} integration points)>"
        )

    def vectorize(self, n_elem: int, offset: Tensor) -> Laminate:
        """Return a copy bound to `n_elem` elements and the reference surface `offset`.

        Each layer material is vectorized and each layer thickness is expanded to
        one value per element.
        """
        for material in self.materials:
            if not hasattr(material, "G") and not hasattr(material, "G_13"):
                raise ValueError(
                    "A laminate layer must define transverse shear moduli "
                    "('G_13'/'G_23') or a shear modulus 'G'."
                )
        new = copy.copy(self)
        new.n_elem = n_elem
        new.offset = offset
        new.angles = list(self.angles)  # its own, so a reassigned angle stays here
        new.materials = [m.vectorize(n_elem) for m in self.materials]
        new.thicknesses = [
            t.expand(n_elem) if t.dim() == 0 else t for t in self.thicknesses
        ]
        return new

    @property
    def materials_per_station(self) -> list[MechanicsMaterial]:
        """The material at each station, rotated by the angle of its layer."""
        layers = [
            cast(MechanicsMaterial, m.rotate(planar_rotation(a)))
            for m, a in zip(self.materials, self.angles)
        ]
        return [layers[k] for k in self.layer.tolist()]

    @property
    def thickness(self) -> Tensor:
        """Total thickness per element."""
        return torch.stack(self.thicknesses).sum(dim=0)

    def _interfaces(self) -> tuple[Tensor, Tensor, Tensor]:
        """Layer thicknesses and the bottom and top of each layer, measured from
        the reference surface, each with shape [n_layers, n_elem]."""
        t = torch.stack(self.thicknesses)
        top = -(0.5 + self.offset) * t.sum(dim=0) + torch.cumsum(t, dim=0)
        return t, top - t, top

    @property
    def z(self) -> Tensor:
        """Through-thickness coordinate of each station, shape [n_z, n_elem]."""
        t, bot, _ = self._interfaces()
        return bot[self.layer] + self.zeta[:, None] * t[self.layer]

    @property
    def w(self) -> Tensor:
        """Integration weight of each station, shape [n_z, n_elem], so that
        `sum_j w_j f_j` approximates the integral of `f` over the thickness."""
        return self.weights[:, None] * torch.stack(self.thicknesses)[self.layer]

    @property
    def As(self) -> Tensor:
        """Transverse shear stiffness `sum_k R_k diag(G13, G23)_k R_k^T t_k`, with
        shape [n_elem, 2, 2]."""
        As = torch.zeros(self.n_elem, 2, 2)
        for m, angle, t in zip(self.materials, self.angles, self.thicknesses):
            # Read dynamically, as only some materials carry the moduli
            names = ("G_13", "G_23") if hasattr(m, "G_13") else ("G", "G")
            moduli = [torch.as_tensor(getattr(m, name)) for name in names]
            G = torch.diag_embed(torch.stack(moduli, dim=-1))
            R = planar_rotation(angle)
            As = (
                As
                + torch.einsum("...mi,...ij,...nj->...mn", R, G, R) * t[:, None, None]
            )
        return As

    @property
    def rho_h(self) -> Tensor:
        """Mass per area, the integral of the density over the thickness."""
        layers = zip(self.materials, self.thicknesses)
        return torch.stack([m.rho * t for m, t in layers]).sum(dim=0)

    @property
    def rho_zz(self) -> Tensor:
        """Rotational inertia per area, the integral of `rho z^2` over the thickness."""
        _, bot, top = self._interfaces()
        layers = zip(self.materials, top, bot)
        moments = [m.rho * (t**3 - b**3) / 3.0 for m, t, b in layers]
        return torch.stack(moments).sum(dim=0)

    def plot(self, ax=None):
        """Illustrate the stacking sequence.

        Each ply is drawn as a band through the thickness (height proportional
        to the ply thickness) with the ply angle annotated.

        Args:
            ax: Existing matplotlib axes to plot into.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        t = torch.stack(self.thicknesses)
        angles = torch.rad2deg(torch.stack(self.angles))
        width = t.sum().item()
        # Interfaces relative to the mid-plane (z = 0).
        z = torch.concatenate([torch.tensor([0.0]), torch.cumsum(t, 0)]) - t.sum() / 2

        # One color per material class and orientation
        layer_keys = [
            f"{type(m).__name__}_{angles[i]:.0f}°" for i, m in enumerate(self.materials)
        ]
        unique_combos = list(dict.fromkeys(layer_keys))
        cmap = plt.get_cmap("Pastel1")
        color = {name: cmap(i % 9) for i, name in enumerate(unique_combos)}

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 5))

        for k in range(self.n_layers):
            z0 = z[k].item()
            z1 = z[k + 1].item()
            ax.add_patch(
                Rectangle(
                    (0.0, z0),
                    width,
                    z1 - z0,
                    facecolor=color[layer_keys[k]],
                    edgecolor="black",
                    lw=2.0,
                )
            )
            ax.text(
                0.5 * width,
                0.5 * (z0 + z1),
                f"{angles[k]:.0f}°",
                va="center",
                ha="center",
                fontsize=8,
            )

        ax.set_xlim(0.0, 1.3 * width)
        ax.set_ylim(z[0].item(), z[-1].item())
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_ylabel("Stacking direction")
        ax.spines[["top", "right", "bottom"]].set_visible(False)
