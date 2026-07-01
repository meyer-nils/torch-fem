"""Laminate definition for layered shell elements.

A `Laminate` is a stacking sequence of (possibly rotated) plane-stress layers.
It owns the through-thickness integration stations and precomputes the
transverse shear stiffness and mass integrals; the `Shell` integrates the
section response (the ABD equivalent) over the stations during the analysis, so
it stays valid for nonlinear, state-bearing layers.

Following classical lamination theory, layers stack from ``z = -h/2`` to
``z = +h/2`` (before any `offset`) and angles are measured from the element's
first local axis, counter-clockwise as in `planar_rotation`.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor

from .materials import Material
from .rotations import planar_rotation


class Laminate:
    """A stacking sequence of plane-stress layers for shell elements.

    Args:
        materials: List of plane-stress materials, one per layer.
        thicknesses: Per-layer thicknesses. Each entry may be a scalar (constant
            over the mesh) or a tensor of shape `(n_elem,)`.
        angles: Per-layer orientation angles in radians, measured from the
            element's first local axis. Each entry may be a scalar or a tensor
            of shape `(n_elem,)`.
        n_simpson: Number of Simpson integration points used *per layer* through
            the thickness. Must be an odd integer (default `3`).
        offset: Reference-surface position within the laminate, as a fraction of
            thickness from the mid-plane (`0.0` mid, `+0.5` top, `-0.5` bottom;
            the strings `"mid"`/`"top"`/`"bottom"` also work). Matches ABAQUS
            `*Shell Section, OFFSET`.
        symmetric: If `True`, the given layers are the half-stack (outer surface
            to mid-plane) and are mirrored to form the full laminate.

    Notes:
        - Layers may carry internal state (e.g. an elastoplastic metal ply); the
          state width is the per-layer maximum. Stations are integrated during
          the analysis, so nonlinear, state-bearing layers need no ABD matrices.
        - The laminate behaves like a `Material` (`is_vectorized`, `vectorize`,
          `n_state`), so it can be passed straight to `Shell`.
        - A nonzero `offset` shifts the stations, adding the membrane-bending
          coupling the offset implies (so an offset symmetric stack couples).
    """

    _OFFSET_ALIASES = {"mid": 0.0, "top": 0.5, "bottom": -0.5}

    def __init__(
        self,
        materials: list[Material],
        thicknesses: list[float] | list[Tensor] | Tensor,
        angles: list[float] | list[Tensor] | Tensor,
        n_simpson: int = 3,
        offset: float | str = 0.0,
        symmetric: bool = False,
    ):
        if not (len(materials) == len(thicknesses) == len(angles)):
            raise ValueError(
                "materials, thicknesses, and angles must have equal length."
            )
        if len(materials) == 0:
            raise ValueError("A laminate must contain at least one layer.")
        if n_simpson % 2 == 0:
            raise ValueError("n_simpson must be an odd integer.")

        if isinstance(offset, str):
            if offset not in self._OFFSET_ALIASES:
                raise ValueError(
                    f"offset string must be one of {list(self._OFFSET_ALIASES)}."
                )
            offset = self._OFFSET_ALIASES[offset]
        self._offset = float(offset)

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

        # Number of through-thickness integration stations
        self.n_z = self.n_layers * n_simpson

        # The laminate is considered vectorized once all layer materials are.
        self.is_vectorized = all(m.is_vectorized for m in self.materials)

        # State width is the per-layer maximum; each layer's `step` touches only
        # the slots it needs, so mixing elastic and state-bearing layers is free.
        self.n_state = max(m.n_state for m in self.materials)

    def __repr__(self) -> str:
        return (
            f"<torch-fem laminate ({self.n_layers} layers, "
            f"{self.n_z} integration points)>"
        )

    def vectorize(self, n_elem: int) -> Laminate:
        """Return a vectorized copy for `n_elem` elements.

        Each layer material is vectorized and rotated into the element frame, and
        the stations, transverse shear stiffness, and mass integrals are
        precomputed.
        """
        if self.is_vectorized:
            return self

        new = Laminate.__new__(Laminate)
        new.n_simpson = self.n_simpson
        new.n_layers = self.n_layers
        new.n_z = self.n_z
        new.n_state = self.n_state
        new.angles = self.angles
        new._offset = self._offset
        new.is_vectorized = True

        # Deep-copy each layer so reusing one material across layers (e.g. the
        # same ply at different angles) does not alias after rotation.
        new.materials = []
        new.thicknesses = []
        for mat, ang, t in zip(self.materials, self.angles, self.thicknesses):
            m = copy.deepcopy(mat).vectorize(n_elem)
            m = m.rotate(planar_rotation(ang))
            new.materials.append(m)
            new.thicknesses.append(t.expand(n_elem) if t.dim() == 0 else t)

        new._build(n_elem)
        return new

    def _build(self, n_elem: int) -> None:
        """Precompute stations, transverse shear, and mass integrals."""
        self.n_elem = n_elem

        # Per-layer thickness [n_layers, n_elem]
        t = torch.stack(self.thicknesses, dim=0)
        self.thickness = t.sum(dim=0)

        # Interface coordinates from the reference surface (z = 0); the offset
        # shifts the stack so the reference sits at the requested fraction.
        z_bot = -(0.5 + self._offset) * self.thickness
        layer_top = z_bot[None, :] + torch.cumsum(t, dim=0)
        layer_bot = layer_top - t
        self._layer_top = layer_top
        self._layer_bot = layer_bot

        # Simpson nodes (in [0, 1]) and weights (summing to 1) within a layer
        zeta = torch.linspace(0.0, 1.0, self.n_simpson)
        w_simpson = torch.ones(self.n_simpson)
        w_simpson[1:-1:2] = 4.0
        w_simpson[2:-2:2] = 2.0
        w_simpson *= 1.0 / (self.n_simpson - 1) / 3.0

        # Stations with absolute z and weights, so sum_j w_j f_j ~ integral f dz
        z_list, w_list, layer_idx = [], [], []
        for k in range(self.n_layers):
            z_k = layer_bot[k][None, :] + zeta[:, None] * t[k][None, :]
            w_k = w_simpson[:, None] * t[k][None, :]
            z_list.append(z_k)
            w_list.append(w_k)
            layer_idx += [k] * self.n_simpson
        self.z = torch.cat(z_list, dim=0)  # [n_z, n_elem]
        self.w = torch.cat(w_list, dim=0)  # [n_z, n_elem]
        self.layer = torch.tensor(layer_idx, dtype=torch.long)  # [n_z]

        # Material at each station, for the Shell integration loop
        self.materials_per_station = [self.materials[k] for k in layer_idx]

        # Transverse shear stiffness As = sum_k R_k diag(G13, G23)_k R_k^T t_k
        self.As = self._transverse_shear(t)

        # Mass integrals: rho_h = integral rho dz, rho_zz = integral rho z^2 dz
        rho_h = torch.zeros(n_elem)
        rho_zz = torch.zeros(n_elem)
        for k in range(self.n_layers):
            rho_k = torch.as_tensor(self.materials[k].rho)
            if rho_k.dim() == 0:
                rho_k = rho_k.expand(n_elem)
            rho_h = rho_h + rho_k * t[k]
            rho_zz = rho_zz + rho_k * (layer_top[k] ** 3 - layer_bot[k] ** 3) / 3.0
        self.rho_h = rho_h
        self.rho_zz = rho_zz

    def _transverse_shear(self, t: Tensor) -> Tensor:
        """Transverse shear stiffness ``sum_k G_k(rotated) * t_k``.

        Each layer's principal-axis shear moduli are rotated into the element
        frame as a second-order tensor.
        """
        n_elem = self.n_elem
        As = torch.zeros(n_elem, 2, 2)
        for k in range(self.n_layers):
            m = self.materials[k]
            if hasattr(m, "G_13"):
                g13 = torch.as_tensor(m.G_13)
                g23 = torch.as_tensor(m.G_23)
            elif hasattr(m, "G"):
                g13 = torch.as_tensor(m.G)
                g23 = g13
            else:
                raise ValueError(
                    "Layer material must define transverse shear moduli "
                    "('G_13'/'G_23') or a shear modulus 'G'."
                )
            g13 = g13.expand(n_elem) if g13.dim() == 0 else g13
            g23 = g23.expand(n_elem) if g23.dim() == 0 else g23
            Gs = torch.zeros(n_elem, 2, 2)
            Gs[:, 0, 0] = g13
            Gs[:, 1, 1] = g23
            R = planar_rotation(self.angles[k])
            if R.dim() == 2:
                R = R.expand(n_elem, 2, 2)
            Gs_rot = torch.einsum("emi,eij,enj->emn", R, Gs, R)
            As = As + Gs_rot * t[k][:, None, None]
        return As

    def plot(self):
        """Illustrate the stacking sequence.

        Each ply is drawn as a band through the thickness (height proportional
        to the ply thickness) with the ply angle annotated.
        """
        import matplotlib.pyplot as plt

        t = torch.stack(self.thicknesses)
        angles = torch.rad2deg(torch.stack(self.angles))
        width = t.sum().item()
        # Interfaces relative to the reference surface (z = 0), offset-shifted.
        z = (
            torch.concatenate([torch.tensor([0.0]), torch.cumsum(t, 0)])
            - t.sum() / 2
            - self._offset * t.sum()
        )

        # One color per material class and orientation
        layer_keys = [
            f"{type(m).__name__}_{angles[i]:.0f}°" for i, m in enumerate(self.materials)
        ]
        unique_combos = list(dict.fromkeys(layer_keys))
        cmap = plt.get_cmap("Pastel1")
        color = {name: cmap(i % 9) for i, name in enumerate(unique_combos)}

        _, ax = plt.subplots(figsize=(4, 5))

        for k in range(self.n_layers):
            z0 = z[k].item()
            z1 = z[k + 1].item()
            ax.add_patch(
                plt.Rectangle(
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
        plt.show()
