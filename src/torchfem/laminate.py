"""Laminate definition for layered shell elements.

A `Laminate` describes a stacking sequence of (possibly rotated) plane-stress
layers. It owns the through-thickness integration stations used by the `Shell`
element and precomputes the transverse shear stiffness and the mass integrals.
The section response (the equivalent of the classical ABD matrices) is
integrated over these stations by the `Shell` during the analysis, so it
remains valid for nonlinear, state-bearing layers.

The convention follows classical lamination theory (CLT): the laminate
mid-surface is located at ``z = 0`` and layers are stacked from bottom
(``z = -h/2``) to top (``z = +h/2``), where ``h`` is the total thickness.
Layer angles are measured from the element's first local axis and rotate the
material counter-clockwise (consistent with `planar_rotation`).
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

    Notes:
        - Layers may carry internal state (e.g. an elastoplastic metal ply in a
          fiber-metal laminate). The laminate state width is the maximum over
          layers; the through-thickness stations are integrated during the
          analysis, so state-bearing layers are supported without precomputed
          ABD matrices.
        - The laminate behaves like a `Material` from the point of view of the
          finite-element base class: it implements `is_vectorized`,
          `vectorize`, and `n_state`, so it can be passed straight to `Shell`.
    """

    def __init__(
        self,
        materials: list[Material],
        thicknesses: list[float] | list[Tensor] | Tensor,
        angles: list[float] | list[Tensor] | Tensor,
        n_simpson: int = 3,
    ):
        if not (len(materials) == len(thicknesses) == len(angles)):
            raise ValueError(
                "materials, thicknesses, and angles must have equal length."
            )
        if len(materials) == 0:
            raise ValueError("A laminate must contain at least one layer.")
        if n_simpson % 2 == 0:
            raise ValueError("n_simpson must be an odd integer.")

        dtype = torch.get_default_dtype()
        self.materials = list(materials)
        self.thicknesses = [torch.as_tensor(t, dtype=dtype) for t in thicknesses]
        self.angles = [torch.as_tensor(a, dtype=dtype) for a in angles]
        self.n_layers = len(materials)
        self.n_simpson = n_simpson

        # Number of through-thickness integration stations
        self.n_z = self.n_layers * n_simpson

        # The laminate is considered vectorized once all layer materials are.
        self.is_vectorized = all(m.is_vectorized for m in self.materials)

        # State width is the maximum over layers, so the shared per-station
        # state slice is wide enough for the most state-bearing layer (e.g. an
        # elastoplastic metal ply). Each layer's `step` only touches the slots
        # it needs and returns the full-width state, so mixing elastic and
        # state-bearing layers needs no per-layer bookkeeping here.
        self.n_state = max(m.n_state for m in self.materials)

    def __repr__(self) -> str:
        return (
            f"<torch-fem laminate ({self.n_layers} layers, "
            f"{self.n_z} integration points)>"
        )

    def vectorize(self, n_elem: int) -> Laminate:
        """Returns a vectorized copy of the laminate for `n_elem` elements.

        Each layer material is vectorized and rotated by its layer angle, and
        the through-thickness stations, transverse shear stiffness, and mass
        integrals are precomputed.

        Args:
            n_elem: Number of elements to vectorize the laminate for.

        Returns:
            Laminate: A new, vectorized laminate instance.
        """
        if self.is_vectorized:
            return self

        new = Laminate.__new__(Laminate)
        new.n_simpson = self.n_simpson
        new.n_layers = self.n_layers
        new.n_z = self.n_z
        new.n_state = self.n_state
        new.angles = self.angles
        new.is_vectorized = True

        # Vectorize and rotate each layer material into the element frame.
        # A deep copy guards against aliasing when a single material instance is
        # reused across layers (e.g. the same ply at different angles).
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

        # Per-layer thickness stacked as [n_layers, n_elem]
        t = torch.stack(self.thicknesses, dim=0)
        self.thickness = t.sum(dim=0)

        # Layer interface coordinates measured from the mid-surface (z = 0)
        z_bot = -0.5 * self.thickness
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

        # Assemble through-thickness stations with absolute z and absolute
        # integration weights (such that sum_j w_j f_j ~ integral f dz).
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

        # Material reference for each station (for the Shell integration loop)
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
        """Effective transverse shear stiffness ``sum_k G_k(rotated) * t_k``.

        The per-layer transverse shear moduli are taken from the (un-rotated)
        material in its principal axes and rotated into the element frame as a
        second-order tensor.
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
