"""Open-hole tension - quasi-isotropic carbon/epoxy laminate (quad membrane).

Test/demo script for the ``Laminate`` feature combined with an all-quad
``Planar`` membrane model.

The torch-fem ``Shell`` element is a flat-facet *triangular* element and does
not support quadrilaterals. Since open-hole tension is an in-plane (membrane)
problem and this layup is symmetric (no bending-extension coupling), we mesh the
plate with quad elements and solve it with the ``Planar`` element, using the
laminate's *homogenized* effective stiffness. The ``Laminate`` is still built to
verify its classical-lamination-theory invariants and to provide that effective
stiffness.

Geometry:
    - Plate 150 mm (x) x 75 mm (y), centered circular hole of radius 15 mm.

Laminate:
    - Carbon/epoxy ply, symmetric quasi-isotropic [0 / 45 / 90 / -45]_s
      (8 plies, 0.25 mm each -> 2.0 mm total).

Loading:
    - Left edge fully clamped (u_x = u_y = 0).
    - Right edge: prescribed u_x = 0.5 mm, u_y = 0.

Run directly::

    python examples/basic/shell/plate_with_hole_laminate.py
"""

import numpy as np
import torch

from torchfem import Laminate, Planar
from torchfem.materials import (
    IsotropicElasticityPlaneStress,
    OrthotropicElasticityPlaneStress,
)
from torchfem.failure import PlyStrength, hashin, rotate_stress_to_ply, tsai_hill
from torchfem.utils import stiffness2voigt

torch.set_default_dtype(torch.float64)

# Representative carbon/epoxy ply strengths (MPa, positive magnitudes).
STRENGTH = PlyStrength(Xt=2200.0, Xc=1400.0, Yt=70.0, Yc=240.0, S12=90.0, S23=60.0)

# --- Geometry ---
LX, LY = 150.0, 75.0
R = 15.0
CX, CY = LX / 2.0, LY / 2.0

# Geometric nonlinearity. The Planar element supports it (the Shell element does
# not); at this strain level (~0.33%) it changes the peak stress by ~0.3%, so it
# is left off by default. Set True to mirror an Abaqus NLGEOM analysis.
NLGEOM = False


def plate_with_hole_quad_mesh(n_theta=96, n_r=24, grading=2.0):
    """All-quad O-grid mesh of a rectangle with a centered circular hole.

    Rays emanate from the hole center; along each ray nodes are placed from the
    hole boundary (r = R) out to the rectangle boundary, graded to cluster near
    the hole. Quads connect adjacent rays and radial layers.

    Args:
        n_theta: Number of angular subdivisions (rays).
        n_r: Number of radial subdivisions (quad layers).
        grading: Radial clustering exponent (>1 refines near the hole).

    Returns:
        (nodes, elements) tensors; ``nodes`` are 2D for the planar model.
    """
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    s = np.linspace(0.0, 1.0, n_r + 1) ** grading  # radial parameter in [0, 1]

    npr = n_r + 1
    pts = np.empty((n_theta * npr, 2))
    for i, th in enumerate(thetas):
        ct, st = np.cos(th), np.sin(th)
        # Distance from center to the rectangle boundary along this ray
        tx = (LX - CX) / ct if ct > 1e-12 else ((-CX) / ct if ct < -1e-12 else np.inf)
        ty = (LY - CY) / st if st > 1e-12 else ((-CY) / st if st < -1e-12 else np.inf)
        r_out = min(tx, ty)
        r = R + (r_out - R) * s
        pts[i * npr : (i + 1) * npr, 0] = CX + r * ct
        pts[i * npr : (i + 1) * npr, 1] = CY + r * st

    elems = []
    for i in range(n_theta):
        inext = (i + 1) % n_theta
        for j in range(n_r):
            elems.append(
                [i * npr + j, inext * npr + j, inext * npr + j + 1, i * npr + j + 1]
            )
    elems = np.array(elems, dtype=np.int64)

    # Ensure counter-clockwise ordering (positive Jacobian).
    p = pts[elems]
    area2 = np.zeros(len(elems))
    for k in range(4):
        a, b = p[:, k], p[:, (k + 1) % 4]
        area2 += a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
    flip = area2 < 0.0
    elems[flip] = elems[flip][:, ::-1]

    nodes = torch.tensor(pts, dtype=torch.get_default_dtype())
    elements = torch.tensor(elems, dtype=torch.long)
    return nodes, elements


def build_laminate():
    """Symmetric quasi-isotropic carbon/epoxy laminate [0 / 45 / 90 / -45]_s."""
    # NOTE: nu_23 = 0.4 is recorded for reference but does not enter the
    #       plane-stress formulation (transverse shear is set by G_23).
    cfrp = OrthotropicElasticityPlaneStress(
        E_1=150000.0,
        E_2=9000.0,
        nu_12=0.34,
        G_12=5000.0,
        G_13=5000.0,
        G_23=5000.0,
        rho=1.6e-9,
    )
    deg = torch.pi / 180.0
    half = [0.0, 45 * deg, 90 * deg, -45 * deg]
    angles = half + half[::-1]  # mirror -> symmetric
    layup = Laminate(
        materials=[cfrp] * len(angles),
        thicknesses=[0.25] * len(angles),
        angles=angles,
    )
    return layup, cfrp.rho


def effective_membrane_material(layup, rho):
    """Homogenized in-plane material from the laminate A-matrix.

    For a quasi-isotropic laminate the membrane stiffness is isotropic, so the
    effective ply is an isotropic plane-stress material with stiffness A / h.
    """
    lam = layup.vectorize(1)
    A, _, _ = lam.abd
    A0 = A[0]
    h = float(lam.thickness[0])
    Q11 = A0[0, 0] / h
    Q12 = A0[0, 1] / h
    nu = float(Q12 / Q11)
    E = float(Q11 * (1.0 - nu**2))
    mat = IsotropicElasticityPlaneStress(E=E, nu=nu, rho=float(rho))
    return mat, h, E, nu, lam


def check_laminate_invariants(lam):
    """A quasi-isotropic, symmetric, balanced laminate must satisfy CLT identities."""
    A, B, D = lam.abd
    A, B = A[0], B[0]
    a_scale = A.abs().max()
    return A, {
        "B = 0 (symmetric)": (B.abs().max() / a_scale).item(),
        "A16 = 0 (balanced)": (A[0, 2].abs() / a_scale).item(),
        "A26 = 0 (balanced)": (A[1, 2].abs() / a_scale).item(),
        "A11 = A22 (quasi-iso)": ((A[0, 0] - A[1, 1]).abs() / a_scale).item(),
        "A66 = (A11-A12)/2": (
            (A[2, 2] - 0.5 * (A[0, 0] - A[0, 1])).abs() / a_scale
        ).item(),
    }


def apply_open_hole_bcs(plate, nodes):
    """Left edge clamped; right edge prescribed u_x = 0.5, u_y = 0."""
    x = nodes[:, 0]
    left = x < 1e-6
    right = x > LX - 1e-6
    plate.constraints[left] = True  # u_x = u_y = 0
    plate.constraints[right] = True  # u_x and u_y fixed
    plate.displacements[right, 0] = 0.5  # prescribed u_x
    return left, right


def von_mises(sigma):
    """Plane-stress von Mises stress per element."""
    sxx, syy, sxy = sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 0, 1]
    return torch.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)


def ply_von_mises(lam, sigma_h, E_eff, nu_eff):
    """Per-ply von Mises stress recovered from the membrane stress (CLT).

    A symmetric laminate under in-plane load shares one mid-plane strain across
    all plies. We recover that strain from the homogenized stress
    (``eps = S_eff : sigma_h``) and push it through each ply's rotated stiffness
    (``sigma_k = Q_k : eps``). This is exactly what a conventional composite
    shell reports at its per-ply section points.

    Args:
        lam: Vectorized laminate (provides rotated ply stiffnesses ``Q_k``).
        sigma_h: Homogenized element stress, shape `(n_elem, 2, 2)`.
        E_eff, nu_eff: Effective isotropic membrane constants.

    Returns:
        Tensor of shape `(n_ply, n_elem)` with per-ply von Mises stress.
    """
    # Engineering membrane strain from the homogenized (isotropic) stress
    sxx, syy, sxy = sigma_h[:, 0, 0], sigma_h[:, 1, 1], sigma_h[:, 0, 1]
    exx = (sxx - nu_eff * syy) / E_eff
    eyy = (syy - nu_eff * sxx) / E_eff
    gxy = 2.0 * (1.0 + nu_eff) / E_eff * sxy
    eps_voigt = torch.stack([exx, eyy, gxy], dim=-1)  # [n_elem, 3]

    vms = []
    for plymat in lam.materials:
        Q = stiffness2voigt(plymat.C)[0]  # [3, 3], uniform over elements
        s = eps_voigt @ Q.T  # [n_elem, 3] -> [sxx, syy, sxy]
        vms.append(
            torch.sqrt(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2 + 3 * s[:, 2] ** 2)
        )
    return torch.stack(vms)  # [n_ply, n_elem]


def ply_stresses_material(lam, sigma_h, E_eff, nu_eff):
    """Per-ply stresses in material (fiber) axes recovered from the membrane stress.

    Returns a tensor of shape `(n_ply, n_elem, 3)` holding ``(s11, s22, t12)``
    for each ply in its own fiber coordinate system.
    """
    sxx, syy, sxy = sigma_h[:, 0, 0], sigma_h[:, 1, 1], sigma_h[:, 0, 1]
    exx = (sxx - nu_eff * syy) / E_eff
    eyy = (syy - nu_eff * sxx) / E_eff
    gxy = 2.0 * (1.0 + nu_eff) / E_eff * sxy
    eps_voigt = torch.stack([exx, eyy, gxy], dim=-1)  # [n_elem, 3]

    out = []
    for plymat, ang in zip(lam.materials, lam.angles):
        Q = stiffness2voigt(plymat.C)[0]  # [3, 3]
        s = eps_voigt @ Q.T  # element-axes [sxx, syy, sxy]
        sig = torch.zeros(s.shape[0], 2, 2)
        sig[:, 0, 0] = s[:, 0]
        sig[:, 1, 1] = s[:, 1]
        sig[:, 0, 1] = s[:, 2]
        sig[:, 1, 0] = s[:, 2]
        sig_mat = rotate_stress_to_ply(sig, ang)  # element -> material axes
        out.append(
            torch.stack([sig_mat[:, 0, 0], sig_mat[:, 1, 1], sig_mat[:, 0, 1]], dim=-1)
        )
    return torch.stack(out)  # [n_ply, n_elem, 3]


def evaluate_failure(lam, ply_mat):
    """Tsai-Hill and Hashin failure indices per ply, plus envelopes.

    Args:
        ply_mat: Per-ply material-axes stresses, shape `(n_ply, n_elem, 3)`.

    Returns:
        Dict with ``tsai_hill`` `(n_ply, n_elem)`, ``hashin`` mode envelopes
        `(n_ply, n_elem)`, and per-element envelopes over all plies.
    """
    s11, s22, t12 = ply_mat[..., 0], ply_mat[..., 1], ply_mat[..., 2]
    th = tsai_hill(s11, s22, t12, STRENGTH)  # [n_ply, n_elem]
    hs = hashin(s11, s22, t12, STRENGTH)
    return {
        "tsai_hill": th,
        "tsai_hill_env": th.max(dim=0).values,
        "hashin_fiber": hs["fiber"],
        "hashin_matrix": hs["matrix"],
        "hashin_env": hs["max"].max(dim=0).values,
        "hashin": hs,
    }


def nodal_average(plate, elem_field):
    """Average an element field to nodes (as Abaqus does for contour plots).

    Returns a per-element field where each element value is replaced by the
    average over its nodes of the node-averaged field. The peak of this field
    is the like-for-like counterpart of an averaged Abaqus contour maximum.
    """
    n_nod = plate.n_nod
    acc = torch.zeros(n_nod)
    cnt = torch.zeros(n_nod)
    e = plate.elements
    for c in range(e.shape[1]):
        acc.index_add_(0, e[:, c], elem_field)
        cnt.index_add_(0, e[:, c], torch.ones_like(elem_field))
    nodal = acc / cnt.clamp_min(1.0)
    return nodal[e].mean(dim=1)  # back to per-element (smoothed)


def solve_model(n_theta, n_r):
    """Build, solve, and recover per-ply stress for one mesh resolution."""
    layup, rho = build_laminate()
    mat, h, E_eff, nu_eff, lam = effective_membrane_material(layup, rho)
    nodes, elements = plate_with_hole_quad_mesh(n_theta=n_theta, n_r=n_r)
    plate = Planar(nodes, elements, mat, thickness=h)
    apply_open_hole_bcs(plate, nodes)
    u, f, sigma, _, _ = plate.solve(nlgeom=NLGEOM)
    vm_ply = ply_von_mises(lam, sigma, E_eff, nu_eff)
    return plate, lam, sigma, vm_ply, (E_eff, nu_eff, h)


def convergence_study():
    """Per-ply peak von Mises vs. mesh refinement.

    The peak converges from below toward ~1530 MPa. Because a conventional
    shell (e.g. Abaqus S4R) samples stress at a single element-centroid
    integration point, a coarser mesh under-samples the sharp gradient at the
    hole: a hole-edge element size of ~3 mm reproduces the Abaqus reference
    (1290 MPa), while the converged value is ~1530 MPa.
    """
    print("\nMesh convergence (critical-ply peak von Mises):")
    print(f"  {'n_theta':>7} {'n_r':>4} {'elements':>9} "
          f"{'hole h [mm]':>11} {'elem peak':>10} {'nodal-avg':>10}")
    for n_theta, n_r in [(24, 6), (32, 8), (48, 12), (96, 24), (144, 36)]:
        plate, lam, sigma, vm_ply, _ = solve_model(n_theta, n_r)
        vm_env = vm_ply.max(dim=0).values
        elem_peak = float(vm_env.max())
        nodal_peak = float(nodal_average(plate, vm_env).max())
        hole_h = 2.0 * np.pi * R / n_theta  # circumferential element size at hole
        print(f"  {n_theta:>7} {n_r:>4} {plate.n_elem:>9} "
              f"{hole_h:>11.2f} {elem_peak:>10.1f} {nodal_peak:>10.1f}")
    print("     (Abaqus conventional-shell per-ply ref: 1290 MPa @ hole h ~ 3 mm)")


def visualize(plate, field, title, label, off_screen=False, screenshot=None):
    """pyvista view of a per-element scalar field on the (undeformed) quad mesh."""
    import pyvista as pv

    pts2 = plate.nodes.detach().cpu().numpy()
    pts = np.hstack([pts2, np.zeros((len(pts2), 1))])
    elems = plate.elements.detach().cpu().numpy()
    nv = elems.shape[1]
    faces = np.hstack([np.full((len(elems), 1), nv, dtype=np.int64), elems]).ravel()

    mesh = pv.PolyData(pts, faces)
    mesh.cell_data[label] = field.detach().cpu().numpy()

    pl = pv.Plotter(off_screen=off_screen)
    pl.add_mesh(
        mesh,
        scalars=label,
        show_edges=True,
        cmap="jet",  # blue (low) -> red (high)
    )
    pl.add_text(title, font_size=10)
    pl.view_xy()
    if screenshot is not None:
        pl.screenshot(screenshot)
    pl.show()


def main():
    # 1) Laminate + homogenized membrane material.
    layup, rho = build_laminate()
    mat, h, E_eff, nu_eff, lam = effective_membrane_material(layup, rho)

    A, results = check_laminate_invariants(lam)
    print("Laminate ABD invariants (relative residuals, should be ~0):")
    for name, val in results.items():
        status = "ok" if val < 1e-9 else "FAIL"
        print(f"  [{status}] {name:<28} {val:.2e}")
    assert all(v < 1e-9 for v in results.values()), "Laminate invariants violated"
    print(f"\nTotal thickness h = {h:.3f} mm")
    print(f"Effective isotropic membrane: E = {E_eff:.1f} MPa, nu = {nu_eff:.4f}")

    # 2) All-quad mesh + planar membrane model.
    nodes, elements = plate_with_hole_quad_mesh()
    plate = Planar(nodes, elements, mat, thickness=h)
    print(f"\nMesh: {plate.n_nod} nodes, {plate.n_elem} quad elements")

    # 3) Solve the open-hole tension problem.
    apply_open_hole_bcs(plate, nodes)
    u, f, sigma, _, _ = plate.solve(nlgeom=NLGEOM)
    assert torch.all(torch.isfinite(u)), "Non-finite displacement"
    print(f"Max |u_x| = {u[:, 0].abs().max():.4e} mm (prescribed 0.5 at right edge)")

    # 4a) Homogenized (smeared-laminate) stress.
    mises_h = von_mises(sigma)

    # 4b) Per-ply stress (what a conventional composite shell reports).
    deg = 180.0 / torch.pi
    vm_ply = ply_von_mises(lam, sigma, E_eff, nu_eff)  # [n_ply, n_elem]
    vm_env = vm_ply.max(dim=0).values  # worst ply per element
    ply_peaks = vm_ply.max(dim=1).values  # peak per ply
    crit_ply = int(ply_peaks.argmax())

    nodal_peak = float(nodal_average(plate, vm_env).max())

    print("\nPeak von Mises stress:")
    print(f"  homogenized (smeared)     = {mises_h.max():.1f} MPa")
    for k, ang in enumerate(lam.angles):
        print(f"  ply {k} ({float(ang) * deg:+5.0f} deg)         = {ply_peaks[k]:.1f} MPa")
    print(
        f"  -> critical ply {crit_ply} ({float(lam.angles[crit_ply]) * deg:+.0f} deg): "
        f"element peak = {ply_peaks[crit_ply]:.1f} MPa, "
        f"nodal-averaged = {nodal_peak:.1f} MPa"
    )
    print("     (Abaqus conventional-shell per-ply ref: 1290 MPa)")

    # 4c) Ply failure criteria (Tsai-Hill and Hashin) in material axes.
    ply_mat = ply_stresses_material(lam, sigma, E_eff, nu_eff)
    fail = evaluate_failure(lam, ply_mat)

    th_ply_peaks = fail["tsai_hill"].max(dim=1).values
    th_crit_ply = int(th_ply_peaks.argmax())
    th_max = float(fail["tsai_hill_env"].max())
    hf_max = float(fail["hashin_fiber"].max())
    hm_max = float(fail["hashin_matrix"].max())

    print("\nFailure analysis (FI >= 1 means failure):")
    print("  Tsai-Hill peak FI per ply:")
    for k, ang in enumerate(lam.angles):
        print(f"    ply {k} ({float(ang) * deg:+5.0f} deg) = {th_ply_peaks[k]:.3f}")
    print(
        f"  -> Tsai-Hill   max FI = {th_max:.3f} "
        f"in ply {th_crit_ply} ({float(lam.angles[th_crit_ply]) * deg:+.0f} deg)"
    )
    print(f"  -> Hashin fiber  max FI = {hf_max:.3f}")
    print(f"  -> Hashin matrix max FI = {hm_max:.3f}")
    # Strength ratio to first-ply failure (Tsai-Hill, quadratic -> 1/sqrt(FI))
    spf = 1.0 / th_max**0.5
    mode = "matrix" if hm_max >= hf_max else "fiber"
    print(f"  -> first-ply failure at load factor ~ {spf:.3f} (Tsai-Hill)")
    print(f"  -> Hashin dominant mode: {mode}")

    # 4d) Mesh convergence of the critical-ply peak.
    convergence_study()

    print("\nAll checks passed.")

    # 5) Interactive pyvista viewers: peak per-ply stress and Tsai-Hill failure index.
    visualize(
        plate,
        vm_env,
        "Open-hole tension - max per-ply von Mises stress",
        "von Mises stress [MPa]",
    )
    visualize(
        plate,
        fail["tsai_hill_env"],
        "Open-hole tension - Tsai-Hill failure index (>=1 fails)",
        "Tsai-Hill FI",
    )


if __name__ == "__main__":
    main()
