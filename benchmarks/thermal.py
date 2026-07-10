import argparse
import time

import torch

from torchfem import SolidHeat
from torchfem.materials import IsotropicConductivity3D
from torchfem.mesh import cube_hexa
from utils import VramMonitor

torch.set_default_dtype(torch.float64)

# Quasi-2D heated slab with SIMP-penalized conductivity, following the
# "thermal-mesh" benchmark in pasteurlabs/mosaic: a single HEX8 layer on
# [0,2]x[0,1]x[0,1], cold left face (T=0), uniform heat flux Q_TOTAL on the
# right face, and per-element densities rho as differentiable parameters.
LX, LY, LZ = 2.0, 1.0, 1.0
K_MAX = 1.0
K_MIN_RATIO = 1e-3
P_EXP = 3.0
RHO_0 = 0.5
Q_TOTAL = 1.0


def get_slab(N):
    nx, ny, nz = N, N // 2, 1
    nodes, elements = cube_hexa(nx + 1, ny + 1, nz + 1, LX, LY, LZ)

    material = IsotropicConductivity3D(kappa=K_MAX)
    model = SolidHeat(nodes, elements, material)

    # SIMP conductivity k = k_max * (k_min_ratio + (1 - k_min_ratio) * rho^p)
    rho = torch.full((model.n_elem,), RHO_0, requires_grad=True)
    scale = K_MIN_RATIO + (1.0 - K_MIN_RATIO) * rho**P_EXP
    model.material = material.vectorize(model.n_elem)
    model.material.KAPPA = scale[:, None, None] * model.material.KAPPA

    # Cold left face (T = 0)
    model.constraints[nodes[:, 0] == 0.0, :] = True

    # Uniform flux on the right face, lumped to nodes with trapezoidal weights
    # (interior nodes carry a full cell width, boundary nodes half).
    q_n = Q_TOTAL / (LY * LZ)
    right = nodes[:, 0] == LX
    wy = torch.full((model.n_nod,), LY / ny)
    wy[(nodes[:, 1] == 0.0) | (nodes[:, 1] == LY)] /= 2.0
    wz = torch.full((model.n_nod,), LZ / nz)
    wz[(nodes[:, 2] == 0.0) | (nodes[:, 2] == LZ)] /= 2.0
    model.heat_flux[right, 0] = q_n * wy[right] * wz[right]

    return model, rho


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the thermal slab problem.")
    parser.add_argument(
        "-N", type=int, help="Elements along x (ny = N/2, nz = 1)", default=16
    )
    parser.add_argument("-device", type=str, help="Torch default device", default="cpu")
    args = parser.parse_args()

    torch.set_default_device(args.device)

    # Sample driver-level VRAM (cuda only) around all problem-specific work.
    monitor = VramMonitor() if args.device == "cuda" else None
    if monitor is not None:
        monitor.start()

    # Start timing
    print(f"START:{time.time()}")

    # Setup
    model, rho = get_slab(args.N)
    print(f"SETUP_DONE:{time.time()}")

    # Forward pass
    u, f, flux, grad, state = model.solve(differentiable_parameters=rho)
    print(f"FWD_DONE:{time.time()}")

    # Backward pass: thermal compliance w.r.t. SIMP densities
    compliance = torch.inner(model.heat_flux.ravel(), u.ravel())
    compliance.backward()
    print(f"BWD_DONE:{time.time()}")

    # Emit memory diagnostics
    if monitor is not None:
        monitor.stop()
        for tag, val in monitor.report().items():
            print(f"{tag}:{val}")
