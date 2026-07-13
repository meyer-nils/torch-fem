import torch
from utils import Case, Problem, run_case

from torchfem import SolidHeat
from torchfem.materials import IsotropicConductivity3D
from torchfem.mesh import cube_hexa

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


def setup(N):
    model, rho = get_slab(N)
    result = {}

    def forward():
        result["u"], *_ = model.solve(differentiable_parameters=rho)

    def backward():
        # Thermal compliance w.r.t. SIMP densities
        compliance = torch.inner(model.heat_flux.ravel(), result["u"].ravel())
        compliance.backward()

    return Case(dofs=model.n_nod, forward=forward, backward=backward)


PROBLEM = Problem(
    id="thermal_slab_simp",
    title="Thermal SIMP slab benchmark",
    plot_prefix="benchmark_thermal",
    default_N=[16, 32, 64, 128, 256, 512, 1024],
    setup=setup,
)

if __name__ == "__main__":
    run_case(PROBLEM)
