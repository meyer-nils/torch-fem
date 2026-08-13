import numpy as np
import torch
from utils import Case, Problem, run_case

from torchfem import Solid
from torchfem.materials import IsotropicElasticity3D
from torchfem.mesh import cube_hexa

# SIMP compliance minimization of a 3D cantilever, following the "structural-mesh"
# benchmark in pasteurlabs/mosaic. Its random density field spreads element stiffness
# over a factor of 762, which conditions the system far worse than a uniform field,
# where SIMP scales the stiffness matrix by a constant.
LX, LY, LZ = 2.0, 1.0, 1.0
E_MAX = 70000.0
NU = 0.3
X_MIN = 1e-3
P_EXP = 3.0
RHO_0 = 0.5
NOISE = 0.3
F_TOTAL = 1.0


def get_cantilever(N):
    nx, ny, nz = 2 * N, N, N
    nodes, elements = cube_hexa(nx + 1, ny + 1, nz + 1, LX, LY, LZ)

    # mosaic's random initial condition
    rng = np.random.default_rng(0)
    values = np.clip(RHO_0 + NOISE * rng.standard_normal(len(elements)), 0.05, 0.95)
    rho = torch.tensor(values, requires_grad=True)

    # SIMP stiffness E(rho) = E_min + (E_max - E_min) * rho^p, exact as a scale on C
    material = IsotropicElasticity3D(E=E_MAX, nu=NU).vectorize(len(elements))
    scale = X_MIN + (1.0 - X_MIN) * rho**P_EXP
    material.C = scale[:, None, None, None, None] * material.C

    model = Solid(nodes, elements, material)

    # Clamped left face
    model.constraints[nodes[:, 0] == 0.0, :] = True

    # Uniform downward traction on the right face, lumped with trapezoidal weights
    right = nodes[:, 0] == LX
    wy = torch.full((model.n_nod,), LY / ny)
    wy[(nodes[:, 1] == 0.0) | (nodes[:, 1] == LY)] /= 2.0
    wz = torch.full((model.n_nod,), LZ / nz)
    wz[(nodes[:, 2] == 0.0) | (nodes[:, 2] == LZ)] /= 2.0
    model.forces[right, 2] = -F_TOTAL / (LY * LZ) * wy[right] * wz[right]

    return model, rho


def setup(N):
    model, rho = get_cantilever(N)
    result = {}

    def forward():
        result["u"], *_ = model.solve(differentiable_parameters=rho)

    def backward():
        # Structural compliance w.r.t. SIMP densities
        compliance = torch.inner(model.forces.ravel(), result["u"].ravel())
        compliance.backward()

    return Case(dofs=3 * model.n_nod, forward=forward, backward=backward)


PROBLEM = Problem(
    id="structural_cantilever_simp",
    title="SIMP cantilever benchmark",
    plot_prefix="topopt",
    default_N=[20, 30, 40, 50, 60],
    setup=setup,
)

if __name__ == "__main__":
    run_case(PROBLEM)
