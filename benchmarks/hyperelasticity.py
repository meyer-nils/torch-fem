import math

import torch
from utils import Case, Problem, run_case

from torchfem import Solid
from torchfem.materials import Hyperelastic3D
from torchfem.mesh import cube_hexa

# Uniaxial stretch of a Neo-Hookean box, following
# examples/basic/solid/large_stretch.ipynb. N refines y and z only, over four
# cubic elements along x, so the benchmark measures size rather than element
# aspect ratio. Increments are geometric in the stretch, which keeps Newton
# convergent where uniform ones fail. Stretching beyond ten turns the tangent
# indefinite, which every algebraic multigrid preconditioner handles far worse.
STRETCH = 10.0
E = 1000.0
NU = 0.3
LBD = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))
N_INC = 11


def psi(F, params):
    """Neo-Hookean strain energy density function."""
    MU = params[0]
    LBD = params[1]
    C = F.transpose(-1, -2) @ F
    logJ = 0.5 * torch.logdet(C)
    return MU / 2 * (torch.trace(C) - 3.0) - MU * logJ + LBD / 2 * logJ**2


def get_stretch(N):
    if N % 2 == 0:
        raise ValueError("N must be odd to place nodes on the y and z center planes.")
    # Four elements deep, matching the y and z spacing
    lx = 4.0 / (N - 1)
    nodes, elements = cube_hexa(5, N, N, lx, 1.0, 1.0)

    # Lamé parameters as differentiable parameters (material calibration)
    params = torch.tensor([MU, LBD], requires_grad=True)
    box = Solid(nodes, elements, Hyperelastic3D(psi, params))

    # Boundary conditions
    left = nodes[:, 0] == 0.0
    right = nodes[:, 0] == lx
    box.constraints[left, 0] = True
    box.constraints[right, 0] = True
    box.constraints[nodes[:, 1] == 0.5, 1] = True
    box.constraints[nodes[:, 2] == 0.5, 2] = True
    box.displacements[right, 0] = (STRETCH - 1.0) * lx

    return box, params, right


def setup(N):
    box, params, right = get_stretch(N)
    lam = torch.logspace(0, math.log10(STRETCH), N_INC)
    increments = (lam - 1.0) / (STRETCH - 1.0)
    result = {}

    def forward():
        _, result["f"], *_ = box.solve(
            increments=increments,
            nlgeom=True,
            differentiable_parameters=params,
            method="cg",
        )

    def backward():
        # Total reaction force w.r.t. Lamé parameters
        reaction = result["f"][right, 0].sum()
        reaction.backward()

    return Case(dofs=3 * box.n_nod, forward=forward, backward=backward)


PROBLEM = Problem(
    id="hyperelasticity_stretch",
    title="Neo-Hookean stretch benchmark",
    plot_prefix="hyperelasticity",
    default_N=[25, 35, 45, 55, 65],
    setup=setup,
)

if __name__ == "__main__":
    run_case(PROBLEM)
