import math

import torch
from utils import Case, Problem, run_case

from torchfem import Solid
from torchfem.materials import Hyperelastic3D
from torchfem.mesh import cube_hexa

# Uniaxial stretch of a unit cube to twenty times its original length with a
# Neo-Hookean material, following examples/basic/solid/large_stretch.ipynb:
# prescribed x-displacement on the right face, fixed x on the left face, and
# symmetry constraints on the center planes (requires an odd number of nodes
# per edge). The load is applied in N_INC increments that are geometric in the
# stretch (constant relative stretch per step), which keeps Newton convergent
# on fine meshes where the notebook's uniform increments fail.
E = 1000.0
NU = 0.3
LBD = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))
U = 19.0
N_INC = 21


def psi(F, params):
    """Neo-Hookean strain energy density function."""
    MU = params[0]
    LBD = params[1]
    C = F.transpose(-1, -2) @ F
    logJ = 0.5 * torch.logdet(C)
    return MU / 2 * (torch.trace(C) - 3.0) - MU * logJ + LBD / 2 * logJ**2


def get_stretch(N):
    if N % 2 == 0:
        raise ValueError("N must be odd to place nodes on the center planes.")
    nodes, elements = cube_hexa(N, N, N)

    # Lamé parameters as differentiable parameters (material calibration)
    params = torch.tensor([MU, LBD], requires_grad=True)
    box = Solid(nodes, elements, Hyperelastic3D(psi, params))

    # Boundary conditions
    left = nodes[:, 0] == 0.0
    right = nodes[:, 0] == 1.0
    box.constraints[left, 0] = True
    box.constraints[right, 0] = True
    box.constraints[nodes[:, 1] == 0.5, 1] = True
    box.constraints[nodes[:, 2] == 0.5, 2] = True
    box.displacements[right, 0] = U

    return box, params, right


def setup(N):
    box, params, right = get_stretch(N)
    lam = torch.logspace(0, math.log10(1.0 + U), N_INC)
    increments = (lam - 1.0) / U
    result = {}

    def forward():
        _, result["f"], *_ = box.solve(
            increments=increments, nlgeom=True, differentiable_parameters=params
        )

    def backward():
        # Total reaction force w.r.t. Lamé parameters
        reaction = result["f"][right, 0].sum()
        reaction.backward()

    return Case(dofs=3 * box.n_nod, forward=forward, backward=backward)


PROBLEM = Problem(
    id="hyperelasticity_stretch",
    title="Neo-Hookean stretch benchmark",
    plot_prefix="benchmark_hyperelasticity",
    default_N=[5, 9, 13],
    setup=setup,
)

if __name__ == "__main__":
    run_case(PROBLEM)
