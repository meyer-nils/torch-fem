import torch
from utils import Case, Problem, run_case

from torchfem import Solid
from torchfem.materials import IsotropicElasticity3D
from torchfem.mesh import cube_hexa


def get_cube(N):
    nodes, elements = cube_hexa(N, N, N)

    # Material model
    material = IsotropicElasticity3D(E=1000.0, nu=0.3)

    # Define cube
    cube = Solid(nodes, elements, material)

    # Assign boundary conditions
    cube.forces = torch.zeros_like(nodes, requires_grad=True)
    cube.constraints[nodes[:, 0] == 0.0, :] = True
    cube.constraints[nodes[:, 0] == 1.0, 0] = True
    cube.displacements[nodes[:, 0] == 1.0, 0] = 0.1

    return cube


def setup(N):
    cube = get_cube(N)
    result = {}

    def forward():
        result["u"], *_ = cube.solve(differentiable_parameters=cube.forces)

    def backward():
        result["u"].sum().backward()

    return Case(dofs=3 * cube.n_nod, forward=forward, backward=backward)


PROBLEM = Problem(
    id="cube_hexa_extension",
    title="Cube extension benchmark",
    plot_prefix="benchmark",
    default_N=[10, 20, 30, 40, 50, 60, 70, 80],
    setup=setup,
)

if __name__ == "__main__":
    run_case(PROBLEM)
