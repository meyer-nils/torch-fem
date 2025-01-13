import argparse
import time

import torch

from torchfem import Solid
from torchfem.elements import linear_to_quadratic
from torchfem.materials import IsotropicElasticity3D
from torchfem.mesh import cube_hexa


def get_cube(N, order=1):
    nodes, elements = cube_hexa(N, N, N)
    if order == 2:
        nodes, elements = linear_to_quadratic(nodes, elements)
    elif order > 2:
        raise ValueError("Only linear and quadratic elements are supported.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the cube problem.")
    parser.add_argument("-N", type=int, help="The value of N", default=10)
    parser.add_argument("-device", type=str, help="Troch default device", default="cpu")
    parser.add_argument("-order", type=int, help="The order of the element", default=1)
    args = parser.parse_args()

    torch.set_default_device(args.device)

    box = get_cube(args.N, args.order)
    dofs = box.n_dofs

    # Forward pass
    start_time = time.time()
    u, f, sigma, epsilon, state = box.solve(rtol=1e-5)
    end_time = time.time()
    fwd_t = end_time - start_time

    # Backward pass
    start_time = time.time()
    u.sum().backward(retain_graph=True)
    end_time = time.time()
    bwd_t = end_time - start_time

    print(f"| {args.N:3d} | {dofs:8d} | {fwd_t:8.2f}s | {bwd_t:8.2f}s |")
