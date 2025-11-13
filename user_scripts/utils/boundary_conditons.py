import numpy as np
import matplotlib.pyplot as plt
import torch


# =============================================================================
def prescribe_disps_by_coords(domain, data, dim):
    """Apply displacements by matching mesh coordinates.

    Args:
        domain: FEM domain object with nodes, displacements, and
            constraints attributes.
        data (dict): Dictionary containing mesh reference coordinates
            and boundary node displacements with keys
            'mesh_nodes_coords_ref' and 'mesh_boundary_nodes_disps'.
        dim (int): Spatial dimension (2 or 3).

    Returns:
        tuple: (num_applied_disps, nodes_constrained) where
            num_applied_disps is the number of successfully applied
            displacements and nodes_constrained is an array of
            constrained node indices.

    Raises:
        RuntimeError: If the number of applied displacements does not
            match the expected number from boundary_disps.
    """
    mesh_coords_ref = data.get('mesh_nodes_coords_ref', {})
    boundary_disps = data.get('mesh_boundary_nodes_disps', {})

    nodes = domain.nodes.numpy()
    tolerance = 1e-6

    num_applied_disps = 0

    nodes_constrained = []

    for node_str, ref_coord in mesh_coords_ref.items():
        if node_str in boundary_disps:
            ref_point = np.array(ref_coord)

            # Find matching node in the mesh
            distances = np.sqrt(np.sum((nodes - ref_point)**2, axis=1))
            closest_idx = np.argmin(distances)

            if distances[closest_idx] < tolerance:
                disp_data = boundary_disps[node_str]

                # Syntax: displacement at [Node_ID, DOF]
                # Constrained displacement at [Node_IDs, DOFs]
                if dim == 2:
                    # Apply displacement for first time step
                    domain.displacements[closest_idx, 0] = float(
                        disp_data[0])
                    domain.displacements[closest_idx, 1] = float(
                        disp_data[1])

                    # Set constraints
                    domain.constraints[closest_idx, 0] = True
                    domain.constraints[closest_idx, 1] = True

                    num_applied_disps += 1

                    nodes_constrained.append(closest_idx)

                elif dim == 3:
                    # Apply displacement for first time step
                    domain.displacements[closest_idx, 0] = float(
                        disp_data[0])
                    domain.displacements[closest_idx, 1] = float(
                        disp_data[1])
                    domain.displacements[closest_idx, 2] = float(
                        disp_data[2])

                    # Set constraints
                    domain.constraints[closest_idx, 0] = True
                    domain.constraints[closest_idx, 1] = True
                    domain.constraints[closest_idx, 2] = True

                    num_applied_disps += 1

                    nodes_constrained.append(closest_idx)

    if num_applied_disps == len(boundary_disps.keys()):
        return num_applied_disps, np.array(nodes_constrained)
    else:
        raise RuntimeError("Wrong number of boundary displacements!")