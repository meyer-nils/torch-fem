import numpy as np
import matplotlib.pyplot as plt

def prescribe_disps_by_coords(domain, data, dim):
    """
    Apply displacements by matching mesh coordinates with 
    reference coordinates read from material patch file.
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

    
def plot_displacement_field(coords_x, coords_y, u_x, u_y, 
                            node_coords, u_nodal, filename):
    """Plot displacement field within RVE at grid points."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Displacement magnitude contour
    u_magnitude = np.sqrt(u_x**2 + u_y**2)
    im1 = ax1.contourf(coords_x, coords_y, u_magnitude,
                       levels=20, cmap='viridis')
    ax1.set_title('Displacement Magnitude')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Plot element boundary and nodes
    element_x = [node_coords[0,0], node_coords[1,0], node_coords[2,0],
                 node_coords[3,0], node_coords[0,0]]
    element_y = [node_coords[0,1], node_coords[1,1], node_coords[2,1],
                 node_coords[3,1], node_coords[0,1]]
    ax1.plot(element_x, element_y, 'k-', linewidth=2,
             label='Element boundary')
    ax1.scatter(node_coords[:,0], node_coords[:,1], c='red', s=50,
                zorder=5, label='Nodes')
    
    # Plot 2: X-displacement contour
    im2 = ax2.contourf(coords_x, coords_y, u_x, levels=20, cmap='RdBu_r')
    ax2.set_title('X-Displacement')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    ax2.plot(element_x, element_y, 'k-', linewidth=2)
    ax2.scatter(node_coords[:,0], node_coords[:,1], c='red', s=50, zorder=5)
    
    # Plot 3: Y-displacement contour
    im3 = ax3.contourf(coords_x, coords_y, u_y, levels=20, cmap='RdBu_r')
    ax3.set_title('Y-Displacement')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)
    ax3.plot(element_x, element_y, 'k-', linewidth=2)
    ax3.scatter(node_coords[:,0], node_coords[:,1], c='red', s=50, zorder=5)
    
    # Plot 4: Deformed shape with displacement vectors
    # Show every 5th point for clarity
    stride = 5
    coords_x_sub = coords_x[::stride, ::stride]
    coords_y_sub = coords_y[::stride, ::stride]
    u_x_sub = u_x[::stride, ::stride]
    u_y_sub = u_y[::stride, ::stride]
    
    # Scale factor for displacement visualization
    scale_factor = 0.1 / np.max(np.sqrt(u_x**2 + u_y**2)) if np.max(
        np.sqrt(u_x**2 + u_y**2)) > 0 else 1
    
    ax4.quiver(coords_x_sub, coords_y_sub, u_x_sub, u_y_sub, 
               scale_units='xy', scale=1/scale_factor, alpha=0.7, width=0.003)
    ax4.set_title('Displacement Vectors')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_aspect('equal')
    ax4.plot(element_x, element_y, 'k-', linewidth=2)
    ax4.scatter(node_coords[:,0], node_coords[:,1], c='red', s=50, zorder=5)
    
    # Add deformed nodes
    deformed_coords = node_coords + u_nodal
    ax4.scatter(deformed_coords[:,0], deformed_coords[:,1], c='blue', 
                s=50, zorder=5, label='Deformed nodes')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()