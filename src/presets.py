from physics import *
from itertools import product


def get_neighbors(i: int, w: int, h: int) -> list[tuple[int, str]]:
    """
    Get valid neighbor indices for a particle in a 2D grid.

    Args:
        i: Particle index
        w: Grid width
        h: Grid height

    Returns:
        List of tuples (neighbor_index, direction) where direction is 'right', 'left', 'bottom', or 'top'
    """
    neighbors = []
    total = w * h

    # Right neighbor (next x position)
    if i + h < total:
        neighbors.append((i + h, "right"))

    # Left neighbor (prev x position)
    if i - h >= 0:
        neighbors.append((i - h, "left"))

    # Bottom neighbor (next y position)
    if (i + 1) % h != 0:
        neighbors.append((i + 1, "bottom"))

    # Top neighbor (prev y position)
    if i % h != 0:
        neighbors.append((i - 1, "top"))

    return neighbors


def add_mesh_2d(
    simulation: Simulation,
    top_left: tuple | list | np.ndarray,
    w: int,
    h: int,
    step: float | np.ndarray,
    k: float,
    **kwargs,
):
    top_left = np.array(top_left)

    if isinstance(step, (list, tuple, np.ndarray)):
        step_x, step_y = step[0], step[1]
    else:
        step_x, step_y = step, step

    x_pos = np.arange(0, w) * step_x
    y_pos = np.arange(0, h) * step_y

    xy_pos = product(x_pos, y_pos)
    particles = []

    for x, y in xy_pos:
        pos = np.array([x, y]) + top_left
        particles.append(Particle(m=kwargs.get("m", 1.0), x=pos))

    # Connect each particle to its neighbors (following the verified playground logic)
    for i, particle in enumerate(particles):
        # Calculate row and col based on the layout: product(x_pos, y_pos)
        col = i // h
        row = i % h
        
        # Particles in the first row are treated as fixed pivots
        if row == 0:
            continue
        
        # Gravity on all non-pivot particles
        gravitational_constraint = make_gravitational_constraint(particle, kwargs.get("g", np.array([0, 500])))
        particle.constraints.append(gravitational_constraint)
        
        # Dampening
        dampening_k = kwargs.get("dampening", 0.02)
        particle.constraints.append(make_dampening_constraint(particle, dampening_k))

        # Mesh Connectivity: connect to left and top neighbors to build the grid
        # Connect to Left neighbor (previous x, same y)
        if col > 0:
            left_neighbor = particles[(col - 1) * h + row]
            dr = kwargs.get("dr", step_x)
            # Add constraint to current particle
            particle.constraints.append(make_elastic_constraint(particle, left_neighbor, k, dr))
            
            # Only add reverse constraint to neighbor if neighbor is NOT a pivot (row 0)
            neighbor_row = row
            if neighbor_row != 0:
                left_neighbor.constraints.append(make_elastic_constraint(left_neighbor, particle, k, dr))
            
        # Connect to Top neighbor (same x, previous y)
        if row > 0:
            top_neighbor = particles[col * h + (row - 1)]
            dr = kwargs.get("dr", step_y)
            particle.constraints.append(make_elastic_constraint(particle, top_neighbor, k, dr))
            
            # Only add reverse constraint if top neighbor is NOT a pivot (row 0)
            neighbor_row = row - 1
            if neighbor_row != 0:
                top_neighbor.constraints.append(make_elastic_constraint(top_neighbor, particle, k, dr))

    simulation.particles.extend(particles)

    return particles
