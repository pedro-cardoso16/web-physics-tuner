from __future__ import annotations

import torch
import math

from .physics import *
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


def create_mesh(
    top_left: tuple | list | torch.Tensor,
    w: int,
    h: int,
    step: float | torch.Tensor,
    k: float,
    k_damp: float | torch.Tensor | None = None,
    **kwargs,
):
    """

    Args:
        simulation (Simulation): target simulation object to add the mesh.
        top_left (tuple | list | torch.Tensor): position of the top-left mesh node.
        w (int): width, number of horizontal nodes
        h (int): height, number of vertical nodes
        step (float | torch.Tensor): distance between nodes

        kwargs (Any): additional params to set for the forces
            - m (float | ArrayLike): nodes' mass
            - k (float | ArrayLike): elastic constant
            - dr (float | ArrayLike): rest distance for elastic force
            - a (float): first order dampening parameter
            - b (float): second order dampening parameter
    """
    top_left = torch.as_tensor(top_left, dtype=torch.float64)

    if isinstance(step, (list, tuple, torch.Tensor)):
        step_x, step_y = step[0], step[1]
    else:
        step_x, step_y = step, step

    x_pos = torch.arange(0, w, dtype=torch.float64) * step_x
    y_pos = torch.arange(0, h, dtype=torch.float64) * step_y

    xy_pos = product(x_pos, y_pos)
    particles = []

    for x, y in xy_pos:
        pos = torch.stack((x, y)) + top_left
        particles.append(Particle(m=kwargs.get("m", 1.0), x=pos))

    if isinstance(k_damp, (float, int)):
        k_damp = torch.full((len(particles),), k_damp, dtype=torch.float64)
    elif k_damp is None:
        k_damp = torch.zeros(len(particles), dtype=torch.float64)

    # Connect each particle to its neighbors (following the verified playground logic)
    for i, particle in enumerate(particles):
        # Calculate row and col based on the layout: product(x_pos, y_pos)
        col = i // h
        row = i % h

        # Particles in the first row are treated as fixed pivots
        if row == 0:
            continue

        # Gravity on all non-pivot particles
        gravitational_constraint = make_gravitational_constraint(
            particle, kwargs.get("g", torch.tensor([0.0, 1.0]))
        )
        particle.constraints.append(gravitational_constraint)

        # Dampening
        dampening_k = kwargs.get("dampening", 0.0)
        particle.constraints.append(make_dampening_constraint(particle, dampening_k))

        # Mesh Connectivity: connect to left and top neighbors to build the grid
        # Connect to Left neighbor (previous x, same y)
        if col > 0:
            left_neighbor = particles[(col - 1) * h + row]
            dr = kwargs.get("dr", step_x)
            # Add constraint to current particle
            particle.constraints.append(
                make_elastic_constraint(
                    particle,
                    left_neighbor,
                    kwargs.get("k", k),
                    dr,
                    k_damp=k_damp[col - 1],
                )
            )

            # Only add reverse constraint to neighbor if neighbor is NOT a pivot (row 0)
            neighbor_row = row
            if neighbor_row != 0:
                left_neighbor.constraints.append(
                    make_elastic_constraint(
                        left_neighbor, particle, kwargs.get("k", k), dr, k_damp=k_damp[col - 1]
                    )
                )

        # Connect to Top neighbor (same x, previous y)
        if row > 0:
            top_neighbor = particles[col * h + (row - 1)]
            dr = kwargs.get("dr", step_y)
           
            try:
                k_val = kwargs.get("k", k)[row - 1]
            except:
                k_val = k

            particle.constraints.append(
                make_elastic_constraint(particle, top_neighbor, k_val, dr,k_damp=k_damp[row - 1])
            )

            # Only add reverse constraint if top neighbor is NOT a pivot (row 0)
            neighbor_row = row - 1
            if neighbor_row != 0:
                top_neighbor.constraints.append(
                    make_elastic_constraint(top_neighbor, particle, k_val, dr, k_damp=k_damp[row - 1])
                )

    # simulation.particles.extend(particles)

    return particles


def create_string(
    anchor: tuple | list | torch.Tensor,
    n: int,
    step: float | torch.Tensor,
    k: float,
    **kwargs,
):
    particles = create_mesh(anchor, 1, n, step, k, **kwargs)

    return particles


def create_fibonacci_spiral_string(
    center: tuple | list | torch.Tensor,
    n: int,
    k: float,
    a: float = 1.0,
    theta_step: float = math.pi / 8,
    **kwargs,
):
    """
    Create a string (chain) of particles arranged along a golden/Fibonacci spiral.

    Uses the golden spiral parametrization r(theta) = a * phi^(2*theta/pi), where
    the radius scales by the golden ratio phi every quarter turn (theta += pi/2).
    This is the standard continuous approximation of the Fibonacci spiral.

    Args:
        simulation: target simulation object (kept for API parity with create_mesh)
        center: position of the spiral's origin; also the fixed pivot particle
        n: number of particles in the chain
        k: elastic constant connecting consecutive particles
        a: scale factor controlling the spiral's starting radius
        theta_step: angle (radians) between consecutive particles along the spiral;
            smaller = tighter, more particles per turn
        kwargs: forwarded to constraint construction, same as create_mesh:
            * m (float): particle mass
            * dr (float): rest distance override for elastic constraints
                (defaults to each segment's actual initial length, since spacing
                grows along the spiral and a single fixed dr would fight that shape)
            * dampening (float): dampening constant
            * g (torch.Tensor): gravity vector

    Returns:
        list[Particle]: particles ordered from the spiral's center outward
    """
    center = torch.as_tensor(center, dtype=torch.float64)
    phi = (1 + torch.sqrt(torch.tensor(5.0))) / 2
    b = torch.log(phi) / (torch.pi / 2)

    particles = []
    for i in range(n):
        theta = i * theta_step
        r = a * torch.exp(b * theta)
        pos = center + r * torch.stack((torch.cos(torch.as_tensor(theta)), torch.sin(torch.as_tensor(theta))))
        particles.append(Particle(m=kwargs.get("m", 1.0), x=pos))

    dr_override = kwargs.get("dr", None)

    for i, particle in enumerate(particles):
        if i == 0:
            continue  # pivot, like row 0 in create_mesh

        particle.constraints.append(
            make_gravitational_constraint(particle, kwargs.get("g", torch.tensor([0.0, 500.0])))
        )
        particle.constraints.append(
            make_dampening_constraint(particle, kwargs.get("dampening", 0.02))
        )

        prev_particle = particles[i - 1]
        segment_dr = (
            dr_override
            if dr_override is not None
            else torch.linalg.vector_norm(particle.x - prev_particle.x).item()
        )

        particle.constraints.append(
            make_elastic_constraint(particle, prev_particle, k, segment_dr)
        )

        if i - 1 != 0:
            prev_particle.constraints.append(
                make_elastic_constraint(prev_particle, particle, k, segment_dr)
            )

    return particles


from collections.abc import Iterable


def create_curling_string(
    anchor: tuple | list | torch.Tensor,
    n: int,
    step: float | torch.Tensor,
    k: float,
    theta0: float | Iterable,
    torsion_k: float | Iterable,
    **kwargs,
):
    """
    Create a straight string of particles that curls via torsion spring
    constraints at each interior joint.

    Args:
        simulation: kept for API parity, unused
        anchor: position of the first (pivot) particle
        n: number of particles in the chain
        step: distance between initially-placed particles (chain starts straight)
        k: elastic constant for the stretch constraints along the chain
        theta0: target angle (radians) for each joint's torsion spring;
            pi = stay straight, values further from pi = tighter curl
        torsion_k: torsion spring constant controlling curl stiffness/speed
        kwargs: forwarded, same as create_string:
            * m, dr, dampening, g, epsilon (torsion spring epsilon)

    Returns:
        list[Particle]: particles ordered from anchor outward
    """
    particles = create_string(anchor, n, step, k, **kwargs)

    epsilon = kwargs.get("epsilon", 1e-4)

    # Ensure theta0 and torsion_k are iterable
    if isinstance(theta0, (float, int)):
        theta0 = [theta0] * (n - 2)
    if isinstance(torsion_k, (float, int)):
        torsion_k = [torsion_k] * (n - 2)

    for i, t0, tk in zip(range(1, n - 1), theta0, torsion_k):
        central = particles[i]
        outer_1 = particles[i + 1]
        outer_2 = particles[i - 1]

        c_constraint, o1_constraint, o2_constraint = make_torsion_spring_constraint(
            central, outer_1, outer_2, t0, tk, epsilon=epsilon
        )

        central.constraints.append(c_constraint)
        outer_1.constraints.append(o1_constraint)
        outer_2.constraints.append(o2_constraint)

    # particles[0].constraints = []
    return particles
