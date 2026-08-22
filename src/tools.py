import numpy as np
from physics import Particle

from numpy.typing import ArrayLike


def rotate_particles(*args: Particle, pivot: np.ndarray, angle_rad: float) -> None:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    rot_matrix = np.array(
        [
            [c, -s],
            [s, c],
        ]
    )

    for particle in args:
        particle.x = (rot_matrix @ (particle.x - pivot)) + pivot
