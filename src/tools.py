import numpy as np
import pandas as pd
import json

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


def normalize_data_for_neural_net(file: str, output_file: str, **kwargs):

    default_kwargs = {"dt_min": 0.001, "dt_max": 0.02}
    kwargs = default_kwargs | kwargs

    with open(file) as f:
        data = json.load(f)
        output_data = data.copy()

    n_nodes = len(data["frames"][0]["nodes"])

    coords = np.array([frame["nodes"] for frame in data["frames"]])
    coords = coords.reshape((-1, 2))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    x_range, y_range = x_max - x_min, y_max - y_min

    coords = (coords - (x_min, y_min)) / (x_range, y_range)

    velocities = np.array([frame["velocity"] for frame in data["frames"]])
    velocities = velocities.reshape((-1, 2))
    velocities /= (x_range, y_range)

    velocities = velocities.reshape((-1, n_nodes, 2))
    coords = coords.reshape((-1, n_nodes, 2))

    for i, frame_data in enumerate(output_data["frames"]):
        frame_data["nodes"] = coords[i].tolist()
        frame_data["velocity"] = velocities[i].tolist()
        frame_data["dt"] = (frame_data["dt"] - kwargs["dt_min"]) / (
            kwargs["dt_max"] - kwargs["dt_min"]
        )

    with open(output_file, "w") as f:
        json.dump(output_data, f)


if __name__ == "__main__":
    normalize_data_for_neural_net("output.json", "output_normalized.json")
