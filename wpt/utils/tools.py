import numpy as np
import pandas as pd
import json
import os
from src.engine.physics import Particle
from numpy.typing import ArrayLike
from src.engine.physics import Simulation, Particle


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
    default_kwargs = {
        "dt_min": 0.001,
        "dt_max": 0.01,
        "n_nodes_max": 103,
        "n_nodes_min": 3,
    }
    kwargs = default_kwargs | kwargs

    with open(file) as f:
        data = json.load(f)
        output_data = data.copy()

    n_nodes = len(data["frames"][0]["nodes"])

    dts = np.array([frame["dt"] for frame in data["frames"]])
    dt_max = float(dts.max())

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

        frame_data["dt"] = frame_data["dt"] / dt_max
        # frame_data["dt"] = (frame_data["dt"] - kwargs["dt_min"]) / (
        #     kwargs["dt_max"] - kwargs["dt_min"]
        # )

    output_data["n_nodes"] = n_nodes
    output_data["n_nodes_normalized"] = (n_nodes - kwargs["n_nodes_min"]) / (
        kwargs["n_nodes_max"] - kwargs["n_nodes_min"]
    )
    output_data["original_range"] = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    output_data["dt_max"] = dt_max

    with open(output_file, "w") as f:
        json.dump(output_data, f)


from pathlib import Path
from collections.abc import Iterable


def show_trajectory(
    data: Iterable,
    ground_truth_data: Iterable,
    dts: list,
    original_resolution,
    new_resolution,
    xy_min,
    xy_max,
    **kwargs
):
    from src.engine.game import run_engine_with_multiple_predefined_chain_paths

    normalized_data = denormalize_data(
        data, original_resolution, new_resolution, xy_max, xy_min
    )
    normalized_ground_truth_data = denormalize_data(
        data, original_resolution, new_resolution, xy_max, xy_min
    )

    run_engine_with_multiple_predefined_chain_paths(
        [data, ground_truth_data], dts, **kwargs
    )


def get_metadata_from_file(file_path) -> dict:
    from wpt.nn.model import VideoDataset

    data = VideoDataset.load_data(file_path)
    dts = []

    x_min, x_max, y_min, y_max = data["original_range"].values()

    x_range = x_max - x_min
    y_range = y_max - y_min

    xy_range = [x_range, y_range]
    xy_min = [x_min, y_min]
    xy_max = [x_max, y_max]

    original_width = data["resolution"]["width"]
    original_height = data["resolution"]["height"]

    data["resolution"]["height"]

    dt_max = data["dt_max"]
    coords = []
    for frame_data in data["frames"]:
        dts.append(frame_data["dt"] * dt_max)

        coords.append(frame_data["nodes"])

    return {
        "coords": coords,
        "xy_range": xy_range,
        "xy_min": xy_min,
        "xy_max": xy_max,
        "width": original_width,
        "height": original_height,
        "dt_max": dt_max,
        "dts": dts,
    }


def show_chain_trajectory(file_path: str | Path):
    from wpt.nn.model import VideoDataset

    # from physics import Simulation
    from src.engine.game import (
        run_engine,
        draw_connections,
        run_engine_with_predefined_chain_path,
        zoom_transform,
    )

    data = VideoDataset.load_data(file_path)
    dts = []
    coords = []

    x_min, x_max, y_min, y_max = data["original_range"].values()

    x_range = x_max - x_min
    y_range = y_max - y_min

    xy_range = [x_range, y_range]
    xy_min = [x_min, y_min]

    original_width = data["resolution"]["width"]
    original_height = data["resolution"]["height"]

    data["resolution"]["height"]

    dt_max = data["dt_max"]
    for frame_data in data["frames"]:
        dts.append(frame_data["dt"] * dt_max)

        c = np.array(frame_data["nodes"]) * (xy_range) + xy_min

        coords.append(
            zoom_transform(
                *c, zoom_factor=min(800 / original_width, 500 / original_height)
            )
        )

    run_engine_with_predefined_chain_path(coords, dts, loop=True)


from collections.abc import Iterable


def denormalize_data(
    data: Iterable, original_resolution, new_resolution, xy_max, xy_min
):
    from src.engine.game import zoom_transform

    xy_range = np.array(xy_max) - np.array(xy_min)

    total_range = xy_range.max()

    coords = []

    for x in data:
        c = (np.array(x) * total_range) + xy_min

        coords.append(
            zoom_transform(
                *c,
                zoom_factor=min(
                    new_resolution[0] / original_resolution[0],
                    new_resolution[1] / original_resolution[1],
                )
            )
        )

    return coords


def simulate_chain_from_file(
    file_path: str | Path,
    n_turns: int = 10,
    start_frame: int = 0,
    simulation: None | Simulation = None,
    output_file: None | Path | str = None,
    dts: None | Iterable[float] = None,
) -> np.ndarray:
    data = pd.read_json(file_path)

    from src.engine.physics import (
        Particle,
        make_dampening_constraint,
        make_elastic_constraint,
        make_torsion_spring_constraint,
        make_gravitational_constraint,
    )

    if simulation is None:
        # Make by hand, it's just easier.
        particles = []

        simulation = Simulation()

        # Create th particles first.
        for i in range(len(data)):
            node_data = data.iloc[i, :]
            particle = Particle(1.0, node_data["x0"], node_data["v0"])
            particles.append(particle)

        for i in range(len(data)):
            node_data = data.iloc[i, :]
            particle = particles[i]

            particle_above = particles[i - 1] if 0 <= (i - 1) < len(particles) else None
            particle_below = particles[i + 1] if 0 <= (i + 1) < len(particles) else None

            # Dampening
            particle.constraints.append(
                make_dampening_constraint(particle, node_data["dampening_k"])
            )

            # Elastic

            if particle_above:
                particle.constraints.append(
                    make_elastic_constraint(
                        particle,
                        particle_above,
                        node_data["elastic_k_1"],
                        node_data["elastic_dr_1"],
                    )
                )
            if particle_below:
                particle.constraints.append(
                    make_elastic_constraint(
                        particle,
                        particle_below,
                        node_data["elastic_k_2"],
                        node_data["elastic_dr_2"],
                    )
                )

            # Torsion
            if particle_below and particle_above:
                torsion_constraints = make_torsion_spring_constraint(
                    particle,
                    particle_below,
                    particle_above,
                    node_data["torsion_theta0_central"],
                    node_data["torsion_k_central"],
                )

                particle.constraints.append(torsion_constraints[0])
                particle_below.constraints.append(torsion_constraints[1])
                particle_above.constraints.append(torsion_constraints[2])

            # Gravity
            particle.constraints.append(
                make_gravitational_constraint(particle, np.array((0,node_data["g"])))
            )

        simulation = Simulation(particles)
    else:
        for i in range(len(data)):
            node_data = data.iloc[i, :]

            simulation.particles[i].x[:] = node_data["x0"]
            simulation.particles[i].xp = None
            simulation.particles[i].v[:] = node_data["v0"]
            simulation.particles[i].vp[:] = node_data["v0"]

    simulation.build_vectorized_constraints()
    data = []   
    data.append([p.x.tolist() for p in simulation.particles])

    for i in range(n_turns):
        if dts is not None:
            simulation.dt = tuple(dts)[i]
        simulation.run(n=1)
        data.append([p.x.tolist() for p in simulation.particles])

    if output_file:
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f)

    return np.array(data)


if __name__ == "__main__":
    # normalize_data_for_neural_net("output.json", "output_normalized.json")

    # show_chain_trajectory("output_normalized.json")
    gt_metadata = get_metadata_from_file("output_normalized.json")
    data = simulate_chain_from_file("coarse_retrieval_test.json", n_turns=int(1e6), dts=[0.001] * int(1e6))


    ground_truth_coords = denormalize_data(
        gt_metadata["coords"],
        (gt_metadata["width"], gt_metadata["height"]),
        (800, 500),
        gt_metadata["xy_max"],
        gt_metadata["xy_min"],
    )

    data = denormalize_data(
        data,
        (gt_metadata["width"], gt_metadata["height"]),
        (800, 500),
        gt_metadata["xy_max"],
        gt_metadata["xy_min"],
    )

    from src.engine.game import run_engine_with_multiple_predefined_chain_paths

    run_engine_with_multiple_predefined_chain_paths(
        [data, ground_truth_coords], dts=[1] * len(data), loop=True, framerate=600
    )
