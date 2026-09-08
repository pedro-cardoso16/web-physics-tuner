import numpy as np
import pandas as pd
import json
import os
from wpt.engine.physics import Particle, Simulation


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
        "dt_max": 0.001,
        "n_nodes_max": 23,
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

    total_range = max(x_max, y_max) - min(x_min, y_min)
    # x_range, y_range = x_max - x_min, y_max - y_min

    coords = (coords - min(x_min, y_min)) / total_range

    velocities = np.array([frame["velocity"] for frame in data["frames"]])
    velocities = velocities.reshape((-1, 2))
    velocities /= total_range

    velocities = velocities.reshape((-1, n_nodes, 2))
    coords = coords.reshape((-1, n_nodes, 2))

    for i, frame_data in enumerate(output_data["frames"]):
        frame_data["nodes"] = coords[i].tolist()
        frame_data["velocity"] = velocities[i].tolist()

        frame_data["dt"] = frame_data["dt"] / dt_max

    output_data["n_nodes"] = n_nodes
    output_data["n_nodes_normalized"] = (n_nodes - kwargs["n_nodes_min"]) / (
        kwargs["n_nodes_max"] - kwargs["n_nodes_min"]
    )
    # Note: k values are max-normalized by dividing by 100 in the dataset generation
    output_data["original_range"] = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "total_range": max(x_max, y_max) - min(x_min, y_min),
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
    **kwargs,
):
    from engine.game import run_engine_with_multiple_predefined_chain_paths

    # Denormalize using the consistent scalar range logic
    normalized_data = denormalize_data(
        data, original_resolution, new_resolution, xy_max, xy_min
    )

    # We use the same denormalization for both to ensure they share the same coordinate space
    normalized_ground_truth_data = denormalize_data(
        ground_truth_data, original_resolution, new_resolution, xy_max, xy_min
    )

    run_engine_with_multiple_predefined_chain_paths(
        [normalized_data, normalized_ground_truth_data], dts, **kwargs
    )


def get_metadata_from_file(file_path) -> dict:
    from wpt.nn.model import VideoDataset

    data = VideoDataset.load_data(file_path)
    dts = []

    x_min, x_max, y_min, y_max, total_range = data["original_range"].values()

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
        "total_range": total_range,
        "width": original_width,
        "height": original_height,
        "dt_max": dt_max,
        "dts": dts,
        "n_iterations": len(data["frames"]),
    }


def show_chain_trajectory(file_path: str | Path):
    from wpt.nn.model import VideoDataset

    # from physics import Simulation
    from engine.game import (
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

        total_range = np.max([x_max, y_max]) - np.min([x_min, y_min])
        offset = np.min([x_min, y_min])
        c = np.array(frame_data["nodes"]) * total_range + offset

        coords.append(
            zoom_transform(
                *c, zoom_factor=min(800 / original_width, 500 / original_height)
            )
        )

    run_engine_with_predefined_chain_path(coords, dts, loop=True)


from collections.abc import Iterable


def denormalize_data(
    data: Iterable,
    original_resolution,
    new_resolution,
    xy_max: np.ndarray,
    xy_min: np.ndarray,
):
    from wpt.engine.game import zoom_transform

    total_range = np.max(xy_max) - np.min(xy_min)
    offset = np.min(xy_min)

    coords = []

    for x in data:
        c = (np.array(x) * total_range) + offset

        coords.append(
            zoom_transform(
                *c,
                zoom_factor=min(
                    new_resolution[0] / original_resolution[0],
                    new_resolution[1] / original_resolution[1],
                ),
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
    metadata: dict = {},
) -> np.ndarray:
    data = pd.read_json(file_path)

    from wpt.engine.physics import (
        Particle,
        make_dampening_constraint,
        make_elastic_constraint,
        make_torsion_spring_constraint,
        make_gravitational_constraint,
    )

    x_min = min(metadata['xy_min']) 
    x_range = metadata["total_range"]

    if simulation is None:
        # Make by hand, it's just easier.
        particles = []
        # first_dt = tuple(dts)[0] if dts is not None else 0.001

        # Create th particles first.
        for i in range(len(data)):
            node_data = data.iloc[i, :]
            mass = node_data["m"] * metadata["m_max"] if i != 0 else 1.0
            particle = Particle(mass, np.array(node_data["x0"]) * x_range + x_min  , np.array(node_data["v0"]) * x_range)

            print(particle.m)
            # particle.xp = particle.x - particle.v * first_dt
            particles.append(particle)

        global_g = data["g"].mean()

        for i in range(len(data)):
            node_data = data.iloc[i, :]
            particle = particles[i]

            if i == 0:
                continue

            particle_above = particles[i - 1] if 0 <= (i - 1) < len(particles) else None
            particle_below = particles[i + 1] if 0 <= (i + 1) < len(particles) else None

            # Dampening
            particle.constraints.append(
                make_dampening_constraint(
                    particle, node_data["dampening_k"] * metadata["dampening_k_max"]*0.001 
                )
            )

            # Elastic

            if particle_above:
                node_data_prev = data.iloc[i - 1, :]

                k_value = (
                    np.mean((node_data["elastic_k_1"], node_data_prev["elastic_k_2"]))
                    if i != 1
                    else node_data["elastic_k_1"]
                )

                dr_value = (
                    np.mean((node_data["elastic_dr_1"], node_data_prev["elastic_dr_2"]))
                    if i != 1
                    else node_data["elastic_dr_1"]
                )

                particle.constraints.append(
                    make_elastic_constraint(
                        particle,
                        particle_above,
                        k_value * metadata["k_max"],
                        dr_value * metadata["total_range"],
                    )
                )

            if particle_below:
                node_data_next = data.iloc[i + 1, :]
                particle.constraints.append(
                    make_elastic_constraint(
                        particle,
                        particle_below,
                        # 100,
                        # 0.1,
                        np.mean(
                            (node_data["elastic_k_2"], node_data_next["elastic_k_1"])
                        )
                        * metadata["k_max"]
                        ,
                        np.mean(
                            (node_data["elastic_dr_2"], node_data_next["elastic_dr_1"])
                        )
                        * metadata["total_range"]
                        ,
                    )
                )

            # --- Torsion ---
            if particle_below and particle_above:
                torsion_constraints = make_torsion_spring_constraint(
                    particle,
                    particle_below,
                    particle_above,
                    node_data["torsion_theta0_central"] * (2 * np.pi),
                    node_data["torsion_k_central"] * metadata["torsion_k_max"],
                )

                particle.constraints.append(torsion_constraints[0])
                particle_below.constraints.append(torsion_constraints[1])

                if i != 1:
                    particle_above.constraints.append(torsion_constraints[2])

            # Gravity
            particle.constraints.append(
                make_gravitational_constraint(
                    particle, np.array((0, global_g * metadata["g_max"] * 10))
                )
            )

        simulation = Simulation(particles)
    else:
        for i in range(len(data)):
            node_data = data.iloc[i, :]

            simulation.particles[i].x[:] = node_data["x0"]
            simulation.particles[i].xp = None
            simulation.particles[i].v[:] = node_data["v0"]
            simulation.particles[i].vp[:] = node_data["v0"]

    data = []
    # data.append([p.x.tolist() for p in simulation.particles])
    data.append([((p.x - x_min)/x_range).tolist() for p in simulation.particles])


    from tqdm import tqdm

    simulation.build_vectorized_constraints()


    for i in tqdm(range(n_turns), desc="Simulating chain", unit="turn"):
        if dts is not None:
            simulation.dt = tuple(dts)[i]

        # ------------------ Test

        # if i == 0:
        # print(
        #     f"Initial velocities: {[np.round(p.v, 3).tolist() for p in simulation.particles[:]]}"
        # )
        # ------------------

        simulation.run(n=1)

        # ------------------ Test
        # if i == 0:
        # print(f"Velocities after 1st turn: {[p.v for p in simulation.particles[:]]}")
        # ------------------

        data.append([((p.x - x_min)/x_range).tolist() for p in simulation.particles])

    if output_file:
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f)

    return np.array(data)

def get_chain_length(*args):
    total_length = 0

    x = np.array(args)

    for i in range(len(x)-1):
        total_length += np.linalg.norm(x[i+1] - x[i])

    return total_length


if __name__ == "__main__":
    # normalize_data_for_neural_net("output.json", "output_normalized.json")

    # show_chain_trajectory("output_normalized.json")
    gt_metadata = get_metadata_from_file("output_normalized.json")
    data = simulate_chain_from_file(
        "coarse_retrieval_test.json", n_turns=int(1e6), dts=[0.001] * int(1e6)
    )

    # Calculate simulation's own ranges to avoid coordinate mirroring/shifting
    coords_np = np.array(data)
    sim_min = coords_np.min(axis=0)
    sim_max = coords_np.max(axis=0)

    ground_truth_coords = denormalize_data(
        gt_metadata["coords"],
        (gt_metadata["width"], gt_metadata["height"]),
        (800, 500),
        sim_max,
        sim_min,
    )

    data = denormalize_data(
        data,
        (gt_metadata["width"], gt_metadata["height"]),
        (800, 500),
        sim_max,
        sim_min,
    )

    from wpt.engine.game import run_engine_with_multiple_predefined_chain_paths

    run_engine_with_multiple_predefined_chain_paths(
        [data, ground_truth_coords], dts=[1] * len(data), loop=True, framerate=600
    )
