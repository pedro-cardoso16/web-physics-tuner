import numpy as np
import sys
import pandas as pd
from tools import rotate_particles
from pathlib import Path

# from copy import deepcopy
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from physics import (
    Simulation,
    Particle,
    make_torsion_spring_constraint,
    make_rigid_connection_constraint,
    make_elastic_constraint,
    make_gravitational_constraint,
)
from presets import (
    create_string,
    create_fibonacci_spiral_string,
    create_curling_string,
)


def extract_nodes_properties(simulation: Simulation):
    particles = simulation.particles
    data = []

    for i in range(len(particles)):
        x = particles[i].xp.tolist()  # type: ignore
        v = particles[i].vp.tolist()
        n = len(particles)
        p = i / (n - 1) if n > 1 else 0

        target = particles[i].x

        dt = simulation.dt
        forces_hyperparams = {}
        elastic_count = (
            1  # Distinguish consecutive elastic constraints for the same node
        )

        for c in particles[i].constraints:
            variables = vars(c.reference).copy()

            force_type = c.func.__name__.removesuffix("_wrapper")

            variables.pop("particle", None)
            variables.pop("x1", None)
            variables.pop("x2", None)
            outer_particle_1 = variables.pop("outer_particle_1", None)
            outer_particle_2 = variables.pop("outer_particle_2", None)
            central_particle = variables.pop("central_particle", None)

            if particles[i] == central_particle:
                force_type += "_central"
            elif particles[i] == outer_particle_1:
                force_type += "_outer_1"
            elif particles[i] == outer_particle_2:
                force_type += "_outer_2"

            # Assign sequential labels to multiple elastic connections (e.g., top and bottom)
            if force_type == "elastic_force":
                force_type = f"elastic_force_{elastic_count}"
                elastic_count += 1

            for key, val in variables.items():
                if isinstance(val, np.ndarray):
                    try:
                        variables[key] = val.tolist()
                    except:
                        pass
                if isinstance(val, np.float64):
                    try:
                        variables[key] = val.item()
                    except:
                        pass

            forces_hyperparams[force_type] = variables

        data.append(
            {
                "input": {"x": x, "v": v, "p": p, "n": n, "dt": dt},
                "target": target.tolist(),
                "forces_hyperparams": forces_hyperparams,
            }
        )

    return data


def execution(seed=None, **kwargs):
    simulation = Simulation()
    rng = np.random.default_rng(seed=seed)

    n_nodes = rng.integers(3, 103, dtype=int)
    # anchor_point = rng.integers((0, 0), (500, 200))
    anchor_point = (0, 0)
    step = rng.uniform(0.001, 1)  # base distance between consecutive nodes
    k = rng.uniform(0, 1)
    dampening_k = rng.uniform(0, 1)
    g = rng.uniform(0, 1)
    torsion_k = rng.uniform(0, 1)
    angle_rad = rng.uniform(0, 2 * np.pi)

    particles = create_curling_string(
        simulation,
        anchor=anchor_point,
        n=n_nodes,
        step=step,
        k=k,
        theta0=angle_rad,
        torsion_k=torsion_k,
        dr=step,
        g=np.array([0, g]),
        dampening=dampening_k,
    )

    rotate_particles(*particles, pivot=particles[0].x, angle_rad=rng.uniform(0, np.pi))

    simulation.particles = particles
    simulation.build_vectorized_constraints()

    data = []
    n_iterations = kwargs.get("n_iterations", 50)
    for _ in range(kwargs.get("n", n_iterations)):
        dt = float(rng.uniform(0, 0.01))  # now it shouldn't break :)
        simulation.dt = dt
        simulation.run(n=1)
        properties = extract_nodes_properties(simulation)
        data.extend(properties)

    x_min = data[0]["input"]["x"][0]
    x_max = x_min
    y_min = data[0]["input"]["x"][1]
    y_max = y_min
    # normalization

    for d in data:
        # find min max values of the x and y coords
        x_min = min(d["input"]["x"][0], x_min, d["target"][0])
        x_max = max(d["input"]["x"][0], x_max, d["target"][0])
        y_min = min(d["input"]["x"][1], y_min, d["target"][1])
        y_max = max(d["input"]["x"][1], y_max, d["target"][1])

    for d in data:
        d["input"]["dt"] /= 0.001
        d["input"]["x"][0] -= x_min
        d["input"]["x"][1] -= y_min
        d["target"][0] -= x_min
        d["target"][1] -= y_min

        x_range = x_max - x_min
        y_range = y_max - y_min

        for i, r in enumerate((x_range, y_range)):
            if r == 0:
                d["input"]["x"][i] = 0
                d["target"][i] = 0
                d["input"]["v"][i] = 0
            else:
                d["input"]["x"][i] /= r
                d["target"][i] /= r
                d["input"]["v"][i] /= r

        d["input"]["n"] = (d["input"]["n"] - 3) / 100

        for key in d["forces_hyperparams"].keys():
            k = d["forces_hyperparams"][key]
            match key:
                case "elastic_force_1" | "elastic_force_2":
                    k["k"] = k["k"]
                    k["dr"] = (k["dr"] - 0.01) / 0.99
                case "torsion_spring_outer_1":
                    # k["k"] /= 500
                    k["theta0"] /= 2 * np.pi
                case "torsion_spring_central":
                    # k["k"] /= 500
                    k["theta0"] /= 2 * np.pi
                case "torsion_spring_outer_2":
                    # k["k"] /= 500
                    k["theta0"] /= 2 * np.pi
                case "dampening_force":
                    k["k"] = k["k"]
                # case "gravitational_force":
                # k["g"][1] = (k["g"][1] - 100) / 500

    file_path = kwargs.get("output_file", None)

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return n_nodes * n_iterations

    return data


def generate_dataset(
    shard_dir: str,
    seed: int | None = None,
    n_simulations: int = 50,
    **kwargs,
):
    Path(shard_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = []

    with ProcessPoolExecutor(kwargs.get("max_workers", None)) as executor:
        futures = {}
        for sim_idx in range(n_simulations):
            simulation_seed = rng.integers(0, sys.maxsize)
            future = executor.submit(
                execution,
                **{
                    "seed": simulation_seed,
                    "n_iterations": kwargs.get("n_iterations", 50),
                    "output_file": Path(shard_dir) / f"sim_{sim_idx:06d}.json",
                },
            )
            futures[future] = sim_idx

        for future in tqdm(
            as_completed(futures),
            desc="Generating synthetic dataset",
            total=len(futures),
        ):
            sim_idx = futures[future]
            del futures[future]
            n_records = future.result()  # only THIS simulation's records in memory

            shard_path = Path(shard_dir) / f"sim_{sim_idx:06d}.json"
            # with open(shard_path, "w", encoding="utf-8") as f:
            #     json.dump(records, f)

            manifest.append({"shard": shard_path.name, "n_records": n_records})

            # `records` goes out of scope next loop iteration and gets GC'd

    with open(Path(shard_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    generate_dataset("data/shards", seed=1, n_simulations=10_000, n_iterations=500)
    # generate_dataset("data/shards_test", seed=1, n_simulations=10, n_iterations=400, max_workers=2)
