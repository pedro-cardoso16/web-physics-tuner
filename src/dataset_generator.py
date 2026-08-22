import numpy as np
import sys
import pandas as pd
from tools import rotate_particles

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
        x = particles[i].xp.tolist()
        v = particles[i].vp.tolist()
        n = len(particles)
        p = i / (n - 1) if n > 1 else 0

        target = particles[i].x

        dt = simulation.dt
        forces_hyperparams = []

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

            forces_hyperparams.append({force_type: variables})

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

    n_nodes = rng.integers(1, 101, dtype=int)
    anchor_point = rng.integers((0, 0), (500, 200))
    step = rng.uniform(0, 100)
    k = rng.uniform(0, 100)

    particles = create_curling_string(
        simulation,
        anchor=anchor_point,
        n=n_nodes,
        step=step,
        k=k,
        theta0=np.deg2rad(20),
        torsion_k=100,
        dr=20,
    )

    simulation.particles = particles
    simulation.build_vectorized_constraints()

    data = []
    for _ in range(kwargs.get("n", kwargs.get("n_iterations", 50))):
        simulation.run(n=1)
        properties = extract_nodes_properties(simulation)
        data.extend(properties)

    x_min = data[0]["input"]["x"][0]
    x_max = x_min
    y_min = data[0]["input"]["x"][1]
    y_max = y_min
    # normalization

    for p in data:
        # find min max values of the x and y coords
        x_min = min(p["input"]["x"][0], x_min, p["target"][0])
        x_max = max(p["input"]["x"][0], x_max, p["target"][0])
        y_min = min(p["input"]["x"][1], y_min, p["target"][1])
        y_max = max(p["input"]["x"][1], y_max, p["target"][1])

    for p in data:
        p["input"]["x"][0] -= x_min
        p["input"]["x"][1] -= y_min
        p["target"][0] -= x_min
        p["target"][1] -= y_min

        x_range = x_max - x_min
        y_range = y_max - y_min

        for i, r in enumerate((x_range, y_range)):
            if r == 0:
                p["input"]["x"][i] = 0
                p["target"][i] = 0
                p["input"]["v"][i] = 0
            else:
                p["input"]["x"][i] /= r
                p["target"][i] /= r
                p["input"]["v"][i] /= r

        p["input"]["n"] = (p["input"]["n"] - 1) / 100

    return data


def generate_dataset(
    file_path: str | None = None,
    seed: int | None = None,
    n_simulations: int = 50,
    normalize: bool = True,
    **kwargs,
):
    data = []
    futures = []

    rng = np.random.default_rng(seed)

    with ProcessPoolExecutor(kwargs.get("max_workers", None)) as executor:
        for _ in range(n_simulations):
            simulation_seed = rng.integers(0, sys.maxsize)

            future = executor.submit(
                execution,
                **{
                    "seed": simulation_seed,
                    "n_iterations": kwargs.get("n_iterations", 50),
                },
            )

            futures.append(future)

        position = 0

        for future in tqdm(
            as_completed(futures),
            desc="Generating synthetic dataset",
            total=len(futures),
            position=position,
        ):
            data.extend(future.result())
            position += 1

    if file_path is not None:
        write_large_json(data, file_path)

    return data


def write_large_json(data, filepath, chunk_size=50000):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i in tqdm(
            range(0, len(data), chunk_size), desc="Writing to json file", unit="chunk"
        ):
            chunk = data[i : i + chunk_size]
            f.write(",\n".join(map(json.dumps, chunk)) + "\n")
        f.write("]\n")


if __name__ == "__main__":
    generate_dataset("data/train_data.json", 15, 5)
