import numpy as np
import sys

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
            variables.pop("outer_particle_1", None)
            variables.pop("outer_particle_2", None)
            variables.pop("central_particle", None)

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
    for _ in range(kwargs.get("n", 100)):
        simulation.run(n=1)

        data.extend(extract_nodes_properties(simulation))

    return data


def generate_dataset(
    file_path: str | None = None, seed: int | None = None, n_simulations: int = 50
):
    # if seed is None:
    #     seed = int(random.randint(0,sys.maxsize))

    data = []
    futures = []

    rng = np.random.default_rng(seed)

    with ProcessPoolExecutor() as executor:
        for _ in range(n_simulations):
            simulation_seed = rng.integers(0, sys.maxsize)

            future = executor.submit(execution, simulation_seed)

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
        write_large_jsonl(data, file_path)

    return data



def write_large_jsonl(data, filepath, chunk_size=50000):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            f.write(",\n".join(map(json.dumps, chunk)) + "\n")
        f.write("]\n")


if __name__ == "__main__":
    generate_dataset("data/train_data.json", 15, 100)
