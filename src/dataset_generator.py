import numpy as np
from copy import deepcopy
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
        x = particles[i].xp
        v = particles[i].vp
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

            forces_hyperparams.append({force_type: variables})

        data.append(
            {
                "input": {"x": x, "v": v, "p": p, "n": n, "dt": dt},
                "target": target,
                "forces_hyperparams": forces_hyperparams,
            }
        )

    return data


def execution():
    simulation = Simulation()
    rng = np.random.default_rng(seed=0)
    n_nodes: int = random.randint(1, 101)
    anchor_point = np.random.randint((0, 0), (500, 200))
    step = np.random.uniform(0, 100)
    particles = create_curling_string(
        simulation,
        anchor=anchor_point,
        n=n_nodes,
        step=rng.uniform(0, 100),
        k=20,
        theta0=np.deg2rad(20),
        torsion_k=100,
    )

    simulation.particles = particles
    data = []
    for _ in range(500):
        simulation.run(n=1)

        data.extend(extract_nodes_properties(simulation))

    return data


def generate_dataset(file_path, seed: int = 0):
    # Create simulation
    rng = np.random.default_rng(seed=seed)

    n_nodes: int = random.randint(1, 101)
    anchor_point = np.random.randint((0, 0), (500, 200))
    step = np.random.uniform(0, 100)

    data = []

    futures = []
    with ProcessPoolExecutor() as executor:
        for _ in range(100):
            future = executor.submit(execution)

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

    return data


def save_to_json(file: str) -> None:
    data = []
    with open(file, "w") as f:

        data.append(
            {
                "input": {"x": None, "v": None, "p": None, "n": None, "dt": None},
                "target": [],
                "hyperparams": {},
            }
        )

        json.dump(data, f)


if __name__ == "__main__":
    print(generate_dataset(None)[250])
