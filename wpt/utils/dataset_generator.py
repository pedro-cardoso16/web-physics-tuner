import json
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from wpt.utils.tools import rotate_particles
from wpt.engine.physics import Simulation
from wpt.engine.presets import create_curling_string


def extract_nodes_properties(simulation: Simulation) -> list[dict]:
    particles = simulation.particles
    data = []

    for i in range(len(particles)):
        m = particles[i].m
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
                "hyperparams": {"m": m},
            }
        )

    return data


def execution(**kwargs):
    default_kwargs = {
        "seed": None,
        "n_nodes_max": 23,
        "n_nodes_min": 3,
        "k_max": 100,
        "k_min": 20,
        "dampening_k_max": 10,
        "dampening_k_min": 0.1,
        "g_max": 1,
        "g_min": 0,
        "torsion_k_max": 0.2,
        "torsion_k_min": 0,
        "torsion_angle_max": 1.5 * np.pi,
        "torsion_angle_min": 0.5 * np.pi,
        "dt_max": 0.001,
        "step_max": 20,
        "step_min": 0.001,
        "m_max": 1.0,
        "m_min": 0.1,
    }

    default_kwargs |= kwargs

    simulation = Simulation()
    rng = np.random.default_rng(seed=default_kwargs["seed"])

    n_nodes_max = default_kwargs["n_nodes_max"]
    n_nodes_min = default_kwargs["n_nodes_min"]

    n_nodes = rng.integers(n_nodes_min, n_nodes_max, dtype=int)
    # anchor_point = rng.integers((0, 0), (500, 200))
    anchor_point = (0, 0)
    step = rng.uniform(
        default_kwargs["step_min"], default_kwargs["step_max"]
    )  # base distance between consecutive nodes
    k = rng.uniform(default_kwargs["k_min"], default_kwargs["k_max"])
    dampening_k = rng.uniform(
        default_kwargs["dampening_k_min"], default_kwargs["dampening_k_max"]
    )
    g = rng.uniform(default_kwargs["g_min"], default_kwargs["g_max"])
    torsion_k = rng.uniform(
        default_kwargs["torsion_k_min"], default_kwargs["torsion_k_max"], n_nodes - 2
    )
    angle_rad = rng.uniform(
        default_kwargs["torsion_angle_min"],
        default_kwargs["torsion_angle_max"],
        n_nodes - 2,
    )
    dt_max = default_kwargs["dt_max"]
    particles = create_curling_string(
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

    for p in particles:
        p.m = rng.uniform(default_kwargs["m_min"], default_kwargs["m_max"])

    rotate_particles(
        *particles, pivot=particles[0].x, angle_rad=rng.uniform(-np.pi / 2, np.pi / 2)
    )

    simulation.particles = particles
    simulation.build_vectorized_constraints()

    data = []
    n_iterations = kwargs.get("n_iterations", 50)
    for _ in range(n_iterations):
        dt = float(rng.uniform(0, dt_max))  # now it shouldn't break :)
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
        d["input"]["dt"] /= dt_max
        d["input"]["x"][0] -= x_min
        d["input"]["x"][1] -= y_min
        d["target"][0] -= x_min
        d["target"][1] -= y_min

        x_range = x_max - x_min
        y_range = y_max - y_min
        total_range = max(x_max, y_max) - min(x_min, y_min)

        # Old normalization type
        # for i, r in enumerate((x_range, y_range)):
        #     if r == 0:
        #         d["input"]["x"][i] = 0
        #         d["target"][i] = 0
        #         d["input"]["v"][i] = 0
        #     else:
        #         d["input"]["x"][i] /= r
        #         d["target"][i] /= r
        #         d["input"]["v"][i] /= r

        for i in range(2):
            if total_range == 0:
                d["input"]["x"][i] = 0
                d["target"][i] = 0
                d["input"]["v"][i] = 0
            else:
                d["input"]["x"][i] /= total_range
                d["target"][i] /= total_range
                d["input"]["v"][i] /= total_range

        d["input"]["n"] = (d["input"]["n"] - n_nodes_min) / (n_nodes_max - n_nodes_min)

        for key in d["forces_hyperparams"].keys():
            k = d["forces_hyperparams"][key]
            match key:
                case "dampening_force":
                    k["k"] /= default_kwargs["dampening_k_max"]
                case "elastic_force_1" | "elastic_force_2":
                    k["dr"] /= total_range
                    k["k"] /= default_kwargs["k_max"]
                    # k["k"] /= 100
                case (
                    "torsion_spring_outer_1"
                    | "torsion_spring_outer_2"
                    | "torsion_spring_central"
                ):
                    k["theta0"] /= 2 * np.pi
                    k["k"] /= default_kwargs["torsion_k_max"]
                # case "torsion_spring_central":
                #     k["theta0"] /= 2 * np.pi
                # case "torsion_spring_outer_2":
                #     k["theta0"] /= 2 * np.pi
                case "gravitational_force":
                    k["g"][1] /= default_kwargs["g_max"]

    file_path = kwargs.get("output_file", None)
    metadata = default_kwargs.copy()

    if file_path:
        file_path = Path(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return n_nodes * n_iterations

    return data


def generate_dataset(
    shard_dir: str | Path,
    seed: int | None = None,
    n_simulations: int = 50,
    clean: bool = False,
    **kwargs,
) -> None:
    """Generate dataset

    Generate the synthetic training dataset for the MLP neural net. All the values
    are normalized in range [0, 1] each one being normalized in the appropiate way for
    the Neural net processing.

    Args:
        shard_dir (str | Path): Path to the shard directory where simulations
            shards are saved.
        seed (int | None): The seed for base random number generator. Defaults to `None`.
        n_simulations (int): How many simulations to run. Defaults to `50`.
        kwargs (Any): Additional configurations
            - n_iterations (int): Number of iterations (turns) per simulation. Defaults to `50`.
            - max_workers (int | None): Max numbers of workers in processing pool execution.
                Defaults to `None`.

    Returns:
        None
    """

    default_kwargs = {
        "seed": None,
        "n_nodes_max": 23,
        "n_nodes_min": 3,
        "k_max": 100,
        "k_min": 20,
        "dampening_k_max": 10,
        "dampening_k_min": 0.1,
        "g_max": 1,
        "g_min": 0,
        "torsion_k_max": 0.2,
        "torsion_k_min": 0,
        "torsion_angle_max": 1.5 * np.pi,
        "torsion_angle_min": 0.5 * np.pi,
        "dt_max": 0.001,
        "step_max": 20,
        "step_min": 0.001,
        "m_max": 1.0,
        "m_min": 0.1,
    }

    default_kwargs |= kwargs

    shard_dir = Path(shard_dir)

    if clean and input(f"Do you want to delete {shard_dir} [Y/n]").lower() in ['y', '']:
        shutil.rmtree(shard_dir)

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
                    **default_kwargs,
                },
            )
            futures[future] = sim_idx

        for future in tqdm(
            as_completed(futures),
            desc="Generating synthetic dataset",
            total=len(futures),
            unit="simulation",
        ):
            sim_idx = futures[future]
            del futures[future]
            n_records = future.result()  # only THIS simulation's records in memory

            shard_path = Path(shard_dir) / f"sim_{sim_idx:06d}.json"

            manifest.append({"shard": shard_path.name, "n_records": n_records})

    with open(Path(shard_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f)

    metadata = dict(default_kwargs)

    metadata_file_path = Path(shard_dir) / "metadata.json"
    with open(
        metadata_file_path, "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f)

    from wpt.utils.compile_and_stack import compile_and_stack_dataset

    compile_and_stack_dataset(
        shard_dir=Path(shard_dir),
        pt_out_dir=Path(shard_dir).parent / "pt_shards",
        max_workers=None,
    )


if __name__ == "__main__":
    generate_dataset(
        "tmp/data/shards", seed=1, n_simulations=200, n_iterations=200, max_workers=2
    )
    # generate_dataset("data/shards_test", seed=1, n_simulations=10, n_iterations=400, max_workers=2)
