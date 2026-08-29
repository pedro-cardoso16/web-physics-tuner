import json
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from model import MLP, TrainDataset, VideoDataset, HP_KEYS, pick_shard_with_all_forces
from physics import (
    Simulation,
    Particle,
    make_gravitational_constraint,
    make_elastic_constraint,
    make_torsion_spring_constraint,
)
from presets import create_string

from concurrent.futures import ProcessPoolExecutor


def get_active_keys_for_node(i: int, n_nodes: int) -> list[str]:
    """
    Returns the list of active hyperparameter keys that should be optimized
    for node index i in a rope containing N total nodes.
    Any key not in this list is automatically frozen at 0.0.
    """
    if i == 0:
        return (
            []
        )  # Pivot/anchor node is completely static and has no active constraints

    # All moving nodes experience gravity, dampening, and their parent spring connection
    active = ["g", "dampening_k", "elastic_k_1", "elastic_dr_1"]

    # If not the tip node, it experiences a child spring connection
    if i < n_nodes - 1:
        active.extend(["elastic_k_2", "elastic_dr_2"])

    # Joint centers range from 1 to N-2
    if 1 <= i <= n_nodes - 2:
        active.extend(["torsion_theta0_central", "torsion_k_central"])

    # Outer_1 arms range from 2 to N-1
    if 2 <= i <= n_nodes - 1:
        active.extend(["torsion_theta0_outer1", "torsion_k_outer1"])

    # Outer_2 arms range from 1 to N-3 (as 0 is fully frozen/pivot)
    if 1 <= i <= n_nodes - 3:
        active.extend(["torsion_theta0_outer2", "torsion_k_outer2"])

    return active


def load_video_data(file_path: Path) -> dict:
    with open(file_path) as f:
        data = json.load(f)

    n_nodes = data["n_nodes"]

    if not isinstance(n_nodes, int):
        raise ValueError("The number of nodes n_nodes must be an integer.")

    if n_nodes <= 0:
        raise ValueError("Could not determine node count from video data.")

    return data


from typing import Literal


def optimize_parallel_machines(
    data: Path | dict,
    model: Path = Path("pinn_model.pt"),
    device: Literal["cpu", "cuda"] = "cpu",
    lr: float = 1e-3,
    lambda_consensus: float = 10.0,
    n_steps: int = 10_000,
    exclude_from_optimization: (
        list[str] | None
    ) = None,  # Added list to explicitly exclude HPs
    output_file: str | None = None,
):
    if isinstance(data, Path):
        data = load_video_data(data)

    n = data["n_nodes"]  # Number of machines (one per node).

    # 1. Instantiate  parallel machines and local datasets
    machines = []
    datasets = []
    x_all = []
    y_all = []

    for i in range(n):
        m = MLP().to(device)  # machine
        if model.exists():
            m.load("pinn_model.pt")

        machines.append(m)

        ds = VideoDataset(data, node_index=i)
        datasets.append(ds)
        x_all.append(ds.input_data.to(device))
        y_all.append(ds.label.to(device))

    # --- Step 2 ---
    # region Apply localized parameter configurations based on node positioning
    for i in range(n):
        active_keys = get_active_keys_for_node(i, n)

        # setup_phase2 sets true_values for frozen parameters (which are 0.0 on inactive nodes)
        # and configures gradients accordingly
        machines[i].setup_phase2(
            datasets[i].hp, optimize_keys=active_keys, initial_val=0.5
        )

    # endregion

    # 3. Collect active parameters from all nodes for the joint optimizer
    params_to_optimize = []
    for i in range(1, n):  # Node 0 is pivot and stays completely frozen
        params_to_optimize.extend(
            [p for p in machines[i].parameters() if p.requires_grad]
        )

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    # 4. Joint Optimization Loop
    for step in range(n):
        optimizer.zero_grad()

        # Compute local fitting and boundary penalties
        local_losses = []
        neg_penalties = []
        for i in range(1, n):
            pred = machines[i](x_all[i], hp=None)
            mse_loss = nn.functional.mse_loss(pred, y_all[i])
            # Penalty of negative hyper-parameters.
            neg_penalty = machines[i].get_hyperparameter_penalty()

            local_losses.append(mse_loss)
            neg_penalties.append(neg_penalty)

        total_local_loss = torch.sum(torch.stack(local_losses))
        total_neg_penalty = torch.sum(torch.stack(neg_penalties))

        # Compute Consensus Penalties.
        # Global gravity g consensus.
        g_vals = torch.stack([machines[i].hyper_params["g"] for i in range(1, n)])
        g_consensus = torch.sum((g_vals - torch.mean(g_vals)) ** 2)

        # Global dampening dampening_k consensus
        damp_vals = torch.stack(
            [machines[i].hyper_params["dampening_k"] for i in range(1, n)]
        )
        damp_consensus = torch.sum((damp_vals - torch.mean(damp_vals)) ** 2)

        # Elastic spring consensus (coupling elastic_k_2 at node i-1 to elastic_k_1 at node i)
        elastic_k_diffs = []
        elastic_dr_diffs = []
        for i in range(2, n):
            elastic_k_diffs.append(
                (
                    machines[i].hyper_params["elastic_k_1"]
                    - machines[i - 1].hyper_params["elastic_k_2"]
                )
                ** 2
            )
            elastic_dr_diffs.append(
                (
                    machines[i].hyper_params["elastic_dr_1"]
                    - machines[i - 1].hyper_params["elastic_dr_2"]
                )
                ** 2
            )

        elastic_k_consensus = (
            torch.sum(torch.stack(elastic_k_diffs))
            if elastic_k_diffs
            else torch.tensor(0.0, device=device)
        )
        elastic_dr_consensus = (
            torch.sum(torch.stack(elastic_dr_diffs))
            if elastic_dr_diffs
            else torch.tensor(0.0, device=device)
        )

        # Torsion spring consensus (coupling central joint j with arms at j+1 and j-1)
        torsion_k_penalty = torch.tensor(0.0, device=device)
        torsion_theta_penalty = torch.tensor(0.0, device=device)
        for j in range(1, n - 1):
            k_terms = [machines[j].hyper_params["torsion_k_central"]]
            theta_terms = [machines[j].hyper_params["torsion_theta0_central"]]

            # Node j+1 represents outer_1
            k_terms.append(machines[j + 1].hyper_params["torsion_k_outer1"])
            theta_terms.append(machines[j + 1].hyper_params["torsion_theta0_outer1"])

            # Node j-1 represents outer_2 (only if it is a moving node, j-1 >= 1)
            if j >= 2:
                k_terms.append(machines[j - 1].hyper_params["torsion_k_outer2"])
                theta_terms.append(
                    machines[j - 1].hyper_params["torsion_theta0_outer2"]
                )

            if len(k_terms) > 1:
                k_stack = torch.stack(k_terms)
                torsion_k_penalty += torch.sum((k_stack - torch.mean(k_stack)) ** 2)

                theta_stack = torch.stack(theta_terms)
                torsion_theta_penalty += torch.sum(
                    (theta_stack - torch.mean(theta_stack)) ** 2
                )

        # Joint total loss calculation
        loss = (
            total_local_loss
            + 10.0 * total_neg_penalty
            + lambda_consensus
            * (
                g_consensus
                + damp_consensus
                + elastic_k_consensus
                + elastic_dr_consensus
                + torsion_k_penalty
                + torsion_theta_penalty
            )
        )

        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            print(
                f"Step {step:04d}/{n_steps} | Total Joint Loss: {loss.item():.6f} | Local Fit Loss: {total_local_loss.item():.6f}"
            )

    # region Generate and print individual summary tables for each node

    pd.set_option("display.float_format", lambda v: f"{v:.6f}")

    for i in range(n):
        print(f"\n>>> Node {i} parameter discovery:")
        if i == 0:
            print("  [Stationary Pivot Node - All parameters locked at 0.0]")
            p_true = [0.0] * len(HP_KEYS)
            p_rec = [0.0] * len(HP_KEYS)
            p_err = [0.0] * len(HP_KEYS)
            p_status = ["Static Pivot"] * len(HP_KEYS)
        else:
            active_keys = get_active_keys_for_node(i, n)

            # Re-apply exclusion check for descriptive logging in console printout
            if exclude_from_optimization is not None:
                active_keys = [
                    k for k in active_keys if k not in exclude_from_optimization
                ]

            p_true = []
            p_rec = []
            p_err = []
            p_status = []
            for idx, k in enumerate(HP_KEYS):
                t_val = datasets[i].hp[idx].item()
                r_val = machines[i].hyper_params[k].item()
                p_true.append(t_val)
                p_rec.append(r_val)
                p_err.append(abs(t_val - r_val))

                # Check status
                if k in active_keys:
                    status_str = "Active"
                elif exclude_from_optimization and k in exclude_from_optimization:
                    status_str = "Frozen (True Value)"
                else:
                    status_str = "Frozen (0.0)"
                p_status.append(status_str)

        node_df = pd.DataFrame(
            {
                "hyper_param": list(HP_KEYS),
                "recovered_val": p_rec,
                "status": p_status,
            }
        )
        print(node_df.to_string(index=False))

    # endregion


def optimize_parallel_system(
    shard_path: Path,
    lr: float = 1e-3,
    n_steps: int = 1000,
    lambda_consensus: float = 10.0,
    initial_val: float | dict[str, float] | torch.Tensor | list[float] = 0.5,
    exclude_from_optimization: (
        list[str] | None
    ) = None,  # Added list to explicitly exclude HPs
) -> pd.DataFrame:
    """
    Instantiates N parallel models for all nodes in the simulation and optimizes
    them jointly under consensus and boundary constraints.
    """
    with open(shard_path) as f:
        real_records = json.load(f)

    # Detect the number of nodes (N) in this simulation by looking at when p resets
    n_nodes = 0
    prev_p = -1.0
    for r in real_records:
        p = r["input"]["p"]
        if p < prev_p:
            break
        n_nodes += 1
        prev_p = p

    if n_nodes == 0:
        raise ValueError("Could not determine node count from simulation shard.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "=" * 70)
    print(f"TARGET SIMULATION SHARD: {shard_path.name}")
    print(f"TOTAL NODES TO OPTIMIZE: {n_nodes}")
    print(f"COMPUTATION DEVICE:      {device}")
    print("=" * 70 + "\n")

    # 1. Instantiate N parallel machines and local datasets
    machines = []
    datasets = []
    x_all = []
    y_all = []

    for i in range(n_nodes):
        m = MLP().to(device)
        if Path("pinn_model.pt").exists():
            m.load("pinn_model.pt")
        machines.append(m)

        ds = TrainDataset(real_records, node_idx=i)
        datasets.append(ds)
        x_all.append(ds.data.to(device))
        y_all.append(ds.label.to(device))

    # 2. Apply localized parameter configurations based on node positioning
    for i in range(n_nodes):
        active_keys = get_active_keys_for_node(i, n_nodes)

        # Filter out keys requested to stay frozen at their true ground-truth values
        if exclude_from_optimization is not None:
            active_keys = [k for k in active_keys if k not in exclude_from_optimization]

        # setup_phase2 sets true_values for frozen parameters (which are 0.0 on inactive nodes)
        # and configures gradients accordingly
        machines[i].setup_phase2(
            datasets[i].hp, optimize_keys=active_keys, initial_val=initial_val
        )

    # 3. Collect active parameters from all nodes for the joint optimizer
    params_to_optimize = []
    for i in range(1, n_nodes):  # Node 0 is pivot and stays completely frozen
        params_to_optimize.extend(
            [p for p in machines[i].parameters() if p.requires_grad]
        )

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    # 4. Joint Optimization Loop
    for step in range(n_steps):
        optimizer.zero_grad()

        # Compute local fitting and boundary penalties
        local_losses = []
        neg_penalties = []
        for i in range(1, n_nodes):
            pred = machines[i](x_all[i], hp=None)
            mse_loss = nn.functional.mse_loss(pred, y_all[i])
            neg_penalty = machines[i].get_hyperparameter_penalty()

            local_losses.append(mse_loss)
            neg_penalties.append(neg_penalty)

        total_local_loss = torch.sum(torch.stack(local_losses))
        total_neg_penalty = torch.sum(torch.stack(neg_penalties))

        # Compute Consensus Penalties
        # Global gravity g consensus
        g_vals = torch.stack([machines[i].hyper_params["g"] for i in range(1, n_nodes)])
        g_consensus = torch.sum((g_vals - torch.mean(g_vals)) ** 2)

        # Global dampening dampening_k consensus
        damp_vals = torch.stack(
            [machines[i].hyper_params["dampening_k"] for i in range(1, n_nodes)]
        )
        damp_consensus = torch.sum((damp_vals - torch.mean(damp_vals)) ** 2)

        # Elastic spring consensus (coupling elastic_k_2 at node i-1 to elastic_k_1 at node i)
        elastic_k_diffs = []
        elastic_dr_diffs = []
        for i in range(2, n_nodes):
            elastic_k_diffs.append(
                (
                    machines[i].hyper_params["elastic_k_1"]
                    - machines[i - 1].hyper_params["elastic_k_2"]
                )
                ** 2
            )
            elastic_dr_diffs.append(
                (
                    machines[i].hyper_params["elastic_dr_1"]
                    - machines[i - 1].hyper_params["elastic_dr_2"]
                )
                ** 2
            )

        elastic_k_consensus = (
            torch.sum(torch.stack(elastic_k_diffs))
            if elastic_k_diffs
            else torch.tensor(0.0, device=device)
        )
        elastic_dr_consensus = (
            torch.sum(torch.stack(elastic_dr_diffs))
            if elastic_dr_diffs
            else torch.tensor(0.0, device=device)
        )

        # Torsion spring consensus (coupling central joint j with arms at j+1 and j-1)
        torsion_k_penalty = torch.tensor(0.0, device=device)
        torsion_theta_penalty = torch.tensor(0.0, device=device)
        for j in range(1, n_nodes - 1):
            k_terms = [machines[j].hyper_params["torsion_k_central"]]
            theta_terms = [machines[j].hyper_params["torsion_theta0_central"]]

            # Node j+1 represents outer_1
            k_terms.append(machines[j + 1].hyper_params["torsion_k_outer1"])
            theta_terms.append(machines[j + 1].hyper_params["torsion_theta0_outer1"])

            # Node j-1 represents outer_2 (only if it is a moving node, j-1 >= 1)
            if j >= 2:
                k_terms.append(machines[j - 1].hyper_params["torsion_k_outer2"])
                theta_terms.append(
                    machines[j - 1].hyper_params["torsion_theta0_outer2"]
                )

            if len(k_terms) > 1:
                k_stack = torch.stack(k_terms)
                torsion_k_penalty += torch.sum((k_stack - torch.mean(k_stack)) ** 2)

                theta_stack = torch.stack(theta_terms)
                torsion_theta_penalty += torch.sum(
                    (theta_stack - torch.mean(theta_stack)) ** 2
                )

        # Joint total loss calculation
        loss = (
            total_local_loss
            + 10.0 * total_neg_penalty
            + lambda_consensus
            * (
                g_consensus
                + damp_consensus
                + elastic_k_consensus
                + elastic_dr_consensus
                + torsion_k_penalty
                + torsion_theta_penalty
            )
        )

        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            print(
                f"Step {step:04d}/{n_steps} | Total Joint Loss: {loss.item():.6f} | Local Fit Loss: {total_local_loss.item():.6f}"
            )

    # 5. Generate and print individual summary tables for each node
    print("\n" + "=" * 70)
    print(f"INDIVIDUAL NODE SUMMARIES FOR SIMULATION: {shard_path.name}")
    print("=" * 70)
    pd.set_option("display.float_format", lambda v: f"{v:.6f}")

    for i in range(n_nodes):
        print(f"\n>>> Node {i} parameter discovery:")
        if i == 0:
            print("  [Stationary Pivot Node - All parameters locked at 0.0]")
            p_true = [0.0] * len(HP_KEYS)
            p_rec = [0.0] * len(HP_KEYS)
            p_err = [0.0] * len(HP_KEYS)
            p_status = ["Static Pivot"] * len(HP_KEYS)
        else:
            active_keys = get_active_keys_for_node(i, n_nodes)

            # Re-apply exclusion check for descriptive logging in console printout
            if exclude_from_optimization is not None:
                active_keys = [
                    k for k in active_keys if k not in exclude_from_optimization
                ]

            p_true = []
            p_rec = []
            p_err = []
            p_status = []
            for idx, k in enumerate(HP_KEYS):
                t_val = datasets[i].hp[idx].item()
                r_val = machines[i].hyper_params[k].item()
                p_true.append(t_val)
                p_rec.append(r_val)
                p_err.append(abs(t_val - r_val))

                # Check status
                if k in active_keys:
                    status_str = "Active"
                elif exclude_from_optimization and k in exclude_from_optimization:
                    status_str = "Frozen (True Value)"
                else:
                    status_str = "Frozen (0.0)"
                p_status.append(status_str)

        node_df = pd.DataFrame(
            {
                "hyper_param": list(HP_KEYS),
                "true_value": p_true,
                "recovered_val": p_rec,
                "abs_error": p_err,
                "status": p_status,
            }
        )
        print(node_df.to_string(index=False))

    # 6. Extract and Average parameters over nodes where they were active
    recovered_vals = {k: [] for k in HP_KEYS}
    true_vals = {k: [] for k in HP_KEYS}

    for i in range(1, n_nodes):
        active_keys = get_active_keys_for_node(i, n_nodes)
        for k in HP_KEYS:
            # We average over keys that were active (or frozen at their true values)
            # excluding parameters that were intentionally forced to 0.0 by boundaries
            if k in active_keys:
                recovered_vals[k].append(machines[i].hyper_params[k].item())
                true_vals[k].append(datasets[i].hp[HP_KEYS.index(k)].item())

    avg_recovered = {}
    avg_true = {}
    for k in HP_KEYS:
        if recovered_vals[k]:
            avg_recovered[k] = sum(recovered_vals[k]) / len(recovered_vals[k])
            avg_true[k] = sum(true_vals[k]) / len(true_vals[k])
        else:
            avg_recovered[k] = 0.0  # Kept frozen at 0.0 as expected physically
            avg_true[k] = 0.0

    # 7. Build Global Comparison Table
    abs_error = [abs(avg_true[k] - avg_recovered[k]) for k in HP_KEYS]
    comparison = pd.DataFrame(
        {
            "hyper_param": list(HP_KEYS),
            "true_value_avg": [avg_true[k] for k in HP_KEYS],
            "recovered_value_avg": [avg_recovered[k] for k in HP_KEYS],
            "abs_error": abs_error,
        }
    )
    return comparison


class Optimizer:
    def __init__(self, data: dict, model: str | torch.Tensor) -> None:
        self.data = data
        self.simulations = []
        pass

    def coarse_optimize(self):
        """Coarse Optimize

        Optimization using MLP parallel execution

        """

        from model import VideoDataset

        shared_data = VideoDataset.load_data("output_normalized.json")


        # Create the instances of dataset for each node in the data.

        optimize_parallel_machines(shared_data)

        # Execute in parallel for each one of the nodes.

        pass

    def fine_optimize(self, hyper_parameters: dict | None = None, n_frames: int = 10):

        # assume the information is in pytorch form
        self.data[""]

        n_nodes = 20

        for i in range(0, len(self.data["frames"]), n_frames - 1):
            simulation = Simulation()
            particles = create_string(
                simulation,
                [0, 0],
                n_nodes,
                1,
                1,
                **hyper_parameters,
            )

            for j, particle in enumerate(particles):
                particle.x[:] = self.data["frames"][i]["nodes"][j]
                particle.v[:] = self.data["frames"][i]["velocities"][j]

            # simulation.v[:] = self.data["frames"][i+1]['dt']

            self.simulations.append(simulation)

        with ProcessPoolExecutor() as executor:
            pass

    def optimize(self):
        self.coarse_optimize()
        self.fine_optimize()


def fit_hyper_parameters(
    start_coords,
    start_velocities,
    simulation: Simulation | None,
    start_hyper_parameters: list[dict] = [{}],
):
    n_nodes = len(start_coords)
    if simulation is None:
        simulation = Simulation()
    else:
        simulation.clear()

    particles = [
        Particle(1.0, start_coords[i], start_velocities[i]) for i in range(n_nodes)
    ]

    for i, (node_hyper_parameters, particle) in enumerate(
        zip(start_hyper_parameters, particles)
    ):
        constraints = []

        keys = get_active_keys_for_node(i, n_nodes)

        if "g" in keys:
            constraints.append(
                make_gravitational_constraint(particle, node_hyper_parameters["g"])
            )
        constraints.append(
            make_elastic_constraint(
                particle,
                particles[i + 1],
                node_hyper_parameters["elastic_k_1"],
                node_hyper_parameters["elastic_dr_1"],
            )
        )
        constraints.append(
            make_elastic_constraint(
                particle,
                particles[i - 1],
                node_hyper_parameters["elastic_k_2"],
                node_hyper_parameters["elastic_dr_2"],
            )
        )

        particle.constraints.extend(constraints)


if __name__ == "__main__":
    shard_dir = Path("data/shards")

    # --- PHASE 2 CONFIGURATION: EASY TO SET ---
    # Customize the starting guesses for your parameter discovery below.
    # Can be a single float (e.g. 0.5) OR a custom dict of starting points per parameter.
    OPTIMIZER_INITIAL_VALUES = 0.5

    # Specify any hyperparameter keys you want to freeze at their true ground-truth values.
    # The script will load their true values and prevent them from being updated during optimization.
    KEYS_TO_FREEZE = []  # Freeze gravity to its correct global value

    try:
        with open(shard_dir / "manifest.json") as f:
            manifest = json.load(f)

        val_fraction = 0.1
        n_val = max(1, int(len(manifest) * val_fraction))
        val_shards = manifest[:n_val]

        real_shard_path = pick_shard_with_all_forces(shard_dir, val_shards)

        # Run parallel multi-node optimization across the whole system
        results = optimize_parallel_system(
            real_shard_path,
            lr=1e-3,
            n_steps=3000,
            lambda_consensus=10.0,
            initial_val=OPTIMIZER_INITIAL_VALUES,
            exclude_from_optimization=KEYS_TO_FREEZE,  # Keep gravity frozen at its true value
        )

        print("\n" + "=" * 70)
        print(f"CONSENSUS AVERAGED REPORT FOR SIMULATION: {real_shard_path.name}")
        print("=" * 70)
        print(results.to_string(index=False))

    except FileNotFoundError:
        print(f"Error: Could not locate dataset or manifest in '{shard_dir}'.")
        print(
            "Please ensure you run dataset_generator.py first to produce simulation shards."
        )
