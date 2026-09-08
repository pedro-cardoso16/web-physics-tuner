"""
Demo file for showing process pipeline end-to-end
"""
from pathlib import Path
import numpy as np
import tensorboard
import json
# from wpt.optim.optimizer_parallel import Optimizer
from wpt.utils.dataset_generator import generate_dataset
from wpt.optim.optimizer_parallel import Optimizer

from wpt.nn.model import train_model

from wpt.img.tools import extract_nodes_from_video, visualize_nodes_in_video

from wpt.utils.tools import normalize_data_for_neural_net

if __name__ == "__main__":
    # Pre pipeline

    # 1. Generate synthetic dataset for training
    shard_dir = Path("tmp/data/shards")
    # generate_dataset(shard_dir, seed=1, n_simulations=10000, n_iterations=300, n_nodes_max=10, max_workers=6)

    # 2. Train the model
    # train_model(shard_dir, "models/pinn.pt", n_epochs=5, window_fraction=0.15, overwrite=True)

    # Trained model is ready.

    # User only pipeline.

    # 1. Node extraction from video

    # extract_nodes_from_video(
    #     "media/videos/demo.mp4",
    #     n_nodes=8,
    #     save_to_file="tmp/data/video_output.json",
    #     thresh=130,
    #     anchor_pt=np.array([500, 0]),
    # )

    # 1.1. Visualize

    # visualize_nodes_in_video("tmp/data/video_output.json", "media/videos/demo.mp4")

    # Data must be normalized for the hyperparameters retrieval

    normalize_data_for_neural_net(
        "tmp/data/video_output.json",
        "tmp/data/video_output_normalized.json",
        n_nodes_max=10,
    )

    optimizer = Optimizer("models/pinn.pt", {})

    # optimizer.coarse_optimize(
    #     normalized_video_data="tmp/data/video_output_normalized.json",
    #     n_steps=5000,
    #     device="cuda",
    #     lambda_consensus=15.0,
    #     output_file="coarse_retrieval_test.json",
    # )
    # Visualize the trajectory
    from wpt.utils.tools import (
        get_metadata_from_file,
        simulate_chain_from_file,
        denormalize_data,
    )

    normalized_ground_truth_metadata = get_metadata_from_file(
        "tmp/data/video_output_normalized.json"
    )

    n_turns = normalized_ground_truth_metadata["n_iterations"]*100

    total_range = normalized_ground_truth_metadata['total_range']

    metadata = json.load(open(shard_dir / "metadata.json")) | {"total_range": total_range} | {"xy_min": normalized_ground_truth_metadata['xy_min']}
    data = simulate_chain_from_file(
        "coarse_retrieval_test.json", n_turns=n_turns, dts=[0.1] * n_turns, metadata=metadata
    )
    

    # Ground truth is denormalized using its own original ranges
    ground_truth_coords = denormalize_data(
        normalized_ground_truth_metadata["coords"],
        (
            normalized_ground_truth_metadata["width"],
            normalized_ground_truth_metadata["height"],
        ),
        (800, 500),
        normalized_ground_truth_metadata["xy_max"],
        normalized_ground_truth_metadata["xy_min"],
    )

    # Simulation data is denormalized using its own simulated ranges to avoid mirroring
    coords_np = np.array(data)
    sim_min = coords_np.min(axis=0)
    sim_max = coords_np.max(axis=0)

    data = denormalize_data(
        data,
        (
            normalized_ground_truth_metadata["width"],
            normalized_ground_truth_metadata["height"],
        ),
        (800, 500),
        normalized_ground_truth_metadata["xy_max"],
        normalized_ground_truth_metadata["xy_min"],
    )

    from wpt.engine.game import run_engine_with_multiple_predefined_chain_paths

    run_engine_with_multiple_predefined_chain_paths(
        [data, ground_truth_coords], dts=[0.1] * len(data), loop=True, framerate=60
    )
