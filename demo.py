"""Run the complete local data, training, retrieval, and simulation pipeline."""

import json
from pathlib import Path

import numpy as np
import torch

from wpt.img.tools import extract_nodes_from_video
from wpt.nn.model import train_model
from wpt.optim.optimizer_parallel import Optimizer
from wpt.utils.dataset_generator import generate_dataset
from wpt.utils.tools import (
    denormalize_data,
    get_metadata_from_file,
    normalize_data_for_neural_net,
    simulate_chain_from_file,
)


if __name__ == "__main__":
    shard_dir = Path("tmp/data/shards")
    video_file = Path("tmp/data/video_output.json")
    normalized_video_file = Path("tmp/data/video_output_normalized.json")
    model_file = Path("models/pinn.pt")

    # Keep the demo self-contained and practical for a local run.
    generate_dataset(
        shard_dir,
        seed=1,
        n_simulations=10,
        n_iterations=30,
        n_nodes_max=10,
        max_workers=1,
    )
    train_model(
        shard_dir,
        model_file,
        n_epochs=1,
        window_fraction=1.0,
        overwrite=True,
        n_workers=0,
    )
    extract_nodes_from_video(
        "media/videos/demo.mp4",
        n_nodes=8,
        save_to_file=video_file,
        thresh=130,
        anchor_pt=np.array([500, 0]),
    )
    normalize_data_for_neural_net(
        video_file,
        normalized_video_file,
        n_nodes_max=10,
    )

    optimizer = Optimizer(model_file, {})
    optimizer.coarse_optimize(
        normalized_video_data=normalized_video_file,
        n_steps=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lambda_consensus=15.0,
        output_file="coarse_retrieval_test.json",
    )

    normalized_metadata = get_metadata_from_file(normalized_video_file)
    n_turns = normalized_metadata["n_iterations"] * 100
    with open(shard_dir / "metadata.json") as metadata_file:
        metadata = json.load(metadata_file)
    metadata |= {
        "total_range": normalized_metadata["total_range"],
        "xy_min": normalized_metadata["xy_min"],
    }
    data = simulate_chain_from_file(
        "coarse_retrieval_test.json",
        n_turns=n_turns,
        dts=[0.001] * n_turns,
        metadata=metadata,
    )

    ground_truth_coords = denormalize_data(
        normalized_metadata["coords"],
        (normalized_metadata["width"], normalized_metadata["height"]),
        (800, 500),
        normalized_metadata["xy_max"],
        normalized_metadata["xy_min"],
    )
    data = denormalize_data(
        data,
        (normalized_metadata["width"], normalized_metadata["height"]),
        (800, 500),
        normalized_metadata["xy_max"],
        normalized_metadata["xy_min"],
    )

    # The interactive Pygame viewer is intentionally opt-in.
    # from wpt.engine.game import run_engine_with_multiple_predefined_chain_paths
    # run_engine_with_multiple_predefined_chain_paths(
    #     [data, ground_truth_coords],
    #     dts=[0.001] * len(data),
    #     loop=True,
    #     framerate=600,
    # )
