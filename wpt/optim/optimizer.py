import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from wpt.nn.model import MLP, TrainDataset, HP_KEYS, pick_shard_with_all_forces

def fit_hyperparams_single_node(
    model: MLP,
    shard_path: Path,
    node_idx: int = -1,
    lr: float = 1e-2,
    n_steps: int = 500,
) -> pd.DataFrame:
    """
    Fits the hyperparameters of a pre-trained MLP model using only a single node's
    observed trajectory across time.
    
    Args:
        model: Pre-trained MLP model.
        shard_path: Path to the JSON simulation shard representing the test case.
        node_idx: The index of the node to follow (e.g., -1 for the tip).
        lr: Learning rate for the hyperparameter optimizer.
        n_steps: Number of optimization steps.
    """
    with open(shard_path) as f:
        real_records = json.load(f)

    # 1. Load the dataset tracking only the specified node
    real_dataset = TrainDataset(real_records, node_idx=node_idx)
    real_loader = DataLoader(real_dataset, batch_size=len(real_dataset), shuffle=True)

    # 2. Freeze network weights and enable hyperparameter gradients
    model.freeze_network()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Detect the device the model is currently mapped to (e.g., CPU, CUDA, MPS)
    device = next(model.parameters()).device

    # 3. Fit hyperparameters
    for step in range(n_steps):
        for x_batch, _hp_ignored, y_batch in real_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(x_batch, hp=None)
            loss = nn.functional.mse_loss(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 4. Compare True vs Recovered HPs
    true_hp = real_dataset.hp.tolist()
    recovered_hp = [model.hyper_params[k].item() for k in HP_KEYS]
    abs_error = [abs(t - r) for t, r in zip(true_hp, recovered_hp)]

    comparison = pd.DataFrame(
        {
            "hyper_param": list(HP_KEYS),
            "true_value": true_hp,
            "recovered_value": recovered_hp,
            "abs_error": abs_error,
        }
    )
    return comparison

if __name__ == "__main__":
    shard_dir = Path("data/shards")
    
    # Load manifest and find a suitable shard
    try:
        with open(shard_dir / "manifest.json") as f:
            manifest = json.load(f)
        
        val_fraction = 0.1
        n_val = max(1, int(len(manifest) * val_fraction))
        val_shards = manifest[:n_val]
        
        real_shard_path = pick_shard_with_all_forces(shard_dir, val_shards)
        
        # Instantiate a model
        model = MLP()
        
        # Perform parameter discovery using only the end tip (node_idx=-1)
        results = fit_hyperparams_single_node(model, real_shard_path, node_idx=-1)
        
        pd.set_option("display.float_format", lambda v: f"{v:.6f}")
        print("\nHyper-parameter recovery comparison (Optimized using a single node):")
        print(results.to_string(index=False))
        
    except FileNotFoundError:
        print(f"Error: Could not locate dataset or manifest in '{shard_dir}'.")
        print("Please ensure you run dataset_generator.py first to produce simulation shards.")