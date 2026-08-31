import json
import random
import logging
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, get_worker_info

HP_KEYS = (
    "g",
    "dampening_k",
    "elastic_k_1",
    "elastic_dr_1",
    "elastic_k_2",
    "elastic_dr_2",
    "torsion_theta0_central",
    "torsion_k_central",
    "torsion_theta0_outer1",
    "torsion_k_outer1",
    "torsion_theta0_outer2",
    "torsion_k_outer2",
)


def merge_hp(records: list[dict]) -> torch.Tensor:
    """Merge forces_hyperparams across all records in one rope instance.
    Different nodes report different subsets (anchor reports none, edge
    nodes miss some connections), but values agree wherever present.
    """
    vals = {k: 0.0 for k in HP_KEYS}
    for r in records:
        fh = r.get("forces_hyperparams", {})
        if "gravitational_force" in fh:
            vals["g"] = fh["gravitational_force"]["g"][1]
        if "dampening_force" in fh:
            vals["dampening_k"] = fh["dampening_force"]["k"]
        if "elastic_force_1" in fh:
            vals["elastic_k_1"] = fh["elastic_force_1"]["k"]
            vals["elastic_dr_1"] = fh["elastic_force_1"]["dr"]
        if "elastic_force_2" in fh:
            vals["elastic_k_2"] = fh["elastic_force_2"]["k"]
            vals["elastic_dr_2"] = fh["elastic_force_2"]["dr"]
        if "torsion_spring_central" in fh:
            vals["torsion_theta0_central"] = fh["torsion_spring_central"]["theta0"]
            vals["torsion_k_central"] = fh["torsion_spring_central"]["k"]
        if "torsion_spring_outer_1" in fh:
            vals["torsion_theta0_outer1"] = fh["torsion_spring_outer_1"]["theta0"]
            vals["torsion_k_outer1"] = fh["torsion_spring_outer_1"]["k"]
        if "torsion_spring_outer_2" in fh:
            vals["torsion_theta0_outer2"] = fh["torsion_spring_outer_2"]["theta0"]
            vals["torsion_k_outer2"] = fh["torsion_spring_outer_2"]["k"]
    return torch.tensor([vals[k] for k in HP_KEYS], dtype=torch.float32)


def record_to_input(r: dict) -> torch.Tensor:
    x, y = r["input"]["x"]
    vx, vy = r["input"]["v"]
    p, n, dt = r["input"]["p"], r["input"]["n"], r["input"]["dt"]
    return torch.tensor([x, y, vx, vy, p, n, dt], dtype=torch.float32)


def split_into_rope_instances(records: list[dict]) -> list[list[dict]]:
    """A new rope instance starts whenever p resets (drops) after climbing toward 1.0.
    Only needed for flat, non-sharded files where multiple rope instances are
    concatenated together (e.g. small hand-inspected test files). Sharded
    files produced by the generator already contain exactly one instance each.
    """
    instances: list[list[dict]] = []
    current: list[dict] = []
    prev_p = None
    for r in records:
        p = r["input"]["p"]
        if prev_p is not None and p < prev_p:
            instances.append(current)
            current = []
        current.append(r)
        prev_p = p
    if current:
        instances.append(current)
    return instances


def shard_has_torsion(shard_dir: Path, shard_name: str) -> bool:
    with open(Path(shard_dir) / shard_name) as f:
        records = json.load(f)
    return any(
        "torsion_spring_central" in r.get("forces_hyperparams", {}) for r in records
    )


def pick_shard_with_all_forces(shard_dir: Path, shard_list: list[dict]) -> Path:
    """Return the path of the first shard in shard_list whose rope has
    enough nodes for every force type (notably torsion, which needs >= 3
    nodes) to actually be present. Avoids phase-2 landing on a degenerate
    shard where some hp dimensions never appear (see check_shards.py).
    """
    for entry in shard_list:
        if shard_has_torsion(shard_dir, entry["shard"]):
            return Path(shard_dir) / entry["shard"]

    # Fallback: return the first shard if no shard with torsion is found
    if shard_list:
        return Path(shard_dir) / shard_list[0]["shard"]

    raise ValueError("No shards available in the provided shard list")


class TrainDataset(Dataset):
    """One rope instance's worth of (input, hp, target) triples, held fully in memory.
    Intended for a SINGLE shard (one simulation) — small enough to load whole.
    """

    def __init__(self, records: list[dict], node_idx: int | None = None) -> None:
        super().__init__()

        # 1. Filter the records to track only the specified node per iteration FIRST.
        # This guarantees that inactive forces on this specific node naturally merge to 0.0.
        if node_idx is not None:
            iterations = split_into_rope_instances(records)
            filtered_records = []
            for iter_records in iterations:
                n_nodes = len(iter_records)
                if n_nodes > 0:
                    # Resolve positive and negative indexing relative to this simulation's size
                    idx = node_idx
                    if idx < 0:
                        idx = n_nodes + idx
                    idx = max(0, min(idx, n_nodes - 1))
                    filtered_records.append(iter_records[idx])
            records = filtered_records

        # 2. Merge hyperparameters across ONLY the filtered records representing the active node.
        self.hp = merge_hp(records)  # shared across every row from this rope instance

        n_data = len(records)
        self.data = torch.empty((n_data, 7), dtype=torch.float32)
        self.label = torch.empty((n_data, 2), dtype=torch.float32)

        for i, r in enumerate(records):
            self.data[i] = record_to_input(r)
            self.label[i] = torch.tensor(r["target"], dtype=torch.float32)

    @classmethod
    def from_file(cls, file: str) -> list["TrainDataset"]:
        """Load a flat JSON file containing one or more concatenated rope
        instances (e.g. a small hand-built test file, NOT a full sharded
        dataset). For large datasets, use ShardedRopeDataset instead.
        """
        with open(file) as f:
            records = json.load(f)
        return [cls(instance) for instance in split_into_rope_instances(records)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        return self.data[index], self.hp, self.label[index]


class VideoDataset(Dataset):
    def __init__(self, data_file: str | dict, node_index: int) -> None:
        super().__init__()

        self.data = self.load_data(data_file)

        if not (0 <= node_index < self.__n_nodes):
            raise ValueError(
                f"Invalid node_index value must be between 0 and {self.__n_nodes - 1} (inclusive)."
            )

        self.node_index = node_index

        # Each consecutive pair of frames is equivelent to one input plus label.
        n_frame_data = len(self.data["frames"]) - 1
        self.input_data = torch.empty((n_frame_data, 7), dtype=torch.float32)
        self.label = torch.empty((n_frame_data, 2), dtype=torch.float32)

        for i in range(n_frame_data):
            self.input_data[i] = torch.tensor(
                self.video_data_to_input(self.data, self.node_index, i),
                dtype=torch.float32,
            )
            self.label[i] = torch.tensor(
                self.video_data_to_label(self.data, self.node_index, i),
                dtype=torch.float32,
            )

    @staticmethod
    def video_data_to_input(data: dict, node_index: int, frame_index: int) -> tuple:
        n_nodes = data["n_nodes"]
        n_nodes_normalized = data["n_nodes_normalized"]

        x, y = data["frames"][frame_index]["nodes"][node_index]
        vx, vy = data["frames"][frame_index]["velocity"][node_index]
        dt = data["frames"][frame_index + 1]["dt"]  # dt into the future
        p = node_index / (n_nodes - 1)
        n = n_nodes_normalized

        return x, y, vx, vy, p, n, dt

    @staticmethod
    def video_data_to_label(data: dict, node_index: int, frame_index: int) -> tuple:
        x, y = data["frames"][frame_index + 1]["nodes"][node_index]

        return x, y

    @staticmethod
    def get_num_nodes(data) -> int:
        return len(VideoDataset.load_data(data)["frames"][0]["nodes"])

    @staticmethod
    def load_data(data_file):
        if isinstance(data_file, dict):
            logging.log(0, "Using shared data for VideoDataloader")
            return data_file

        with open(data_file) as f:
            data = json.load(f)

        return data

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, x):
        self.__data = x
        self.__n_nodes = self.data["n_nodes"]
        self.__n_nodes_normalized = self.data["n_nodes_normalized"]  # normalized

    @property
    def n_nodes(self):
        return self.__n_nodes

    @property
    def n_nodes_normalized(self):
        return self.__n_nodes_normalized

    def __len__(self):
        return len(self.data["frames"]) - 1

    def __getitem__(self, index) -> Any:
        return self.input_data[index], self.label[index]


import random
from torch.utils.data import IterableDataset, get_worker_info


class BufferedPtDataset(IterableDataset):
    """
    Hybrid Dataset: Keeps a small buffer of pre-compiled .pt files in memory.
    Shuffles their data globally, trains on them, and then discards them
    to load the next batch of shards.
    """

    def __init__(
        self, pt_dir: str, buffer_size: int = 50, shuffle: bool = True
    ) -> None:
        super().__init__()
        self.pt_paths = sorted(Path(pt_dir).glob("sim_*.pt"))
        if not self.pt_paths:
            raise ValueError(f"No compiled .pt files found in {pt_dir}")

        self.buffer_size = buffer_size
        self.shuffle = shuffle

    def _paths_for_this_worker(self) -> list[Path]:
        paths = list(self.pt_paths)
        if self.shuffle:
            random.shuffle(paths)
        info = get_worker_info()
        if info is None:
            return paths
        return paths[info.id :: info.num_workers]

    def __iter__(self):
        paths = self._paths_for_this_worker()

        # Process the paths in small, memory-safe chunks (buffers)
        for chunk_idx in range(0, len(paths), self.buffer_size):
            chunk_paths = paths[chunk_idx : chunk_idx + self.buffer_size]

            buffer_data = []
            buffer_hp = []
            buffer_label = []

            # 1. Load the small buffer of .pt files into memory
            for path in chunk_paths:
                # torch.load on binary files is near-instantaneous
                shard = torch.load(path)
                buffer_data.append(shard["data"])
                buffer_hp.append(shard["hp"])
                buffer_label.append(shard["label"])

            # 2. Concatenate the buffer into single contiguous tensors
            flat_data = torch.cat(buffer_data)
            flat_hp = torch.cat(buffer_hp)
            flat_label = torch.cat(buffer_label)

            # 3. Shuffle only the active buffer (global-shuffling within the window)
            order = list(range(len(flat_data)))
            if self.shuffle:
                random.shuffle(order)

            # 4. Yield the samples to the training loop
            for idx in order:
                yield flat_data[idx], flat_hp[idx], flat_label[idx]

            # At the end of the chunk loop, the buffer goes out of scope
            # and is immediately garbage-collected, keeping your RAM flat!


import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json

class MappedRopeDataset(Dataset):
    """
    Hybrid, High-Performance Dataset.
    - Stored safely as binary files on disk.
    - Pre-loads the ENTIRE compiled dataset directly into your 32 GB of RAM on 
      startup to maximize training speed (uses exactly ~2.0 GB of RAM) [1].
    - If your dataset grows past your RAM in the future, set load_to_ram=False
      to run directly from your SSD memory-map! [1]
    """
    def __init__(self, pt_dir: str, load_to_ram: bool = True) -> None:
        super().__init__()
        pt_path = Path(pt_dir)
        
        # Load total records count N from metadata
        with open(pt_path / "metadata.json") as f:
            N = json.load(f)["total_records"]
            
        self.N = N
        self.load_to_ram = load_to_ram
        
        if load_to_ram:
            print(f"Pre-loading {N} records (approx 2.0 GB) directly into your 32 GB RAM...")
            # 1. Map the files temporarily to copy their raw binary data
            raw_data = np.memmap(pt_path / "data.bin", dtype="float32", mode="r", shape=(N, 7))
            raw_hp = np.memmap(pt_path / "hp.bin", dtype="float32", mode="r", shape=(N, 12))
            raw_label = np.memmap(pt_path / "label.bin", dtype="float32", mode="r", shape=(N, 2))
            
            # 2. Copy them directly into RAM as native, contiguous PyTorch Tensors
            self.data = torch.from_numpy(raw_data.copy())
            self.hp = torch.from_numpy(raw_hp.copy())
            self.label = torch.from_numpy(raw_label.copy())
            
            # 3. Delete the temporary memmaps to free file handles
            del raw_data, raw_hp, raw_label
            print("Dataset fully cached in RAM. No disk I/O will occur during training!")
            
        else:
            print(f"Memory-mapping {N} records from {pt_dir} (reading directly from SSD on-demand)...")
            self.data_mmap = np.memmap(pt_path / "data.bin", dtype="float32", mode="r", shape=(N, 7))
            self.hp_mmap = np.memmap(pt_path / "hp.bin", dtype="float32", mode="r", shape=(N, 12))
            self.label_mmap = np.memmap(pt_path / "label.bin", dtype="float32", mode="r", shape=(N, 2))

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.load_to_ram:
            # Instantaneous RAM read (Buttery smooth, maximum hardware speed)
            return self.data[index], self.hp[index], self.label[index]
        else:
            # Direct SSD read fallback (Uses 0 MB of RAM if dataset exceeds memory)
            x = torch.from_numpy(self.data_mmap[index].copy())
            hp = torch.from_numpy(self.hp_mmap[index].copy())
            y = torch.from_numpy(self.label_mmap[index].copy())
            return x, hp, y

class InMemoryRopeDataset(Dataset):
    """
    Loads and parses all JSON shards into memory ONCE during startup.
    Keeps everything as compiled PyTorch Tensors in RAM, making training
    epochs run in seconds with 0% disk-I/O overhead.
    """

    def __init__(self, shard_dir: str, shard_names: list[str]) -> None:
        super().__init__()

        all_inputs = []
        all_hps = []
        all_targets = []

        print(f"Pre-loading {len(shard_names)} shards into RAM...")
        for name in tqdm(shard_names, desc="Loading shards", unit="shard"):
            path = Path(shard_dir) / name
            with open(path) as f:
                records = json.load(f)  # Parsed once

            # Merge HPs for this specific shard
            hp = merge_hp(records)

            for r in records:
                all_inputs.append(record_to_input(r))
                all_hps.append(hp)
                all_targets.append(torch.tensor(r["target"], dtype=torch.float32))

        # Stack all lists into single, contiguous, high-performance tensors
        self.data = torch.stack(all_inputs)
        self.hp = torch.stack(all_hps)
        self.label = torch.stack(all_targets)

        print(f"Dataset successfully cached in RAM: {len(self.data)} total samples.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[index], self.hp[index], self.label[index]


class ShardedRopeDataset(IterableDataset):
    """Streams one rope-instance shard (one simulation's JSON file) at a time
    from disk. Never holds more than one shard's records in memory, so large
    datasets that don't fit in RAM can still be trained on.
    """

    def __init__(
        self,
        shard_dir: str,
        shard_names: list[str] | None = None,
        shuffle_shards: bool = True,
    ) -> None:
        super().__init__()
        if shard_names is not None:
            self.shard_paths = [Path(shard_dir) / name for name in shard_names]
        else:
            self.shard_paths = sorted(Path(shard_dir).glob("sim_*.json"))
        if not self.shard_paths:
            raise ValueError(f"No shard files found in {shard_dir}")
        self.shuffle_shards = shuffle_shards

    def _paths_for_this_worker(self) -> list[Path]:
        paths = list(self.shard_paths)
        if self.shuffle_shards:
            random.shuffle(paths)
        info = get_worker_info()
        if info is None:
            return paths
        return paths[info.id :: info.num_workers]  # split shards across workers

    def __iter__(self) -> Iterator:
        for path in self._paths_for_this_worker():
            with open(path) as f:
                records = json.load(f)  # ONE simulation only — small

            hp = merge_hp(records)
            order = list(range(len(records)))
            if self.shuffle_shards:
                random.shuffle(order)

            for i in order:
                r = records[i]
                x7 = record_to_input(r)
                target = torch.tensor(r["target"], dtype=torch.float32)
                yield x7, hp, target
            # `records` and `hp` fall out of scope here, freed before next shard loads


class FiLMLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_hyperparams: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(n_hyperparams, hidden_dim)
        self.to_beta = nn.Linear(n_hyperparams, hidden_dim)

    def forward(self, h: torch.Tensor, hp: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(hp)
        beta = self.to_beta(hp)
        return gamma * h + beta


class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 100, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # phase 2 only: single rope, hp unknown, shared learnable vector
        self.hyper_params = nn.ParameterDict(
            {k: nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) for k in HP_KEYS}
        )
        n_hyperparams = len(HP_KEYS)

        self.fc1 = nn.Linear(7, hidden_dim)
        self.film1 = FiLMLayer(hidden_dim, n_hyperparams)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMLayer(hidden_dim, n_hyperparams)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.film3 = FiLMLayer(hidden_dim, n_hyperparams)
        self.fc_out = nn.Linear(hidden_dim, 2)  # next x, y

    def _shared_hp_tensor(self, batch_size: int, device: torch.device) -> torch.Tensor:
        hp = torch.stack([self.hyper_params[k] for k in HP_KEYS]).to(device)
        return hp.unsqueeze(0).expand(batch_size, -1)

    def forward(self, x: torch.Tensor, hp: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, 7) inputs
        hp: (batch, n_hyperparams) ground-truth hp per example (phase 1),
            or None to use the single shared learnable hp (phase 2).
        """
        if hp is None:
            hp = self._shared_hp_tensor(x.shape[0], x.device)

        h = torch.relu(self.fc1(x))
        h = self.film1(h, hp)
        h = torch.relu(self.fc2(h))
        h = self.film2(h, hp)
        h = torch.relu(self.fc3(h))
        h = self.film3(h, hp)
        return self.fc_out(h)

    def freeze_hyperparams(self) -> None:
        """Phase 1: train the network; shared hyper_params unused here."""
        for p in self.hyper_params.values():
            p.requires_grad_(False)
        for module in [
            self.fc1,
            self.film1,
            self.fc2,
            self.film2,
            self.fc3,
            self.film3,
            self.fc_out,
        ]:
            for param in module.parameters():
                param.requires_grad_(True)

    def freeze_network(self, exclude_hps: list[str] | None = None) -> None:
        """Phase 2: optimize only the shared hyper-params, network fixed.
        Optionally specify a list of hyperparameter keys to exclude from optimization (keep frozen).
        """
        for module in [
            self.fc1,
            self.film1,
            self.fc2,
            self.film2,
            self.fc3,
            self.film3,
            self.fc_out,
        ]:
            for param in module.parameters():
                param.requires_grad_(False)

        exclude_set = set(exclude_hps) if exclude_hps is not None else set()
        for k, p in self.hyper_params.items():
            if k in exclude_set:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)
        self.eval()

    def get_hyperparameter_penalty(self, multiplier: float = 1.0) -> torch.Tensor:
        """Computes a soft L2 penalty for any negative hyperparameters."""
        device = next(self.parameters()).device
        penalty = torch.tensor(0.0, device=device)

        for p in self.hyper_params.values():
            penalty += torch.sum(torch.relu(-p) ** 2)

        penalty *= multiplier

        return penalty

    def save(self, filepath: str | Path) -> None:
        """Save the model's state dictionary to disk."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str | Path) -> None:
        """Load the model's state dictionary from disk."""
        self.load_state_dict(torch.load(filepath, map_location="cpu"))

    def setup_phase2(
        self,
        true_hp: torch.Tensor,
        optimize_keys: list[str] | None = None,
        initial_val: float | dict[str, float] | torch.Tensor | list[float] = 0.0,
    ) -> None:
        """
        Set up Phase 2 in a single call.
        Fills frozen parameters with their true values, initializes optimized parameters
        to initial_val, and configures requires_grad on parameters accordingly.
        If optimize_keys is empty or not given, optimizes all parameters.
        """
        if not optimize_keys:
            optimize_keys = list(HP_KEYS)

        with torch.no_grad():
            for i, k in enumerate(HP_KEYS):
                if k in optimize_keys:
                    # Assign the correct starting value format
                    if isinstance(initial_val, dict):
                        val = initial_val.get(k, 0.0)
                    elif isinstance(initial_val, (list, tuple, torch.Tensor)):
                        val = initial_val[i]
                    else:
                        val = initial_val
                    self.hyper_params[k].copy_(
                        torch.as_tensor(val, dtype=torch.float32)
                    )
                else:
                    self.hyper_params[k].copy_(true_hp[i])

        # Freeze network parameters and exclude non-optimized variables from backpropagation
        exclude_keys = [k for k in HP_KEYS if k not in optimize_keys]
        self.freeze_network(exclude_hps=exclude_keys)

    def setup_hyperparameters_tuning(
        self,
        optimize_keys: list[str] | None = None,
        initial_val: float | dict[str, float] | torch.Tensor | list[float] = 0.5,
    ) -> None:
        """
        Set up coarse optimization in a single call.
        Fills frozen parameters with their true values, initializes optimized parameters
        to initial_val, and configures requires_grad on parameters accordingly.
        If optimize_keys is empty or not given, optimizes all parameters.
        """
        if not optimize_keys:
            optimize_keys = list(HP_KEYS)

        with torch.no_grad():
            for i, k in enumerate(HP_KEYS):
                if k in optimize_keys:
                    # Assign the correct starting value format
                    if isinstance(initial_val, dict):
                        val = initial_val.get(k, 0.5)
                    elif isinstance(initial_val, (list, tuple, torch.Tensor)):
                        val = initial_val[i]
                    else:
                        val = initial_val
                    self.hyper_params[k].copy_(
                        torch.as_tensor(val, dtype=torch.float32)
                    )
                # else:
                #     self.hyper_params[k].copy_(true_hp[i])

        # Freeze network parameters and exclude non-optimized variables from backpropagation
        exclude_keys = [k for k in HP_KEYS if k not in optimize_keys]
        self.freeze_network(exclude_hps=exclude_keys)


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    shard_dir = Path("data/shards")
    model_checkpoint_path = Path("pinn_model.pt")

    try:
        with open(shard_dir / "manifest.json") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find manifest.json in '{shard_dir}'. "
            "Please run 'python dataset_generator.py' to generate shards before running model.py."
        )

    if not manifest:
        raise ValueError(
            "The manifest.json file is empty. Please run dataset_generator.py to generate shards."
        )

    # Gracefully handle split sizes, including datasets with 1 or 2 shards
    val_fraction = 0.1
    n_val = int(len(manifest) * val_fraction)

    if n_val == 0:
        if len(manifest) == 1:
            # Fallback: with only 1 shard, we must train and validate on the same file
            val_shards = manifest
            train_shards = manifest
        else:
            # Fallback: with 2+ shards but n_val rounds to 0, use exactly 1 for validation
            val_shards = manifest[:1]
            train_shards = manifest[1:]
    else:
        val_shards = manifest[:n_val]
        train_shards = manifest[n_val:]

    model = MLP()

    # --- check for existing trained checkpoint to save time ---
    if model_checkpoint_path.exists():
        print(
            f"Loading pre-trained Phase 1 model checkpoint from '{model_checkpoint_path}'..."
        )
        model.load(model_checkpoint_path)
    else:
        print("No checkpoint found. Starting Phase 1 training...")
        # --- phase 1: train the network across many rope instances ---

        # ALTERATION 1: Point directly to your pre-compiled binary .pt shards directory
        pt_dir = Path("data/pt_shards")

        train_dataset = MappedRopeDataset(
            pt_dir, load_to_ram=True  # type: ignore
        )

        # ALTERATION 2: Removed shuffle=True because BufferedPtDataset is an IterableDataset.
        # (Shuffling is handled internally by the dataset's memory buffer!)
        train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=4)

        model.freeze_hyperparams()
        optimizer1 = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        n_epochs = 100
        epoch_bar = tqdm(range(n_epochs), desc="Phase 1 (train)", unit="epoch")
        for epoch in epoch_bar:
            total_loss, n_batches = 0.0, 0
            batch_bar = tqdm(
                train_loader, desc=f"epoch {epoch}", unit="batch", leave=False
            )
            for x_batch, hp_batch, y_batch in batch_bar:
                pred = model(x_batch, hp=hp_batch)
                loss = nn.functional.mse_loss(pred, y_batch)
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                total_loss += loss.item()
                n_batches += 1
                batch_bar.set_postfix(loss=f"{loss.item():.8f}")
            avg_loss = total_loss / n_batches
            epoch_bar.set_postfix(avg_loss=f"{avg_loss:.8f}")

        # Save model after completing training
        print(f"Saving trained Phase 1 model to '{model_checkpoint_path}'...")
        model.save(model_checkpoint_path)

    # --- phase 2: pick ONE held-out shard to simulate the "real, hp unknown" case ---
    real_shard_path = pick_shard_with_all_forces(shard_dir, val_shards)
    with open(real_shard_path) as f:
        real_records = json.load(f)

    # We follow ONLY ONE specific node (node_idx=-1 corresponds to the free end / tip)
    # Because of the updated order, self.hp correctly maps inactive local parameters to 0.0
    real_dataset = TrainDataset(real_records, node_idx=2)
    real_loader = DataLoader(real_dataset, batch_size=len(real_dataset), shuffle=True)

    # --- Phase 2 Configuration: ONE-LINER SETUP ---
    # We choose ["elastic_k_1"] here because node_idx = -1 only has an active parent elastic connection.
    model.setup_phase2(
        real_dataset.hp,
        optimize_keys=[
            "elastic_k_1",
            "elastic_k_2",
            "dampening_k",
            "torsion_k_central",
        ],
        initial_val=0.5,
    )

    # Instantiate Phase 2 optimizer (only receives parameters that require gradients)
    optimizer2 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    n_steps = 1000
    step_bar = tqdm(range(n_steps), desc="Phase 2 (hp fit)", unit="step")
    for step in step_bar:
        for x_batch, _hp_ignored, y_batch in real_loader:
            pred = model(x_batch, hp=None)
            mse_loss = nn.functional.mse_loss(pred, y_batch)

            # Penalize negative hyperparameters
            penalty = model.get_hyperparameter_penalty()
            loss = mse_loss + 10.0 * penalty

            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
        step_bar.set_postfix(loss=f"{loss.item():.8f}")  # type: ignore

    # --- comparison table: true vs recovered hyper-params ---
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
    pd.set_option("display.float_format", lambda v: f"{v:.6f}")
    print("\nHyper-parameter recovery comparison:")
    print(comparison.to_string(index=False))
