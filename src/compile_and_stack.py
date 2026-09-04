import orjson
import shutil
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import numpy as np
from tqdm import tqdm
from model import record_to_input, merge_hp


def compile_single_shard(json_path: Path, temp_pt_dir: Path):
    """Processes a single JSON shard and saves it as a temporary binary .pt file."""
    with open(json_path, "rb") as f:
        records = orjson.loads(f.read())

    hp = merge_hp(records)

    inputs = []
    targets = []
    hps = []

    for r in records:
        inputs.append(record_to_input(r))
        hps.append(hp)
        targets.append(torch.tensor(r["target"], dtype=torch.float32))

    compiled_shard = {
        "data": torch.stack(inputs),
        "hp": torch.stack(hps),
        "label": torch.stack(targets),
    }

    output_file = temp_pt_dir / f"{json_path.stem}.pt"
    torch.save(compiled_shard, output_file)


def compile_and_stack_dataset(
    json_dir: str, pt_out_dir: str, max_workers: int | None = None
):
    json_dir_path = Path(json_dir)
    pt_out_path = Path(pt_out_dir)
    pt_out_path.mkdir(parents=True, exist_ok=True)

    # 1. Read manifest to calculate the exact total records (N) across all files
    manifest_path = json_dir_path / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    N = sum(item["n_records"] for item in manifest)

    # Create temporary folder for individual .pt files
    temp_pt_dir = json_dir_path.parent / "temp_pt_shards"
    temp_pt_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # PHASE 1: Parallel Shard Compilation (Saturates CPU Cores)
    # -------------------------------------------------------------
    print(
        f"Phase 1: Compiling {len(json_paths := sorted(json_dir_path.glob('sim_*.json')))} JSON shards..."
    )
    with ProcessPoolExecutor(max_workers) as executor:
        futures = {
            executor.submit(compile_single_shard, path, temp_pt_dir): path
            for path in json_paths
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Compiling Shards"
        ):
            future.result()  # Propagates exceptions

    # -------------------------------------------------------------
    # PHASE 2: Incremental Writing to Disk (0 MB RAM Spike!)
    # -------------------------------------------------------------
    print(f"\nPhase 2: Pre-allocating {N} records to disk-backed binary files...")

    # Pre-allocate files directly on your SSD
    data_mmap = np.memmap(
        pt_out_path / "data.bin", dtype="float32", mode="w+", shape=(N, 7)
    )
    hp_mmap = np.memmap(
        pt_out_path / "hp.bin", dtype="float32", mode="r+", shape=(N, 12)
    )
    label_mmap = np.memmap(
        pt_out_path / "label.bin", dtype="float32", mode="r+", shape=(N, 2)
    )

    temp_pt_paths = sorted(temp_pt_dir.glob("sim_*.pt"))

    start_idx = 0
    for path in tqdm(temp_pt_paths, desc="Writing in parts to SSD"):
        # Load exactly ONE small binary shard (150 KB)
        shard = torch.load(path, weights_only=True)
        n_samples = len(shard["data"])
        end_idx = start_idx + n_samples

        # Write directly to the SSD-mapped file slice
        data_mmap[start_idx:end_idx] = shard["data"].numpy()
        hp_mmap[start_idx:end_idx] = shard["hp"].numpy()
        label_mmap[start_idx:end_idx] = shard["label"].numpy()

        # Flush the buffer to write changes to disk and free memory
        data_mmap.flush()
        hp_mmap.flush()
        label_mmap.flush()

        start_idx = end_idx

    # Write metadata info file so PyTorch knows the total N on startup
    meta_path = pt_out_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"total_records": N}, f)

    # 3. Clean up the temporary binary shards to save disk space
    print("Cleaning up temporary binary shards...")
    shutil.rmtree(temp_pt_dir)

    print(
        "\nSUCCESS: Dataset compiled, written in parts, and optimized for Memory Mapping!"
    )


if __name__ == "__main__":
    compile_and_stack_dataset(
        json_dir="data/shards",
        pt_out_dir="data/pt_shards",
        max_workers=None,
    )
