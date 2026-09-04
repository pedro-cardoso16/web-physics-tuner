import json
import orjson
from pathlib import Path
import torch
from tqdm import tqdm
from wpt.nn.model import record_to_input, merge_hp

def pre_compile_dataset(shard_dir: str, output_file: str):
    shard_paths = sorted(Path(shard_dir).glob("sim_*.json"))
    
    all_inputs = []
    all_hps = []
    all_targets = []
    
    print(f"Pre-compiling {len(shard_paths)} JSON shards into a single binary file...")
    for path in tqdm(shard_paths, unit="shard"):
        # Using fast binary read + orjson for speed
        with open(path, "rb") as f:
            records = orjson.loads(f.read())
            
        hp = merge_hp(records)
        
        for r in records:
            all_inputs.append(record_to_input(r))
            all_hps.append(hp)
            all_targets.append(torch.tensor(r["target"], dtype=torch.float32))
            
    # Compile into raw binary tensors
    print("Stacking tensors...")
    compiled_data = {
        "data": torch.stack(all_inputs),
        "hp": torch.stack(all_hps),
        "label": torch.stack(all_targets)
    }
    
    print(f"Saving compiled dataset to {output_file}...")
    torch.save(compiled_data, output_file)
    print("Done!")

if __name__ == "__main__":
    pre_compile_dataset("data/shards", "data/compiled_dataset.pt")