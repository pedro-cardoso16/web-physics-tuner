import orjson
from pathlib import Path
import torch
from tqdm import tqdm
from wpt.nn.model import record_to_input, merge_hp

def compile_shards_incrementally(json_dir: str, pt_dir: str):
    json_paths = sorted(Path(json_dir).glob("sim_*.json"))
    pt_path_dir = Path(pt_dir)
    pt_path_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Incrementally compiling {len(json_paths)} JSON shards to binary .pt files...")
    
    for path in tqdm(json_paths, desc="Compiling json files to shards",unit="shard"):
        # 1. Load and parse only ONE JSON file at a time
        with open(path, "rb") as f:
            records = orjson.loads(f.read())
            
        hp = merge_hp(records)
        
        inputs = []
        targets = []
        hps = []
        
        for r in records:
            inputs.append(record_to_input(r))
            hps.append(hp)
            targets.append(torch.tensor(r["target"], dtype=torch.float32))
            
        # 2. Compile into a dictionary of stacked tensors for this simulation
        compiled_shard = {
            "data": torch.stack(inputs),
            "hp": torch.stack(hps),
            "label": torch.stack(targets)
        }
        
        # 3. Save as a binary .pt file and free the memory immediately
        output_file = pt_path_dir / f"{path.stem}.pt"
        torch.save(compiled_shard, output_file)
        
    print(f"Successfully compiled all shards into {pt_dir}!")

if __name__ == "__main__":
    compile_shards_incrementally("data/shards", "data/pt_shards")