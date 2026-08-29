from pathlib import Path
import json

shard_dir = Path("data/shards")
print("cwd:", Path.cwd())
print("resolved shard_dir:", shard_dir.resolve())
print("shard_dir exists:", shard_dir.exists())

with open(shard_dir / "manifest.json") as f:
    manifest = json.load(f)
print("manifest length:", len(manifest))