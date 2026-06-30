#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("navsim/planning/script/config/common/scene_filter/navtest_hard_composite_p90.yaml"))
    parser.add_argument("--metric-cache-metadata", type=Path, default=Path("/mnt/data/navsim_workspace/exp/metric_cache/metadata/metric_cache_metadata_node_0.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("navsim/planning/script/config/common/scene_filter"))
    parser.add_argument("--prefix", default="navtest_hard_composite_p90_shard")
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    source = load_yaml(args.source)
    tokens = list(dict.fromkeys(source.get("tokens") or []))
    metadata = pd.read_csv(args.metric_cache_metadata)
    if "token" in metadata:
        cache_tokens = set(metadata["token"].astype(str))
    else:
        cache_tokens = {Path(file_name).parent.name for file_name in metadata["file_name"].astype(str)}
    tokens = [token for token in tokens if token in cache_tokens]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for shard_idx in range(args.num_shards):
        shard = dict(source)
        shard["tokens"] = tokens[shard_idx:: args.num_shards]
        out_path = args.out_dir / f"{args.prefix}{shard_idx}.yaml"
        with out_path.open("w") as f:
            yaml.safe_dump(shard, f, sort_keys=False)
        print(f"{out_path}: {len(shard['tokens'])} tokens")


if __name__ == "__main__":
    main()
