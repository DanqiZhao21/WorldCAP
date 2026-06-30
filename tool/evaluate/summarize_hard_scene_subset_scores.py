#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


DEFAULT_FILTERS = [
    "navtest_hard_curved_p90",
    "navtest_hard_curved_p95",
    "navtest_hard_dynamic_p90",
    "navtest_hard_fast_curve_p90",
    "navtest_hard_interaction_p90",
    "navtest_hard_composite_p90",
]


def load_tokens(scene_filter_dir: Path, name: str) -> set[str]:
    with (scene_filter_dir / f"{name}.yaml").open("r") as f:
        data = yaml.safe_load(f)
    return set(data.get("tokens") or [])


def find_score_csv(run_dir: Path) -> Path:
    final_csvs = sorted(p for p in run_dir.glob("*.csv") if not p.name.startswith("partial_"))
    if final_csvs:
        return final_csvs[-1]
    partial_csvs = sorted(run_dir.glob("partial_*.csv"))
    if partial_csvs:
        return partial_csvs[-1]
    raise FileNotFoundError(f"No score CSV found in {run_dir}")


def find_score_csvs(run_dir: Path) -> list[Path]:
    shard_dirs = sorted(run_dir.glob("navtest_hard_composite_p90_shard*"))
    if shard_dirs:
        return [find_score_csv(shard_dir) for shard_dir in shard_dirs]
    return [find_score_csv(run_dir / "navtest_hard_composite_p90")]


def load_scores(run_dir: Path) -> pd.DataFrame:
    csv_paths = find_score_csvs(run_dir)
    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame = frame[frame["token"] != "average"].copy()
        frame["source_csv"] = str(csv_path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["token"] != "average"].copy()
    df = df.drop_duplicates(subset=["token"], keep="last")
    return df


def summarize(
    *,
    scene_filter_dir: Path,
    exp_root: Path,
    out_base: str,
    model_runs: Dict[str, str],
    filters: List[str],
) -> pd.DataFrame:
    filter_tokens = {name: load_tokens(scene_filter_dir, name) for name in filters}
    rows = []
    score_dfs = {
        model_name: load_scores(exp_root / out_base / run_tag)
        for model_name, run_tag in model_runs.items()
    }

    for filter_name in filters:
        tokens = filter_tokens[filter_name]
        row = {
            "subset": filter_name,
            "yaml_scenes": len(tokens),
        }
        for model_name, df in score_dfs.items():
            sub = df[df["token"].isin(tokens)]
            valid = sub[sub["valid"].astype(bool)] if "valid" in sub else sub
            prefix = model_name
            row[f"{prefix}_evaluated"] = int(len(sub))
            row[f"{prefix}_valid"] = int(len(valid))
            row[f"{prefix}_score"] = float(valid["score"].mean()) if len(valid) else float("nan")
            for metric in [
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "driving_direction_compliance",
                "ego_progress",
                "time_to_collision_within_bound",
                "comfort",
            ]:
                if metric in valid:
                    row[f"{prefix}_{metric}"] = float(valid[metric].mean()) if len(valid) else float("nan")
        model_names = list(score_dfs)
        if len(model_names) == 2:
            lhs, rhs = model_names
            row[f"delta_{lhs}_minus_{rhs}"] = row[f"{lhs}_score"] - row[f"{rhs}_score"]
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-filter-dir", type=Path, default=Path("navsim/planning/script/config/common/scene_filter"))
    parser.add_argument("--exp-root", type=Path, default=Path("/mnt/data/navsim_workspace/exp"))
    parser.add_argument("--out-base", default="eval/hard_scene_sharded_20260513")
    parser.add_argument("--output", type=Path, default=Path("tool/evaluate/hard_scene_subset_scores_20260513.csv"))
    parser.add_argument(
        "--model-runs-json",
        default=json.dumps(
            {
                "epoch19_step21280": "epoch19_step21280",
                "epoch29_step19950": "epoch29_step19950",
            }
        ),
    )
    args = parser.parse_args()
    model_runs = json.loads(args.model_runs_json)
    table = summarize(
        scene_filter_dir=args.scene_filter_dir,
        exp_root=args.exp_root,
        out_base=args.out_base,
        model_runs=model_runs,
        filters=DEFAULT_FILTERS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
