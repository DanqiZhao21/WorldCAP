#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import yaml
from pyquaternion import Quaternion


DEFAULT_SCENE_FILTER_DIR = Path("navsim/planning/script/config/common/scene_filter")
DEFAULT_DATA_ROOT = Path("/mnt/data/navsim_workspace/dataset/navsim_logs")


def _yaw_from_frame(frame: Dict[str, Any]) -> float:
    return float(Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0])


def _local_delta(start_xy: np.ndarray, end_xy: np.ndarray, start_yaw: float) -> Tuple[float, float]:
    dx, dy = end_xy - start_xy
    forward = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
    lateral = -dx * math.sin(start_yaw) + dy * math.cos(start_yaw)
    return float(forward), float(lateral)


def _quantile(rows: List[Dict[str, Any]], key: str, percentile: float) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.percentile(values, percentile))


def _dedupe_preserve_order(tokens: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def load_scene_filter(scene_filter_path: Path) -> Dict[str, Any]:
    with scene_filter_path.open("r") as f:
        return yaml.safe_load(f)


def collect_scene_metrics(
    *,
    scene_filter_path: Path,
    navsim_log_root: Path,
    split: str,
) -> List[Dict[str, Any]]:
    scene_filter = load_scene_filter(scene_filter_path)
    log_names = scene_filter.get("log_names")
    if not log_names:
        raise ValueError(f"{scene_filter_path} must contain log_names for split-local extraction")

    num_history = int(scene_filter.get("num_history_frames", 4))
    num_future = int(scene_filter.get("num_future_frames", 10))
    frame_interval = int(scene_filter.get("frame_interval") or (num_history + num_future))
    num_frames = num_history + num_future
    current_idx = num_history - 1
    dt = 0.5

    rows: List[Dict[str, Any]] = []
    data_path = navsim_log_root / split

    for log_name in sorted(log_names):
        log_path = data_path / f"{log_name}.pkl"
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        with log_path.open("rb") as f:
            frames = pickle.load(f)

        for start in range(0, len(frames), frame_interval):
            frame_list = frames[start : start + num_frames]
            if len(frame_list) < num_frames:
                continue

            current = frame_list[current_idx]
            if scene_filter.get("has_route", True) and len(current["roadblock_ids"]) == 0:
                continue

            future_frames = frame_list[current_idx : current_idx + num_future + 1]
            yaws = np.unwrap([_yaw_from_frame(frame) for frame in future_frames])
            positions = np.asarray(
                [frame["ego2global_translation"][:2].astype(float) for frame in future_frames],
                dtype=np.float64,
            )
            step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            path_length = float(step_distances.sum())
            forward_disp, lateral_disp = _local_delta(positions[0], positions[-1], float(yaws[0]))

            yaw_rates = np.diff(yaws) / dt
            yaw_accels = np.diff(yaw_rates) / dt if len(yaw_rates) > 1 else np.asarray([0.0])
            speeds = np.asarray(
                [np.linalg.norm(frame["ego_dynamic_state"][:2]) for frame in future_frames],
                dtype=np.float64,
            )
            accels = np.asarray(
                [np.linalg.norm(frame["ego_dynamic_state"][2:]) for frame in future_frames],
                dtype=np.float64,
            )

            boxes = current["anns"]["gt_boxes"]
            num_agents = int(len(boxes))
            near_agents_30m = 0
            if num_agents:
                xy = np.asarray(boxes, dtype=np.float64)[:, :2]
                near_agents_30m = int((np.linalg.norm(xy, axis=1) <= 30.0).sum())

            abs_yaw = abs(float(yaws[-1] - yaws[0]))
            curvature_proxy = abs_yaw / max(path_length, 1e-3)
            speed = float(speeds[0])
            latacc_proxy = speed * speed * curvature_proxy

            rows.append(
                {
                    "token": current["token"],
                    "log_name": log_name,
                    "abs_yaw_deg": math.degrees(abs_yaw),
                    "max_yaw_rate_deg_s": math.degrees(float(np.max(np.abs(yaw_rates)))),
                    "max_yaw_acc_deg_s2": math.degrees(float(np.max(np.abs(yaw_accels)))),
                    "path_length_m": path_length,
                    "forward_disp_m": forward_disp,
                    "abs_lateral_disp_m": abs(lateral_disp),
                    "speed_mps": speed,
                    "speed_delta_mps": float(speeds.max() - speeds.min()),
                    "max_acc_mps2": float(accels.max()),
                    "num_agents": num_agents,
                    "near_agents_30m": near_agents_30m,
                    "has_traffic_light": bool(current["traffic_lights"]),
                    "driving_command": int(np.argmax(current["driving_command"])),
                    "curvature_proxy": curvature_proxy,
                    "latacc_proxy": latacc_proxy,
                }
            )

    return rows


def build_subsets(
    rows: List[Dict[str, Any]],
    *,
    hard_percentile: float = 90.0,
    curved_strict_percentile: float = 95.0,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    if not rows:
        raise ValueError("No scene rows to select from")

    p = hard_percentile
    strict_p = curved_strict_percentile
    thresholds = {
        f"abs_yaw_deg_p{int(p)}": _quantile(rows, "abs_yaw_deg", p),
        f"abs_yaw_deg_p{int(strict_p)}": _quantile(rows, "abs_yaw_deg", strict_p),
        f"abs_yaw_deg_p75": _quantile(rows, "abs_yaw_deg", 75.0),
        f"latacc_proxy_p{int(p)}": _quantile(rows, "latacc_proxy", p),
        f"speed_delta_mps_p{int(p)}": _quantile(rows, "speed_delta_mps", p),
        f"max_acc_mps2_p{int(p)}": _quantile(rows, "max_acc_mps2", p),
        f"max_yaw_rate_deg_s_p{int(p)}": _quantile(rows, "max_yaw_rate_deg_s", p),
        f"near_agents_30m_p{int(p)}": _quantile(rows, "near_agents_30m", p),
    }

    curved = [
        row["token"]
        for row in rows
        if float(row["abs_yaw_deg"]) >= thresholds[f"abs_yaw_deg_p{int(p)}"]
    ]
    curved_strict = [
        row["token"]
        for row in rows
        if float(row["abs_yaw_deg"]) >= thresholds[f"abs_yaw_deg_p{int(strict_p)}"]
    ]
    dynamic = [
        row["token"]
        for row in rows
        if (
            float(row["speed_delta_mps"]) >= thresholds[f"speed_delta_mps_p{int(p)}"]
            or float(row["max_acc_mps2"]) >= thresholds[f"max_acc_mps2_p{int(p)}"]
            or float(row["max_yaw_rate_deg_s"]) >= thresholds[f"max_yaw_rate_deg_s_p{int(p)}"]
        )
    ]
    fast_curve = [
        row["token"]
        for row in rows
        if (
            float(row["abs_yaw_deg"]) >= thresholds["abs_yaw_deg_p75"]
            and float(row["latacc_proxy"]) >= thresholds[f"latacc_proxy_p{int(p)}"]
        )
    ]
    interaction = [
        row["token"]
        for row in rows
        if (
            float(row["near_agents_30m"]) >= thresholds[f"near_agents_30m_p{int(p)}"]
            and (float(row["abs_yaw_deg"]) >= thresholds["abs_yaw_deg_p75"] or bool(row["has_traffic_light"]))
        )
    ]

    subsets = {
        f"navtest_hard_curved_p{int(p)}": _dedupe_preserve_order(curved),
        f"navtest_hard_curved_p{int(strict_p)}": _dedupe_preserve_order(curved_strict),
        f"navtest_hard_dynamic_p{int(p)}": _dedupe_preserve_order(dynamic),
        f"navtest_hard_fast_curve_p{int(p)}": _dedupe_preserve_order(fast_curve),
        f"navtest_hard_interaction_p{int(p)}": _dedupe_preserve_order(interaction),
    }
    subsets[f"navtest_hard_composite_p{int(p)}"] = _dedupe_preserve_order(
        token for name in subsets for token in subsets[name]
    )
    return subsets, thresholds


def write_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scene_filter_yaml(tokens: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "_target_: navsim.common.dataclasses.SceneFilter",
        "_convert_: 'all'",
        "",
        "num_history_frames: 4",
        "num_future_frames: 10",
        "frame_interval: 1",
        "has_route: true",
        "",
        "max_scenes: null",
        "log_names: null",
        "tokens:",
    ]
    lines.extend(f"  - '{token}'" for token in tokens)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract NAVSIM hard scene-filter YAMLs from split metadata.")
    parser.add_argument("--scene-filter", type=Path, default=DEFAULT_SCENE_FILTER_DIR / "navtest.yaml")
    parser.add_argument("--navsim-log-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCENE_FILTER_DIR)
    parser.add_argument("--report-dir", type=Path, default=Path("tool/smalltool/navsim_hard_scene_reports"))
    parser.add_argument("--hard-percentile", type=float, default=90.0)
    parser.add_argument("--curved-strict-percentile", type=float, default=95.0)
    args = parser.parse_args()

    rows = collect_scene_metrics(
        scene_filter_path=args.scene_filter,
        navsim_log_root=args.navsim_log_root,
        split=args.split,
    )
    subsets, thresholds = build_subsets(
        rows,
        hard_percentile=args.hard_percentile,
        curved_strict_percentile=args.curved_strict_percentile,
    )

    write_metrics_csv(rows, args.report_dir / f"{args.split}_hard_scene_metrics.csv")
    for name, tokens in subsets.items():
        write_scene_filter_yaml(tokens, args.output_dir / f"{name}.yaml")

    summary = {
        "scene_filter": str(args.scene_filter),
        "split": args.split,
        "num_scenes": len(rows),
        "thresholds": thresholds,
        "subsets": {name: len(tokens) for name, tokens in subsets.items()},
    }
    summary_path = args.report_dir / f"{args.split}_hard_scene_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
