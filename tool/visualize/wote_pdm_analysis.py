from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectoryOrderKey:
    yaw_delta: float
    lateral_offset: float
    arc_length: float
    index: int


def _rank_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rank input must be 1D")
    if values.size == 0:
        return values.copy()

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    sorted_vals = values[order]

    i = 0
    while i < sorted_vals.size:
        j = i + 1
        while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def compute_spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < 2:
        return float("nan")

    rx = _rank_1d(x)
    ry = _rank_1d(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom == 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _trajectory_order_key(traj: np.ndarray, index: int) -> TrajectoryOrderKey:
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[-1] < 3:
        raise ValueError("trajectory must have shape [T, >=3]")

    start_yaw = float(traj[0, 2])
    end_yaw = float(traj[-1, 2])
    yaw_delta = end_yaw - start_yaw
    lateral_offset = float(traj[-1, 1] - traj[0, 1])
    xy = traj[:, :2]
    if xy.shape[0] > 1:
        diffs = np.diff(xy, axis=0)
        arc_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        arc_length = 0.0
    return TrajectoryOrderKey(yaw_delta=yaw_delta, lateral_offset=lateral_offset, arc_length=arc_length, index=index)


def rank_trajectories_left_to_right(trajectories: np.ndarray) -> np.ndarray:
    trajectories = np.asarray(trajectories, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[-1] < 3:
        raise ValueError("trajectories must have shape [K, T, >=3]")
    keys = [_trajectory_order_key(trajectories[i], i) for i in range(trajectories.shape[0])]
    order = sorted(
        range(len(keys)),
        key=lambda i: (
            -keys[i].yaw_delta,
            -keys[i].lateral_offset,
            keys[i].arc_length,
            keys[i].index,
        ),
    )
    return np.asarray(order, dtype=np.int64)


def save_analysis_artifacts(output_dir: Path, data: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "wote_pdm_scores.csv"
    npz_path = output_dir / "wote_pdm_scores.npz"

    import csv

    fieldnames = list(data["rows"][0].keys()) if data.get("rows") else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(data["rows"])

    np.savez_compressed(
        npz_path,
        trajectories=np.asarray(data["trajectories"]),
        wote_scores=np.asarray(data["wote_scores"]),
        pdm_scores=np.asarray(data["pdm_scores"]),
        rank_order=np.asarray(data["rank_order"]),
        spearman_rho=np.asarray([data["spearman_rho"]], dtype=np.float64),
    )
    return csv_path, npz_path


def plot_sorted_distribution(
    output_path: Path,
    trajectories: np.ndarray,
    wote_scores: np.ndarray,
    pdm_scores: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt

    trajectories = np.asarray(trajectories, dtype=np.float64)
    wote_scores = np.asarray(wote_scores, dtype=np.float64).reshape(-1)
    pdm_scores = np.asarray(pdm_scores, dtype=np.float64).reshape(-1)
    order = rank_trajectories_left_to_right(trajectories)
    x = np.arange(order.size, dtype=np.float64)
    wote_sorted = wote_scores[order]
    pdm_sorted = pdm_scores[order]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(x, wote_sorted, color="#7b61ff", linewidth=2.0, label="WoTE reward")
    ax.plot(x, pdm_sorted, color="#2d9cdb", linewidth=2.0, label="PDM score")
    ax.fill_between(x, wote_sorted, alpha=0.12, color="#7b61ff")
    ax.fill_between(x, pdm_sorted, alpha=0.12, color="#2d9cdb")
    ax.set_xlim(-0.5, max(len(x) - 0.5, 0.5))
    ax.set_xlabel("Trajectory shape: left-turn -> straight -> right-turn")
    ax.set_ylabel("Score")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
