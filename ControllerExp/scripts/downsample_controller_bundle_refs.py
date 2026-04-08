#!/usr/bin/env python3
"""Downsample reference trajectories in a controller bundle .npz.

Use case:
  Starting from the original 256-ref bundle (ControllerExp/generated/controller_styles.npz),
  create 128/64-ref variants by uniformly subsampling the reference index dimension.

The bundle is expected to contain at least:
  - ref_traj   (N_ref, T, 3)
  - exec_trajs (N_style, N_ref, T, 3)

All other keys (style metadata, splits, etc.) are copied as-is.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


def _uniform_indices(n_ref: int, target: int) -> np.ndarray:
    if target <= 0:
        raise ValueError(f"target must be >0, got {target}")
    if target > n_ref:
        raise ValueError(f"target ({target}) cannot exceed n_ref ({n_ref})")

    if n_ref % target != 0:
        # Fallback: linspace + unique, then pad if needed.
        idx = np.unique(np.round(np.linspace(0, n_ref - 1, target)).astype(int))
        if idx.size < target:
            missing = [i for i in range(n_ref) if i not in set(idx.tolist())]
            idx = np.sort(np.concatenate([idx, np.asarray(missing[: target - idx.size], dtype=int)]))
        return idx[:target]

    step = n_ref // target
    return np.arange(0, n_ref, step, dtype=int)


def _backup_if_exists(dst: Path) -> None:
    if not dst.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = dst.with_suffix(dst.suffix + f".bak_{ts}")
    shutil.copy2(dst, bak)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        required=True,
        help="Source controller bundle .npz (e.g. ControllerExp/generated/controller_styles.npz)",
    )
    p.add_argument(
        "--dst",
        required=True,
        help="Destination .npz path to write",
    )
    p.add_argument("--target-nref", type=int, required=True, help="Target number of reference trajectories")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination if exists (a .bak_YYYYmmdd_HHMMSS copy is kept)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    target = int(args.target_nref)

    if not src.exists():
        raise FileNotFoundError(src)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"Destination exists: {dst} (use --overwrite)")

    data = np.load(src, allow_pickle=True)
    if "ref_traj" not in data.files or "exec_trajs" not in data.files:
        raise KeyError(f"Missing ref_traj/exec_trajs in {src}; keys={sorted(data.files)}")

    ref_traj = np.asarray(data["ref_traj"])
    exec_trajs = np.asarray(data["exec_trajs"])

    if ref_traj.ndim != 3 or exec_trajs.ndim != 4:
        raise ValueError(f"Unexpected shapes: ref_traj={ref_traj.shape} exec_trajs={exec_trajs.shape}")

    n_ref = int(ref_traj.shape[0])
    if exec_trajs.shape[1] != n_ref:
        raise ValueError(f"Shape mismatch: exec_trajs.shape={exec_trajs.shape} ref_traj.shape={ref_traj.shape}")

    idx = _uniform_indices(n_ref, target)
    if idx.size != target:
        raise RuntimeError(f"Indexing failed: expected {target} indices, got {idx.size}")

    out: dict[str, object] = {}
    for k in data.files:
        if k == "ref_traj":
            out[k] = ref_traj[idx]
        elif k == "exec_trajs":
            out[k] = exec_trajs[:, idx]
        else:
            out[k] = data[k]

    if dst.exists() and args.overwrite:
        _backup_if_exists(dst)

    np.savez_compressed(dst, **out)

    # Minimal stdout for scripting.
    print(f"[OK] {src} -> {dst} (n_ref {n_ref} -> {target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
