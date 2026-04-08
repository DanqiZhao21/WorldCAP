#!/usr/bin/env python
"""Inspect ControllerExp/generated/controller_styles.npz.

Prints style index -> name, and optionally saves a CSV.

Example:
  python tool/evaluate/inspect_controller_bundle.py \
    --bundle ControllerExp/generated/controller_styles.npz --csv /tmp/styles.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bundle",
        type=str,
        default="ControllerExp/generated/controller_styles.npz",
        help="Path to controller bundle .npz",
    )
    p.add_argument("--csv", type=str, default="", help="Optional output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = Path(args.bundle)
    data = np.load(str(bundle), allow_pickle=True)

    style_names = data.get("style_names", None)
    exec_trajs = data.get("exec_trajs", None)
    ref_traj = data.get("ref_traj", None)

    if style_names is None or exec_trajs is None or ref_traj is None:
        raise RuntimeError("bundle must contain style_names, exec_trajs, ref_traj")

    print(f"bundle={bundle}")
    print(f"styles={len(style_names)} exec_trajs={exec_trajs.shape} ref_traj={ref_traj.shape}")

    train_idx = data.get("train_style_indices", None)
    val_idx = data.get("val_style_indices", None)
    if train_idx is not None and val_idx is not None:
        try:
            print(f"split: train_styles={len(train_idx)} val_styles={len(val_idx)}")
            print(f"  train_idx[:10]={list(train_idx[:10])}")
            print(f"  val_idx[:10]={list(val_idx[:10])}")
        except Exception:
            pass

    rows = []
    for i, name in enumerate(list(style_names)):
        s = str(name)
        print(f"{i:04d}  {s}")
        rows.append((i, s))

    if args.csv:
        import csv

        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "style_name"])
            w.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
