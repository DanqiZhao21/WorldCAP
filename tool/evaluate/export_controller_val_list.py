#!/usr/bin/env python
"""Export validation style list from a controller bundle.

Writes a CSV and TXT list for manual selection.

Example:
  python tool/evaluate/export_controller_val_list.py \
    --bundle ControllerExp/generated/controller_styles.npz \
    --out-dir ControllerExp/generated
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bundle",
        type=str,
        default="ControllerExp/generated/controller_styles.npz",
        help="Path to controller bundle .npz",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="ControllerExp/generated",
        help="Output directory",
    )
    return p.parse_args()


def _as_py(x):
    if hasattr(x, "item") and not isinstance(x, (dict, list, str)):
        try:
            return x.item()
        except Exception:
            return x
    return x


def main() -> None:
    args = parse_args()
    bundle_path = Path(args.bundle)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(bundle_path), allow_pickle=True)

    style_names = data.get("style_names", None)
    kind = data.get("style_kind", None)
    big = data.get("style_group_big", None)
    sub = data.get("style_group_sub", None)
    val_idx = data.get("val_style_indices", None)

    if style_names is None or val_idx is None:
        raise RuntimeError("bundle must contain style_names and val_style_indices")

    val_rows = []
    for idx in list(val_idx):
        i = int(_as_py(idx))
        name = str(_as_py(style_names[i]))
        k = str(_as_py(kind[i])) if kind is not None else ""
        b = str(_as_py(big[i])) if big is not None else ""
        s = str(_as_py(sub[i])) if sub is not None else ""
        val_rows.append((i, name, k, b, s))

    val_rows.sort(key=lambda r: r[0])

    csv_path = out_dir / "val_styles.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "style_name", "kind", "group_big", "group_sub"])
        w.writerows(val_rows)

    txt_path = out_dir / "val_styles.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, name, k, b, s in val_rows:
            f.write(f"{i:04d}\t{name}\t{k}\t{b}\t{s}\n")

    print(f"wrote {csv_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
