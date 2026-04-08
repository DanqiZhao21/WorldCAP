#!/usr/bin/env python
"""Create a simple montage grid from a set of PNGs.

Example:
  python tool/evaluate/montage_grid.py \
    --in-dir ControllerInTheLoop/step0_validationOfSimulation/plots_post_dynamics_suite \
    --pattern 'PD*.png' \
    --rows 3 --cols 5 \
    --out ControllerInTheLoop/step0_validationOfSimulation/plots_post_dynamics_suite/montage_3x5.png

If --order is provided, it should be a comma-separated list of basenames (without extension)
that will be placed left-to-right, top-to-bottom.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, required=True)
    p.add_argument("--pattern", type=str, default="*.png")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--order",
        type=str,
        default="",
        help="Comma-separated basenames (no extension) to control placement.",
    )
    p.add_argument("--pad", type=int, default=10)
    p.add_argument("--bg", type=str, default="#ffffff")
    return p.parse_args()


def _load_images(in_dir: Path, pattern: str, order: List[str]) -> List[Image.Image]:
    if order:
        paths = [in_dir / f"{name}.png" for name in order]
    else:
        paths = sorted(in_dir.glob(pattern))

    images: List[Image.Image] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
        images.append(Image.open(p).convert("RGB"))
    return images


def main() -> None:
    args = parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = [s.strip() for s in args.order.split(",") if s.strip()]
    images = _load_images(in_dir, args.pattern, order)

    n = args.rows * args.cols
    if len(images) < n:
        raise ValueError(f"Need at least {n} images, found {len(images)}")
    images = images[:n]

    # Use the first image size as cell size
    cell_w, cell_h = images[0].size

    pad = int(args.pad)
    montage_w = args.cols * cell_w + (args.cols + 1) * pad
    montage_h = args.rows * cell_h + (args.rows + 1) * pad

    montage = Image.new("RGB", (montage_w, montage_h), color=args.bg)

    idx = 0
    for r in range(args.rows):
        for c in range(args.cols):
            x = pad + c * (cell_w + pad)
            y = pad + r * (cell_h + pad)
            montage.paste(images[idx].resize((cell_w, cell_h), Image.Resampling.LANCZOS), (x, y))
            idx += 1

    montage.save(out_path)


if __name__ == "__main__":
    main()
