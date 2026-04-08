#!/usr/bin/env python
"""Stitch S01-S10 PNGs into a single 2x5 montage.

Default input dir matches step0_validationOfSimulation outputs.
Default order matches the S01-S10 matrix used elsewhere in this repo.

Example:
  python tool/evaluate/montage_s01_s10_2x5.py \
    --in-dir 'ControllerInTheLoop/step0_validationOfSimulation/plots_s01_s10_origianchor&control' \
    --out   'ControllerInTheLoop/step0_validationOfSimulation/plots_s01_s10_origianchor&control/montage_2x5.png'
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image

DEFAULT_ORDER: List[str] = [

    "S01_default_none.png",
    "S02_default_post1515.png",
    "S03_aggressive_none.png",
    "S04_safe_none.png",
    "S05_sluggish_none.png",
    "S06_high_jitter_none.png",
    "S07_unstable_none.png",
    "S08_yaw_scale_12.png",
    "S09_speed_scale_08.png",
    "S10_noise_02.png",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in-dir",
        type=str,
        default="ControllerInTheLoop/step0_validationOfSimulation/plots_s01_s10_origianchor&control",
        help="Directory containing the 10 scenario PNGs.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="ControllerInTheLoop/step0_validationOfSimulation/plots_s01_s10_origianchor&control/montage_2x5.png",
        help="Output PNG path.",
    )
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument(
        "--bg",
        type=str,
        default="#FFFFFF",
        help="Background color for padding, e.g. '#FFFFFF'.",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Pixels between tiles and around edges.",
    )
    p.add_argument(
        "--order",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit filename order (10 pngs). Defaults to S07..S06 order.",
    )
    return p.parse_args()


def _load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order = args.order if args.order is not None else DEFAULT_ORDER
    expected = args.rows * args.cols
    if len(order) != expected:
        raise SystemExit(f"Expected {expected} images for {args.rows}x{args.cols}, got {len(order)}")

    paths = [in_dir / name for name in order]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("Missing files:\n" + "\n".join(missing))

    images = [_load_rgb(p) for p in paths]

    # Normalize tile sizes: pad each image to max width/height.
    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)

    bg = args.bg
    tiles: List[Image.Image] = []
    for im in images:
        if im.width == max_w and im.height == max_h:
            tiles.append(im)
            continue
        canvas = Image.new("RGB", (max_w, max_h), bg)
        x = (max_w - im.width) // 2
        y = (max_h - im.height) // 2
        canvas.paste(im, (x, y))
        tiles.append(canvas)

    m = int(args.margin)
    out_w = args.cols * max_w + (args.cols + 1) * m
    out_h = args.rows * max_h + (args.rows + 1) * m

    montage = Image.new("RGB", (out_w, out_h), bg)

    for idx, tile in enumerate(tiles):
        r = idx // args.cols
        c = idx % args.cols
        x0 = m + c * (max_w + m)
        y0 = m + r * (max_h + m)
        montage.paste(tile, (x0, y0))

    montage.save(out_path)
    print(f"[OK] wrote montage: {out_path}")


if __name__ == "__main__":
    main()
