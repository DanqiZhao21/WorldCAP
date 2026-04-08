#!/usr/bin/env python
"""Plot validation styles from a controller bundle.

Generates two overlay sets under an output root:
1) Red = original anchors (ref .npy), Blue = style executed trajs (bundle exec_trajs[idx])
2) Red = default executed trajs (bundle exec_trajs[0]), Blue = style executed trajs

Also creates two 8x4 montages by sampling ONE style per (kind, group_big, group_sub).

Example:
  python tool/evaluate/plot_val_bundle_overlays.py \
    --bundle ControllerExp/generated/controller_styles.npz \
    --ref ControllerExp/Anchors_Original_256_centered.npy \
    --out-root ControllerInTheLoop/step0_validationOfSimulation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bundle",
        type=str,
        default="ControllerExp/generated/controller_styles.npz",
        help="Controller bundle .npz (must contain exec_trajs/style_names/val_style_indices)",
    )
    p.add_argument(
        "--ref",
        type=str,
        default="ControllerExp/Anchors_Original_256_centered.npy",
        help="Original anchor .npy for red overlay (256,T,>=3)",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="ControllerInTheLoop/step0_validationOfSimulation",
        help="Output root directory",
    )
    p.add_argument("--alpha-blue", type=float, default=0.75)
    p.add_argument("--alpha-red", type=float, default=0.55)
    p.add_argument("--lw-blue", type=float, default=1.2)
    p.add_argument("--lw-red", type=float, default=1.0)
    p.add_argument("--max-trajs", type=int, default=256)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--montage-rows", type=int, default=8)
    p.add_argument("--montage-cols", type=int, default=4)
    return p.parse_args()


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _resample_traj_xyz(xy_yaw: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a single trajectory [T,3] -> [target_len,3]."""
    t_old = np.arange(xy_yaw.shape[0], dtype=float)
    t_new = np.linspace(0, xy_yaw.shape[0] - 1, target_len, dtype=float)

    x_old, y_old, yaw_old = xy_yaw[:, 0], xy_yaw[:, 1], xy_yaw[:, 2]
    x_new = np.interp(t_new, t_old, x_old)
    y_new = np.interp(t_new, t_old, y_old)

    yaw_unwrap = np.unwrap(yaw_old)
    yaw_new = np.interp(t_new, t_old, yaw_unwrap)
    yaw_new = _wrap_angle(yaw_new)

    return np.stack([x_new, y_new, yaw_new], axis=-1).astype(np.float32)


def _resample_anchors(anchors: np.ndarray, target_len: int) -> np.ndarray:
    if anchors.ndim != 3 or anchors.shape[0] != 256 or anchors.shape[2] < 3:
        raise ValueError(f"Expected anchors (256,T,>=3), got {anchors.shape}")

    if anchors.shape[1] == target_len:
        return anchors[:, :, :3].astype(np.float32)

    out = np.zeros((256, target_len, 3), dtype=np.float32)
    for i in range(256):
        out[i] = _resample_traj_xyz(anchors[i, :, :3], target_len)
    return out


def _sanitize(name: str) -> str:
    s = name.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:160] if len(s) > 160 else s


def _to_dict_item(x):
    if isinstance(x, dict):
        return x
    if hasattr(x, "item"):
        try:
            v = x.item()
            return v if isinstance(v, dict) else x
        except Exception:
            return x
    return x


def _add_trajs(ax, trajs_xy: np.ndarray, *, color: str, lw: float, alpha: float) -> None:
    """Draw many trajectories at once using LineCollection."""
    segs = trajs_xy.astype(np.float32)
    lc = LineCollection(segs, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection(lc)


def _plot_overlay(
    red: np.ndarray,
    blue: np.ndarray,
    *,
    title: str,
    out_png: Path,
    alpha_red: float,
    alpha_blue: float,
    lw_red: float,
    lw_blue: float,
    max_trajs: int,
    dpi: int,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n = min(int(max_trajs), int(red.shape[0]), int(blue.shape[0]))
    red2 = red[:n, :, :2]
    blue2 = blue[:n, :, :2]

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 7.6))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Red first (background), Blue on top
    _add_trajs(ax, red2, color="tab:red", lw=lw_red, alpha=alpha_red)
    _add_trajs(ax, blue2, color="tab:blue", lw=lw_blue, alpha=alpha_blue)

    ax.plot([], [], color="tab:red", label="Red")
    ax.plot([], [], color="tab:blue", label="Blue")
    ax.legend(loc="upper right", fontsize=8)

    ax.axis("equal")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def _make_montage(
    *,
    in_dir: Path,
    order: List[str],
    rows: int,
    cols: int,
    out_path: Path,
    pad: int = 10,
    bg: str = "#ffffff",
) -> None:
    """Create a fixed-order montage from in_dir/<name>.png files."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = rows * cols
    if len(order) < n:
        raise ValueError(f"Need at least {n} images for montage, got {len(order)}")

    paths = [in_dir / f"{name}.png" for name in order[:n]]
    images = [Image.open(p).convert("RGB") for p in paths]
    cell_w, cell_h = images[0].size

    montage_w = cols * cell_w + (cols + 1) * pad
    montage_h = rows * cell_h + (rows + 1) * pad
    montage = Image.new("RGB", (montage_w, montage_h), color=bg)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            x = pad + c * (cell_w + pad)
            y = pad + r * (cell_h + pad)
            montage.paste(images[idx], (x, y))
            idx += 1

    montage.save(out_path)


@dataclass(frozen=True)
class StyleInfo:
    idx: int
    name: str
    kind: str
    big: str
    sub: str


def main() -> None:
    args = parse_args()

    bundle_path = Path(args.bundle)
    ref_path = Path(args.ref)
    out_root = Path(args.out_root)

    data = np.load(str(bundle_path), allow_pickle=True)
    exec_trajs = data.get("exec_trajs", None)
    style_names = data.get("style_names", None)
    val_idx = data.get("val_style_indices", None)

    if exec_trajs is None or style_names is None or val_idx is None:
        raise RuntimeError("bundle must contain exec_trajs, style_names, val_style_indices")

    style_kind = data.get("style_kind", None)
    group_big = data.get("style_group_big", None)
    group_sub = data.get("style_group_sub", None)

    exec_trajs = np.asarray(exec_trajs)
    if exec_trajs.ndim != 4:
        raise ValueError(f"exec_trajs must be [S,256,8,3], got {exec_trajs.shape}")

    S, A, T, D = exec_trajs.shape
    if A != 256 or D < 3:
        raise ValueError(f"exec_trajs expected [S,256,T,>=3], got {exec_trajs.shape}")

    anchors = np.load(str(ref_path))
    anchors_ego = _resample_anchors(anchors, target_len=T)

    default_exec = exec_trajs[0]

    # Build style metadata list
    infos: List[StyleInfo] = []
    for i in list(val_idx):
        idx = int(i)
        name = str(style_names[idx])
        k = str(style_kind[idx]) if style_kind is not None else ""
        b = str(group_big[idx]) if group_big is not None else ""
        s = str(group_sub[idx]) if group_sub is not None else ""
        infos.append(StyleInfo(idx=idx, name=name, kind=k, big=b, sub=s))

    infos.sort(key=lambda x: x.idx)

    out_dir_anchor = out_root / "plots_val_anchor_vs_styleexec"
    out_dir_default = out_root / "plots_val_defaultexec_vs_styleexec"
    out_dir_anchor.mkdir(parents=True, exist_ok=True)
    out_dir_default.mkdir(parents=True, exist_ok=True)

    # Generate all 64*2 images
    for info in infos:
        blue = exec_trajs[info.idx]

        base = f"{info.idx:04d}_{_sanitize(info.name)}"
        if info.kind or info.big or info.sub:
            base = f"{info.idx:04d}_{_sanitize(info.kind)}_{_sanitize(info.big)}_{_sanitize(info.sub)}_{_sanitize(info.name)}"

        title_suffix = f"idx={info.idx} name={info.name}"
        if info.kind or info.big or info.sub:
            title_suffix += f" | {info.kind}/{info.big}/{info.sub}"

        # 1) red=original anchor
        _plot_overlay(
            anchors_ego,
            blue,
            title=f"VAL anchor vs style exec | {title_suffix}",
            out_png=out_dir_anchor / f"{base}.png",
            alpha_red=args.alpha_red,
            alpha_blue=args.alpha_blue,
            lw_red=args.lw_red,
            lw_blue=args.lw_blue,
            max_trajs=args.max_trajs,
            dpi=args.dpi,
        )

        # 2) red=default exec
        _plot_overlay(
            default_exec,
            blue,
            title=f"VAL default-exec vs style exec | {title_suffix}",
            out_png=out_dir_default / f"{base}.png",
            alpha_red=args.alpha_red,
            alpha_blue=args.alpha_blue,
            lw_red=args.lw_red,
            lw_blue=args.lw_blue,
            max_trajs=args.max_trajs,
            dpi=args.dpi,
        )

    # Pick one style per (kind,big,sub) group to make a 32-tile montage
    group_to_first: Dict[Tuple[str, str, str], StyleInfo] = {}
    for info in infos:
        key = (info.kind, info.big, info.sub)
        if key not in group_to_first:
            group_to_first[key] = info

    picked = list(group_to_first.values())
    picked.sort(key=lambda x: (x.kind, x.big, x.sub, x.idx))

    rows = int(args.montage_rows)
    cols = int(args.montage_cols)
    need = rows * cols
    if len(picked) < need:
        print(f"[WARN] only {len(picked)} groups, montage needs {need}; will truncate grid size")

    picked = picked[:need]

    def basename(info: StyleInfo) -> str:
        b = f"{info.idx:04d}_{_sanitize(info.name)}"
        if info.kind or info.big or info.sub:
            b = f"{info.idx:04d}_{_sanitize(info.kind)}_{_sanitize(info.big)}_{_sanitize(info.sub)}_{_sanitize(info.name)}"
        return b

    order = [basename(i) for i in picked]

    # Save montage order for reproducibility
    (out_dir_anchor / "montage_order_32.txt").write_text("\n".join(order) + "\n", encoding="utf-8")
    (out_dir_default / "montage_order_32.txt").write_text("\n".join(order) + "\n", encoding="utf-8")

    # Create two 32-tile montages (one per group)
    _make_montage(
        in_dir=out_dir_anchor,
        order=order,
        rows=rows,
        cols=cols,
        out_path=out_dir_anchor / f"montage_{rows}x{cols}.png",
    )
    _make_montage(
        in_dir=out_dir_default,
        order=order,
        rows=rows,
        cols=cols,
        out_path=out_dir_default / f"montage_{rows}x{cols}.png",
    )

    print(f"[OK] wrote images: {out_dir_anchor} and {out_dir_default}")
    print(f"[OK] wrote montage order (32): montage_order_32.txt")
    print(f"[OK] wrote montages: montage_{rows}x{cols}.png")


if __name__ == "__main__":
    main()
