#!/usr/bin/env python3
"""Plot original anchors vs executed trajectories for selected controller styles.

Inputs:
  - anchors: ControllerExp/Anchors_Original_256.npy (N_ref, T, 3)
  - bundle : ControllerExp/generated/controller_styles.npz
      exec_trajs: (N_style, N_ref, T, 3) in *ego* frame (t0 = 0)

Since the original anchors are NOT centered (their t0 varies), we transform exec_trajs
from ego -> the anchor's local global frame using each anchor's initial pose:

  xg = x0 + xe*cos(yaw0) - ye*sin(yaw0)
  yg = y0 + xe*sin(yaw0) + ye*cos(yaw0)
  yawg = yaw0 + yawe

This makes the overlay comparable to Anchors_Original_256.npy as-is.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sanitize(s: str, *, max_len: int = 140) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return (s or "unnamed")[:max_len]


def _ego_to_global(exec_ego: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """exec_ego: (T,3) in ego coords; init_pose: (3,) [x0,y0,yaw0] in global."""
    x0, y0, yaw0 = float(init_pose[0]), float(init_pose[1]), float(init_pose[2])
    c = math.cos(yaw0)
    s = math.sin(yaw0)
    out = np.empty_like(exec_ego, dtype=np.float32)
    xe = exec_ego[:, 0]
    ye = exec_ego[:, 1]
    ya = exec_ego[:, 2]
    out[:, 0] = x0 + xe * c - ye * s
    out[:, 1] = y0 + xe * s + ye * c
    out[:, 2] = yaw0 + ya
    return out


def _plot_style(
    anchors_global: np.ndarray,
    exec_global: np.ndarray,
    *,
    title: str,
    out_svg: Path,
    alpha_ref: float = 0.20,
    alpha_exec: float = 0.28,
    lw_ref: float = 0.9,
    lw_exec: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160)

    for i in range(anchors_global.shape[0]):
        ax.plot(
            anchors_global[i, :, 0],
            anchors_global[i, :, 1],
            color=(1.0, 0.2, 0.2),
            alpha=alpha_ref,
            linewidth=lw_ref,
        )
        ax.plot(
            exec_global[i, :, 0],
            exec_global[i, :, 1],
            color=(0.15, 0.35, 1.0),
            alpha=alpha_exec,
            linewidth=lw_exec,
        )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--anchors",
        default="ControllerExp/Anchors_Original_256.npy",
        help="Path to Anchors_Original_256.npy (global/uncentered anchors)",
    )
    p.add_argument(
        "--bundle",
        default="ControllerExp/generated/controller_styles.npz",
        help="Controller bundle .npz with exec_trajs in ego frame",
    )
    p.add_argument(
        "--style-idx",
        nargs="+",
        required=True,
        type=int,
        help="Style indices to plot (e.g. 2 25 41)",
    )
    p.add_argument(
        "--out-dir",
        default="ControllerExp/generated/plots_original_anchor_vs_exec_styles",
        help="Output directory under which SVGs are written",
    )
    p.add_argument("--max-ref", type=int, default=0, help="Optionally plot only first K refs (0 = all)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    anchors_path = Path(args.anchors).expanduser().resolve()
    bundle_path = Path(args.bundle).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    anchors = np.asarray(np.load(anchors_path), dtype=np.float32)  # (N_ref,T,3)
    d = np.load(bundle_path, allow_pickle=True)
    exec_trajs = np.asarray(d["exec_trajs"], dtype=np.float32)  # (S,N_ref,T,3) ego
    style_names = d.get("style_names", None)

    if anchors.ndim != 3 or anchors.shape[-1] != 3:
        raise ValueError(f"Unexpected anchors shape: {anchors.shape}")
    if exec_trajs.ndim != 4 or exec_trajs.shape[-1] != 3:
        raise ValueError(f"Unexpected exec_trajs shape: {exec_trajs.shape}")
    if exec_trajs.shape[1] != anchors.shape[0] or exec_trajs.shape[2] != anchors.shape[1]:
        raise ValueError(
            f"Shape mismatch: anchors={anchors.shape} exec_trajs={exec_trajs.shape} (need same N_ref,T)"
        )

    n_ref = int(anchors.shape[0])
    if args.max_ref and int(args.max_ref) > 0:
        n_ref = min(n_ref, int(args.max_ref))
        anchors = anchors[:n_ref]

    # Initial pose per reference trajectory (global)
    init_pose = anchors[:, 0, :]  # (N_ref,3)

    for idx in args.style_idx:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= int(exec_trajs.shape[0]):
            raise ValueError(f"style_idx out of range: {idx_i} (num_styles={exec_trajs.shape[0]})")

        name = str(style_names[idx_i]) if style_names is not None else f"style_{idx_i}"
        name_s = _sanitize(name)

        # Transform exec from ego->global for each ref.
        exec_ego = exec_trajs[idx_i, : anchors.shape[0]]  # (N_ref,T,3)
        exec_global = np.empty_like(exec_ego, dtype=np.float32)
        for r in range(exec_ego.shape[0]):
            exec_global[r] = _ego_to_global(exec_ego[r], init_pose[r])

        out_svg = out_dir / f"{idx_i:04d}__{name_s}.svg"
        title = f"style_idx={idx_i:04d}  {name}"
        _plot_style(anchors, exec_global, title=title, out_svg=out_svg)
        print(f"[OK] wrote {out_svg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
