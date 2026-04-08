#!/usr/bin/env python3
"""Visualize a single token for two WoTE checkpoints (baseline vs film03).

This script:
1) Loads the scene + metric cache for a given token
2) Runs WoTE to generate multi-modal candidate trajectories
3) Runs PDM scoring in *evaluate_all_trajectories* mode (for visualization)
4) Saves two BEV images (one per checkpoint)

It reuses the repo's existing implementations:
- navsim.evaluate.pdm_score.pdm_score_multiTraj
- navsim.visualization.plots.plot_bev_with_agent_and_simulation

Typical usage:
  python tool/smalltool/visualize_token_compare.py \
    --token <TOKEN> \
    --split val \
    --baseline-ckpt /path/to/baseline.ckpt \
    --film-ckpt /path/to/film03.ckpt \
    --out-dir /path/to/out
    
    
#FIXME:单独一张token
python tool/smalltool/visualize_token_compare.py \
    --token c3ace87d2f985eaa \
    --split val \
    --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
    --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
    --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault \
    --noScore \
    --format svg等
#无分数的
python tool/smalltool/visualize_token_compare_thick.py     --token 055c41d3c8e75bdc     --split val     --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt     --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt     --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault --noScore
#FIXME:单独一张token的：
  python /home/zhaodanqi/clone/WoTE/tool/smalltool/visualize_token_compare_thick.py \
        --token d37e71f395695c6f \
        --split val \
        --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
        --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
        --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/special
        
#FIXME:collision
python tool/smalltool/visualize_token_compare.py \
    --tokens 25226305bdff5efd 7b8d8487706f501e bc91cb648d525c6c 055c41d3c8e75bdc 9b16e4fea2c25446 ae06592110305073 98c7a48dd75052ab 4c22cdcc527e5a36 432598c0bda65445 198bc5f3280e52cd 3d4616d64a4c5f53 431869f33ace51a0 5a80299213875068 7ae00644dbef537f 885f450f0b875861 8fb11d5808355072 b77b4b6eb149553f b87ed2985e545397 c5ff90667143574a f1dabe118a6955d6 b66db24cee7957a2 9164913b818a58b1 667afe1f010351c5 79a151c333745253 d1bae9e7d9785598 \
    --split val \
    --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
    --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
    --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault



#FIXME:drivable
python tool/smalltool/visualize_token_compare.py \
    --tokens 12f5053055935463 2be2e48e80985bee 75f5cc1f425c501d 8164612a623156ae 891fe2fc30c95109 9c9b62965da75764 a9c368a110585b1a b50b8f11d75a5cb0 f4814e7eb01252b6 fb42b4cbf6b95b6c 32a19a2ab50f563d 0defa8939c9851ea ea21c4b17b865a4d 2751f9eb641455c9 0558c7a64ef157b1 4909b88b347c5764 13fa82d6564e5bac 6950076b024c51db aaa11cdbc8d35178 bb9c441a4c2b5791 1e9a42ef8f4057a6 d37e71f395695c6f 778219f3cac65d35 f99992756cbb5adb 8364af67153a5193 62462203db6b5ba5 1039e136e6605cfb e03da8beb33a5e06 b113e988ede45a4f 87b32a1aeeb85613 87d3c1135ac85583 c87fe1d7a3bd57cb 5c5d006eb7b854c3 bc2314763cfc545d ea34282dc63d5a9a cb5022a3bef557e3 6c28c001109f5718 91fc00df56ae5aca c2ed826b31065c66 602e1bc4f8575d4e ce3d0bc0b2d55876 1da6196444e35b0f fc9b5914f47e58fe c93a302d2fb2508f 406f8b299de35ce2 6d0a7f0bb4e7584b 0c91824ce1e65b6e be85b447a33b59f8 d5fc95fa66025d7a 012432bd62b85f80 e13b89a8813159df 8183cdc6ff5a5726 f9c8ec6aefc05be4 14b6e7ce317d531f 759af2e479de5bbb \
    --split val \
    --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
    --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
    --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_drivable





/home/zhaodanqi/clone/WoTE/pic/scenario/scenarios_drivable
#FIXME:



Paths can be provided explicitly, otherwise env vars are used:
- OPENSCENE_DATA_ROOT (for navsim logs + sensor blobs)
- NAVSIM_EXP_ROOT     (for metric cache)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import lzma
import os
import pickle
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# NOTE: Do not override CUDA_VISIBLE_DEVICES here.
# Users typically set it from the shell (e.g. export CUDA_VISIBLE_DEVICES=2).
# Hard-coding a device can accidentally route jobs to a busy GPU and cause OOM.


def _resolve_token_out_dir(out_dir_root: Path, token: str) -> Path:
    """Resolve per-token output directory.

    If there is an existing directory under out_dir_root whose name starts with
    the token (e.g. '<token>--good'), reuse it. This supports user-added notes
    in folder names.
    """

    try:
        if not out_dir_root.exists():
            return (out_dir_root / token).resolve()
        matches = [
            p
            for p in out_dir_root.iterdir()
            if p.is_dir() and p.name.startswith(token)
        ]
        if not matches:
            return (out_dir_root / token).resolve()

        exact = next((p for p in matches if p.name == token), None)
        if exact is not None:
            return exact.resolve()

        # If multiple matches exist, pick the most recently modified.
        best = max(matches, key=lambda p: (p.stat().st_mtime, -len(p.name)))
        if len(matches) > 1:
            names = ", ".join(sorted([p.name for p in matches]))
            print(
                f"[WARN] multiple out dirs start with token '{token}': {names}; using '{best.name}'",
                file=sys.stderr,
            )
        return best.resolve()
    except Exception:
        return (out_dir_root / token).resolve()


def _plot_bev_focus_compare(
    *,
    scene: Any,
    human_traj: Any,
    baseline_traj: Any,
    film_traj: Any,
    baseline_score: Optional[float],
    film_score: Optional[float],
    no_score: bool,
    out_format: str,
    out_dir: Path,
    out_name: str,
) -> str:
    """Focused visualization: background unchanged, only 3 trajectories.

    Draws:
      - GT (human)
      - baseline ckpt model-selected trajectory
      - film ckpt model-selected trajectory

    Also shows a minimal score box with only PDM total scores.

    Returns the saved image path.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
    from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
    from navsim.visualization.plots import configure_ax, configure_bev_ax  # also keeps BEV defaults consistent

    frame_idx = scene.scene_metadata.num_history_frames - 1

    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

    # Style: no markers, thin lines, explicit colors.
    # User-requested colors:
    # - GT: red
    # - WoTE (original ckpt): royal blue
    # - WorldCAP (finetuned ckpt): purple
    COLOR_GT = "#80c5f0"
    COLOR_WOTE = "#27cfbe"       # royal blue
    COLOR_WORLDCAP = "#fa7f6f"   # purple
    GT_TRAJ_ALPHA = 0.85
    WOTE_TRAJ_ALPHA = 0.95
    WORLDCAP_TRAJ_ALPHA = 0.98

    gt_cfg = dict(TRAJECTORY_CONFIG["human"])
    gt_cfg["line_color"] = COLOR_GT
    gt_cfg["line_color_alpha"] = GT_TRAJ_ALPHA
    # gt_cfg["line_width"] = 1.6
    gt_cfg["line_width"] = 0.6
    gt_cfg["marker"] = None
    gt_cfg["marker_size"] = 0

    wote_cfg = dict(TRAJECTORY_CONFIG["agent"])
    wote_cfg["line_color"] = COLOR_WOTE
    wote_cfg["line_color_alpha"] = WOTE_TRAJ_ALPHA
    # wote_cfg["line_width"] = 1.4
    wote_cfg["line_width"] = 0.4
    wote_cfg["line_style"] = "-"
    wote_cfg["marker"] = None
    wote_cfg["marker_size"] = 0

    worldcap_cfg = dict(TRAJECTORY_CONFIG["agent"])
    worldcap_cfg["line_color"] = COLOR_WORLDCAP
    worldcap_cfg["line_color_alpha"] = WORLDCAP_TRAJ_ALPHA
    # worldcap_cfg["line_width"] = 1.4
    worldcap_cfg["line_width"] = 0.4
    worldcap_cfg["line_style"] = "-"
    worldcap_cfg["marker"] = None
    worldcap_cfg["marker_size"] = 0

    # Draw only 3 trajectories.
    add_trajectory_to_bev_ax(ax, human_traj, gt_cfg)
    add_trajectory_to_bev_ax(ax, baseline_traj, wote_cfg)
    add_trajectory_to_bev_ax(ax, film_traj, worldcap_cfg)

    # Minimal score/legend box
    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v) * 100.0:.1f}"
        except Exception:
            return "n/a"

    # Minimal score/legend box: rounded rectangle with comfortable margins.
    # Use an in-axes margin so the box doesn't hug the border even with bbox_inches='tight'.
    edge_margin = 0.045
    box_x, box_y = edge_margin, 1.0 - edge_margin  # top-left anchor in axes coords
    fs_title = 10
    fs_line = 9
    # Tighter line spacing inside the box.
    line_h = 0.052

    # Create the patch first (dummy size) and update bounds after we measure text extents.
    box_patch = FancyBboxPatch(
        (0.0, 0.0),
        0.1,
        0.1,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        transform=ax.transAxes,
        facecolor="black",
        edgecolor="#333333",
        linewidth=1.0,
        alpha=0.35,
        zorder=299,
    )
    ax.add_patch(box_patch)

    bar_char = "━"  # small colored bar
    t_title = None
    if not no_score:
        t_title = ax.text(
            box_x,
            box_y,
            "PDM Score",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fs_title,
            color="white",
            zorder=300,
        )
    
    row0 = 0 if no_score else 1

    # GT colored bar
    t_gt_bar = ax.text(
        box_x,
        box_y - (row0 + 0) * line_h,
        bar_char,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fs_line,
        color=COLOR_GT,
        zorder=300,
    )

    text_x = box_x + 0.035
    t_gt = ax.text(
        text_x,
        box_y - (row0 + 0) * line_h,
        "GT",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fs_line,
        color="white",
        zorder=300,
    )
    #WOTE
    t_wote_bar = ax.text(
        box_x,
        box_y - (row0 + 1) * line_h,
        bar_char,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fs_line,
        color=COLOR_WOTE,
        zorder=300,
    )

    t_wote = ax.text(
        text_x,
        box_y - (row0 + 1) * line_h,
        ("WoTE" if no_score else f"WoTE: {_fmt(baseline_score)}"),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fs_line,
        color="white",
        zorder=300,
    )
    # WorldCAP
    t_wc_bar = ax.text(
        box_x,
        box_y - (row0 + 2) * line_h,
        bar_char,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fs_line,
        color=COLOR_WORLDCAP,
        zorder=300,
    )

    t_wc = ax.text(
        text_x,
        box_y - (row0 + 2) * line_h,
        ("WorldCAP" if no_score else f"WorldCAP: {_fmt(film_score)}"),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fs_line,
        color="white",
        zorder=300,
    )

    # Auto-resize the background patch to tightly cover all legend texts.
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [t_gt_bar, t_gt, t_wote_bar, t_wote, t_wc_bar, t_wc]
        if t_title is not None:
            texts = [t_title] + texts
        bboxes = [t.get_window_extent(renderer=renderer) for t in texts]
        x0 = min(bb.x0 for bb in bboxes)
        y0 = min(bb.y0 for bb in bboxes)
        x1 = max(bb.x1 for bb in bboxes)
        y1 = max(bb.y1 for bb in bboxes)

        # Convert display coords -> axes coords.
        inv = ax.transAxes.inverted()
        (ax_x0, ax_y0) = inv.transform((x0, y0))
        (ax_x1, ax_y1) = inv.transform((x1, y1))

        pad_x, pad_y = 0.012, 0.010
        bx0 = max(edge_margin, ax_x0 - pad_x)
        by0 = max(edge_margin, ax_y0 - pad_y)
        bx1 = min(1.0 - edge_margin, ax_x1 + pad_x)
        by1 = min(1.0 - edge_margin, ax_y1 + pad_y)

        box_patch.set_bounds(bx0, by0, bx1 - bx0, by1 - by0)
    except Exception:
        # If renderer-based sizing fails, keep a small default.
        box_patch.set_bounds(box_x - 0.008, box_y - 0.22, 0.32, 0.21)
    
    # ax.text(
    #     box_x,
    #     box_y - 1 * line_h,
    #     f"{bar_char} GT",
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     fontsize=fs_line,
    #     color=COLOR_GT,
    #     zorder=300,
    # )
    # ax.text(
    #     box_x,
    #     box_y - 2 * line_h,
    #     f"{bar_char} WoTE: {_fmt(baseline_score)}",
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     fontsize=fs_line,
    #     color=COLOR_WOTE,
    #     zorder=300,
    # )
    # ax.text(
    #     box_x,
    #     box_y - 3 * line_h,
    #     f"{bar_char} WorldCAP: {_fmt(film_score)}",
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     fontsize=fs_line,
    #     color=COLOR_WORLDCAP,
    #     zorder=300,
    # )

    configure_bev_ax(ax)
    configure_ax(ax)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    stem = out_path.stem

    # 2026-03 update: SVG-only outputs.
    # Keep the `out_format` argument for backward compatibility, but ignore it.
    svg_path = out_path.with_name(f"{stem}.svg")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.close(fig)
    return str(svg_path)


def _plot_bev_focus_compare_with_rollout(
    *,
    scene: Any,
    human_traj: Any,
    baseline_traj: Any,
    film_traj: Any,
    baseline_rollout: Any,
    film_rollout: Any,
    baseline_score: Optional[float],
    film_score: Optional[float],
    no_score: bool,
    out_format: str,
    out_dir: Path,
    out_name: str,
) -> str:
    """Focused visualization with simulator/controller rollout trajectories overlaid.

    Draws:
      - GT (human)
      - baseline/worldcap model-selected proposal trajectories
      - baseline/worldcap simulator rollout trajectories (dashed)

    Returns the saved image path.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
    from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
    from navsim.visualization.plots import configure_ax, configure_bev_ax

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

    COLOR_GT = "#80c5f0"
    COLOR_WOTE = "#27cfbe"
    COLOR_WORLDCAP = "#fa7f6f"
    
    COLOR_WOTE_SIM = "#0d736b"
    COLOR_WORLDCAP_SIM = "#a74335"

    gt_cfg = dict(TRAJECTORY_CONFIG["human"])
    gt_cfg["line_color"] = COLOR_GT
    gt_cfg["line_color_alpha"] = 0.85
    gt_cfg["line_width"] = 1.6
    # gt_cfg["line_width"] = 0.2
    gt_cfg["marker"] = None
    gt_cfg["marker_size"] = 0

    wote_cfg = dict(TRAJECTORY_CONFIG["agent"])
    wote_cfg["line_color"] = COLOR_WOTE
    wote_cfg["line_color_alpha"] = 0.95
    wote_cfg["line_width"] = 1.4
    # wote_cfg["line_width"] = 0.2
    wote_cfg["line_style"] = "-"
    wote_cfg["marker"] = None
    wote_cfg["marker_size"] = 0

    worldcap_cfg = dict(TRAJECTORY_CONFIG["agent"])
    worldcap_cfg["line_color"] = COLOR_WORLDCAP
    worldcap_cfg["line_color_alpha"] = 0.98
    worldcap_cfg["line_width"] = 1.4
    # worldcap_cfg["line_width"] = 0.2
    worldcap_cfg["line_style"] = "-"
    worldcap_cfg["marker"] = None
    worldcap_cfg["marker_size"] = 0

    # Proposals
    add_trajectory_to_bev_ax(ax, human_traj, gt_cfg)
    add_trajectory_to_bev_ax(ax, baseline_traj, wote_cfg)
    add_trajectory_to_bev_ax(ax, film_traj, worldcap_cfg)

    # Rollouts: expect (T, 3) or (T, 2) or (T, D). Only use x,y.
    def _to_xy(a: Any):
        try:
            import numpy as np

            arr = np.asarray(a)
            if arr.ndim == 3:
                # If a full bank accidentally passed, take first.
                arr = arr[0]
            if arr.ndim != 2 or arr.shape[1] < 2:
                return None
            return arr[:, :2]
        except Exception:
            return None

    wote_xy = _to_xy(baseline_rollout)
    wc_xy = _to_xy(film_rollout)
    if wote_xy is not None:
        # Keep axis convention consistent with navsim.visualization.bev
        # (it plots poses[:, 1] on x-axis and poses[:, 0] on y-axis).
        try:
            import numpy as np

            wote_xy_plot = np.concatenate([np.array([[0.0, 0.0]]), wote_xy], axis=0)
        except Exception:
            wote_xy_plot = wote_xy
        ax.plot(
            wote_xy_plot[:, 1],
            wote_xy_plot[:, 0],
            linestyle="--",
            linewidth=0.9,
            # linewidth=0.1,
            color=COLOR_WOTE_SIM,
            alpha=0.85,
            zorder=250,
        )
    if wc_xy is not None:
        try:
            import numpy as np

            wc_xy_plot = np.concatenate([np.array([[0.0, 0.0]]), wc_xy], axis=0)
        except Exception:
            wc_xy_plot = wc_xy
        ax.plot(
            wc_xy_plot[:, 1],
            wc_xy_plot[:, 0],
            linestyle="--",
            linewidth=0.9,
            # linewidth=0.1,
            color=COLOR_WORLDCAP_SIM,
            alpha=0.98,
            zorder=250,
        )

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v) * 100.0:.1f}"
        except Exception:
            return "n/a"

    edge_margin = 0.045
    box_x, box_y = edge_margin, 1.0 - edge_margin
    fs_title = 10
    fs_line = 9
    line_h = 0.052

    box_patch = FancyBboxPatch(
        (0.0, 0.0),
        0.1,
        0.1,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        transform=ax.transAxes,
        facecolor="black",
        edgecolor="#333333",
        linewidth=1.0,
        alpha=0.35,
        zorder=299,
    )
    ax.add_patch(box_patch)

    bar_solid = "━"
    bar_dash = "┄"
    t_title = None
    if not no_score:
        t_title = ax.text(
            box_x,
            box_y,
            "PDM Score",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fs_title,
            color="white",
            zorder=300,
        )

    row0 = 0 if no_score else 1
    text_x = box_x + 0.035

    # 5 rows: GT / WoTE / WorldCAP / WoTE(sim) / WorldCAP(sim)
    t_gt_bar = ax.text(box_x, box_y - (row0 + 0) * line_h, bar_solid, transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color=COLOR_GT, zorder=300)
    t_gt = ax.text(text_x, box_y - (row0 + 0) * line_h, "GT", transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color="white", zorder=300)

    t_wote_bar = ax.text(box_x, box_y - (row0 + 1) * line_h, bar_solid, transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color=COLOR_WOTE, zorder=300)
    t_wote = ax.text(text_x, box_y - (row0 + 1) * line_h, ("WoTE" if no_score else f"WoTE: {_fmt(baseline_score)}"), transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color="white", zorder=300)

    t_wc_bar = ax.text(box_x, box_y - (row0 + 2) * line_h, bar_solid, transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color=COLOR_WORLDCAP, zorder=300)
    t_wc = ax.text(text_x, box_y - (row0 + 2) * line_h, ("WorldCAP" if no_score else f"WorldCAP: {_fmt(film_score)}"), transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color="white", zorder=300)

    t_wote_sim_bar = ax.text(box_x, box_y - (row0 + 3) * line_h, bar_dash, transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color=COLOR_WOTE_SIM, zorder=300)
    t_wote_sim = ax.text(text_x, box_y - (row0 + 3) * line_h, "WoTE(sim)", transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color="white", zorder=300)

    t_wc_sim_bar = ax.text(box_x, box_y - (row0 + 4) * line_h, bar_dash, transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color=COLOR_WORLDCAP_SIM, zorder=300)
    t_wc_sim = ax.text(text_x, box_y - (row0 + 4) * line_h, "WorldCAP(sim)", transform=ax.transAxes, va="top", ha="left", fontsize=fs_line, color="white", zorder=300)

    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [t_gt_bar, t_gt, t_wote_bar, t_wote, t_wc_bar, t_wc, t_wote_sim_bar, t_wote_sim, t_wc_sim_bar, t_wc_sim]
        if t_title is not None:
            texts = [t_title] + texts
        bboxes = [t.get_window_extent(renderer=renderer) for t in texts]
        x0 = min(bb.x0 for bb in bboxes)
        y0 = min(bb.y0 for bb in bboxes)
        x1 = max(bb.x1 for bb in bboxes)
        y1 = max(bb.y1 for bb in bboxes)

        inv = ax.transAxes.inverted()
        (ax_x0, ax_y0) = inv.transform((x0, y0))
        (ax_x1, ax_y1) = inv.transform((x1, y1))

        pad_x, pad_y = 0.012, 0.010
        bx0 = max(edge_margin, ax_x0 - pad_x)
        by0 = max(edge_margin, ax_y0 - pad_y)
        bx1 = min(1.0 - edge_margin, ax_x1 + pad_x)
        by1 = min(1.0 - edge_margin, ax_y1 + pad_y)
        box_patch.set_bounds(bx0, by0, bx1 - bx0, by1 - by0)
    except Exception:
        box_patch.set_bounds(box_x - 0.008, box_y - 0.30, 0.40, 0.30)

    configure_bev_ax(ax)
    configure_ax(ax)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    stem = out_path.stem

    # 2026-03 update: SVG-only outputs.
    # Keep the `out_format` argument for backward compatibility, but ignore it.
    svg_path = out_path.with_name(f"{stem}.svg")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.close(fig)
    return str(svg_path)


def _ensure_local_devkit_on_path() -> None:
    """Mimic the repo's eval scripts that export PYTHONPATH for local packages."""

    repo_root = Path(__file__).resolve().parents[2]
    navsim_pkg = repo_root / "navsim"
    nuplan_pkg = repo_root / "nuplan-devkit"

    for p in [repo_root, navsim_pkg, nuplan_pkg]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    # Consistent map version default (same as eval scripts)
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")


# Headless backend for servers/ssh.
import matplotlib

matplotlib.use("Agg")


def _default_navsim_log_path(split: str) -> Optional[str]:
    root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
    if not root:
        return None
    base = Path(root) / "navsim_logs"
    candidate = base / split
    if candidate.exists():
        return str(candidate)

    fallback_map = {"val": "trainval", "train": "trainval"}
    alt = fallback_map.get(split)
    if alt:
        alt_candidate = base / alt
        if alt_candidate.exists():
            print(
                f"[WARN] navsim_logs split '{split}' not found; falling back to '{alt}'.",
                file=sys.stderr,
            )
            return str(alt_candidate)

    return str(candidate)


# Backward compatible alias (older drafts used the plural form).
def _default_navsim_logs_path(split: str) -> Optional[str]:
    return _default_navsim_log_path(split)


def _default_sensor_blobs_path(split: str) -> Optional[str]:
    root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
    if not root:
        return None
    base = Path(root) / "sensor_blobs"
    candidate = base / split
    if candidate.exists():
        return str(candidate)

    fallback_map = {"val": "trainval", "train": "trainval"}
    alt = fallback_map.get(split)
    if alt:
        alt_candidate = base / alt
        if alt_candidate.exists():
            print(
                f"[WARN] sensor_blobs split '{split}' not found; falling back to '{alt}'.",
                file=sys.stderr,
            )
            return str(alt_candidate)

    return str(candidate)


def _list_subdirs(p: Path) -> str:
    try:
        if not p.exists():
            return "<missing>"
        names = [c.name for c in p.iterdir() if c.is_dir()]
        names = sorted(names)
        return ", ".join(names) if names else "<none>"
    except Exception:
        return "<unavailable>"


def _default_metric_cache_path() -> Optional[str]:
    root = (os.environ.get("NAVSIM_EXP_ROOT") or "").strip()
    if not root:
        return None
    return str(Path(root) / "metric_cache")


def _load_metric_cache(metric_cache_path: Path, token: str) -> MetricCache:
    from navsim.common.dataloader import MetricCacheLoader

    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    if token not in metric_cache_loader.metric_cache_paths:
        raise KeyError(f"token not found in metric cache: {token}")

    metric_cache_file = metric_cache_loader.metric_cache_paths[token]
    with lzma.open(metric_cache_file, "rb") as f:
        metric_cache = pickle.load(f)
    return metric_cache


def _get_metric_cache_file(metric_cache_path: Path, token: str) -> Path:
    from navsim.common.dataloader import MetricCacheLoader

    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    if token not in metric_cache_loader.metric_cache_paths:
        raise KeyError(f"token not found in metric cache: {token}")
    return Path(metric_cache_loader.metric_cache_paths[token])


def _infer_log_name_from_metric_cache_file(metric_cache_file: Path) -> Optional[str]:
    """Infer log_name from metric_cache path.

    Typical layout:
      <metric_cache_root>/<log_name>/unknown/<token>/metric_cache.pkl
    """

    parts = list(metric_cache_file.parts)
    try:
        idx = parts.index("metric_cache")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except Exception:
        pass
    return None


def _detect_split_for_log(
    *,
    openscene_root: Path,
    preferred_split: str,
    log_name: str,
) -> Optional[str]:
    """Pick the dataset split that contains <log_name>.pkl under navsim_logs."""

    logs_root = openscene_root / "navsim_logs"
    if not logs_root.exists():
        return None

    candidates = []
    if preferred_split:
        candidates.append(preferred_split)
    # common splits in this repo
    for s in ["test", "trainval", "mini", "val", "train"]:
        if s not in candidates:
            candidates.append(s)
    # also scan any other dirs
    for d in sorted([p.name for p in logs_root.iterdir() if p.is_dir()]):
        if d not in candidates:
            candidates.append(d)

    for split in candidates:
        p = logs_root / split / f"{log_name}.pkl"
        if p.exists():
            return split
    return None


def _build_scene_loader(
    *,
    cfg: Any,
    agent: AbstractAgent,
    token: str,
    log_name: Optional[str],
    navsim_log_path: Path,
    sensor_blobs_path: Path,
) -> SceneLoader:
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter

    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    # Critical for single-token lookup:
    # Some default scene filters leave frame_interval=None, which becomes num_frames (non-overlapping).
    # That means only every-kth token is kept, and arbitrary tokens may be missing.
    # Eval scripts often use frame_interval=1 (e.g., navtest), so we enforce it here.
    scene_filter.frame_interval = 1
    scene_filter.tokens = [token]
    if log_name:
        scene_filter.log_names = [log_name]

    return SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )


def _instantiate_sim_and_scorer(cfg: Any) -> Tuple[PDMSimulator, PDMScorer]:
    from hydra.utils import instantiate
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    simulator: PDMSimulator = instantiate(cfg.simulator)

    # Sanitize scorer config to avoid unexpected root-level keys being passed in.
    scorer_cfg: Dict[str, Any] = {
        "_target_": cfg.scorer._target_,
        "_convert_": getattr(cfg.scorer, "_convert_", "all"),
        "proposal_sampling": cfg.scorer.proposal_sampling,
        "config": cfg.scorer.config,
    }
    scorer: PDMScorer = instantiate(scorer_cfg)

    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    return simulator, scorer


def _compose_cfg(
    *,
    config_dir: Path,
    split: str,
    token: str,
    output_dir: Path,
    navsim_log_path: Path,
    sensor_blobs_path: Path,
    metric_cache_path: Path,
    ckpt_path: Path,
    cluster_file_path: Optional[Path],
    sim_reward_dict_path: Optional[Path],
    controller_ref_traj_path: Optional[Path],
    controller_exec_traj_path: Optional[Path],
    controller_response_predictor_path: Optional[Path],
    simulator_post_apply_mode: str,
    ctrl_enable_world_model: Optional[bool],
    ctrl_world_model_fusion: Optional[str],
    ctrl_world_model_strength: Optional[float],
) -> Any:
    from hydra import compose

    def _q(v: str) -> str:
        """Quote a string for Hydra override grammar.

        Needed for paths containing special chars like '=' (e.g. epoch=29-step=...).
        """

        v = str(v)
        return "'" + v.replace("'", "\\'") + "'"

    overrides = [
        "agent=WoTE_agent",
        f"split={_q(split)}",
        "experiment_name=token_viz",
        f"output_dir={_q(output_dir)}",
        f"navsim_log_path={_q(navsim_log_path)}",
        f"sensor_blobs_path={_q(sensor_blobs_path)}",
        f"metric_cache_path={_q(metric_cache_path)}",
        "evaluate_all_trajectories=true",
        f"agent.checkpoint_path={_q(ckpt_path)}",
        # Match eval script behavior: allow online post transforms to affect rollout dynamics.
        f"+simulator.post_params.apply_mode={_q(simulator_post_apply_mode)}",
    ]

    # Match eval script controller overrides (only WM toggles here; other controller flags remain as agent defaults).
    if ctrl_enable_world_model is not None:
        overrides.append(f"++agent.config.controller_condition_on_world_model={str(bool(ctrl_enable_world_model)).lower()}")
    if ctrl_world_model_fusion is not None:
        overrides.append(f"++agent.config.controller_world_model_fusion={_q(ctrl_world_model_fusion)}")
    if ctrl_world_model_strength is not None:
        overrides.append(f"++agent.config.controller_world_model_strength={float(ctrl_world_model_strength)}")

    # Optional: force anchors + sim reward dict.
    if cluster_file_path is not None:
        overrides.append(f"++agent.config.cluster_file_path={_q(cluster_file_path)}")
    if sim_reward_dict_path is not None:
        overrides.append(f"++agent.config.sim_reward_dict_path={_q(sim_reward_dict_path)}")
    if controller_ref_traj_path is not None:
        overrides.append(f"++agent.config.controller_ref_traj_path={_q(controller_ref_traj_path)}")
    if controller_exec_traj_path is not None:
        overrides.append(f"++agent.config.controller_exec_traj_path={_q(controller_exec_traj_path)}")
    if controller_response_predictor_path is not None:
        overrides.append(
            f"++agent.config.controller_response_predictor_path={_q(controller_response_predictor_path)}"
        )

    # For a single-token visualization script we don't need Ray workers.
    overrides.append("worker=sequential")

    cfg = compose(config_name="default_run_pdm_score", overrides=overrides)

    # Narrow down to single token as a guardrail (even if scene_filter defaults change).
    try:
        cfg.scene_filter.tokens = [token]
    except Exception:
        pass

    return cfg


def _run_one(
    *,
    cfg: Any,
    token: str,
    out_dir: Path,
    out_name: str,
    log_name: Optional[str],
    compute_isolated_scores: bool = False,
    skip_detailed_plot: bool = False,
) -> Dict[str, Any]:
    from hydra.utils import instantiate
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.evaluate.pdm_score import pdm_score_multiTraj
    from navsim.visualization.plots import plot_bev_with_agent_and_simulation

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.is_eval = True

    simulator, scorer = _instantiate_sim_and_scorer(cfg)

    metric_cache = _load_metric_cache(Path(cfg.metric_cache_path), token)

    scene_loader = _build_scene_loader(
        cfg=cfg,
        agent=agent,
        token=token,
        log_name=log_name,
        navsim_log_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
    )

    use_future_frames = bool(getattr(getattr(agent, "config", None), "use_fut_frames", False))
    agent_input = scene_loader.get_agent_input_from_token(token, use_fut_frames=use_future_frames)

    if agent.requires_scene:
        scene = scene_loader.get_scene_from_token(token)
        agentout = agent.compute_trajectory(agent_input, scene)
    else:
        scene = scene_loader.get_scene_from_token(token)
        agentout = agent.compute_trajectory(agent_input)

    # NOTE:
    # Batch eval CSVs are produced with evaluate_all_trajectories=false (fast mode).
    # Visualization needs evaluate_all_trajectories=true (full mode) to get all simulated proposals.
    # These can differ in practice, so compute BOTH.
    (
        pdm_pred_fast,
        _pdm_best_fast,
        _sim_fast,
        _pred_fast,
        _best_pdm_idx_fast,
        pred_idx_fast,
    ) = pdm_score_multiTraj(
        metric_cache=metric_cache,
        model_trajectories=agentout["trajectories"],
        trajectory_Anchor=agentout.get("trajectoryAnchor"),
        model_scores=agentout["trajectory_scores"],
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        anchor_save_dir=None,
        anchor_save_name=None,
        anchor_overwrite=False,
        evaluate_all_trajectories=False,
    )

    (
        pdm_pred_full,
        pdm_best_full,
        simulated_states_all_egoframe,
        _pred_states_all_egoframe,
        best_pdm_idx,
        pred_idx_full,
    ) = pdm_score_multiTraj(
        metric_cache=metric_cache,
        model_trajectories=agentout["trajectories"],
        trajectory_Anchor=agentout.get("trajectoryAnchor"),
        model_scores=agentout["trajectory_scores"],
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        anchor_save_dir=None,
        anchor_save_name=None,
        anchor_overwrite=False,
        evaluate_all_trajectories=True,
    )

    isolated_scores_path: Optional[str] = None
    if compute_isolated_scores:
        import numpy as np

        model_trajectories = agentout["trajectories"]
        num_traj = len(model_trajectories)
        isolated_scores = np.full((num_traj,), np.nan, dtype=np.float32)

        # Isolated fast-mode scoring: each candidate is evaluated alone (baseline + that candidate).
        # This matches the batch-eval CSV's scoring path (evaluate_all_trajectories=false), but for all candidates.
        failures = 0
        for i in range(num_traj):
            try:
                pdm_pred_i, *_rest = pdm_score_multiTraj(
                    metric_cache=metric_cache,
                    model_trajectories=[model_trajectories[i]],
                    trajectory_Anchor=agentout.get("trajectoryAnchor"),
                    model_scores=[0.0],
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                    anchor_save_dir=None,
                    anchor_save_name=None,
                    anchor_overwrite=False,
                    evaluate_all_trajectories=False,
                )
                isolated_scores[i] = float(getattr(pdm_pred_i, "score"))
            except Exception:
                failures += 1
                isolated_scores[i] = np.nan

        out_dir.mkdir(parents=True, exist_ok=True)
        isolated_name = f"{Path(out_name).stem}_isolated_scores.npy"
        isolated_path = out_dir / isolated_name
        np.save(isolated_path, isolated_scores)
        isolated_scores_path = str(isolated_path)
        if failures:
            print(
                f"[WARN] isolated scoring had {failures}/{num_traj} failures (saved as NaN).",
                file=sys.stderr,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    human_traj = scene.get_future_trajectory()
    if not skip_detailed_plot:
        plot_bev_with_agent_and_simulation(
            scene,
            agent,
            best_pdm_idx=best_pdm_idx,
            best_pred_idx=pred_idx_full,
            human_trajectory=human_traj,
            agent_trajectory=agentout,
            simulation_state=simulated_states_all_egoframe,
            pdm_result_pred=pdm_pred_fast,
            pdm_result_best=pdm_best_full,
            save_path=str(out_dir),
            file_name=out_name,
        )

    # Model-selected trajectory (for focused compare plot).
    import numpy as np

    chosen_idx_model = int(np.argmax(agentout["trajectory_scores"]))
    chosen_traj = agentout["trajectories"][chosen_idx_model]

    chosen_rollout = None
    try:
        sim_arr = np.asarray(simulated_states_all_egoframe)
        if sim_arr.ndim == 3 and 0 <= chosen_idx_model < sim_arr.shape[0]:
            chosen_rollout = sim_arr[chosen_idx_model]
    except Exception:
        chosen_rollout = None

    return {
        "token": token,
        "out": str(out_dir / out_name),
        "scene": scene,
        "human_traj": human_traj,
        "chosen_traj": chosen_traj,
        "chosen_rollout": chosen_rollout,
        # CSV-aligned (fast mode)
        "pdm_pred": asdict(pdm_pred_fast),
        "pred_idx": int(pred_idx_fast),
        # Debug (full mode)
        "pdm_pred_full": asdict(pdm_pred_full),
        "pdm_best": asdict(pdm_best_full),
        "pred_idx_full": int(pred_idx_full),
        "best_pdm_idx": int(best_pdm_idx),
        "isolated_scores_path": isolated_scores_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--token", default=None, help="scenario token to visualize (single)")
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=None,
        help=(
            "batch tokens to visualize. Accepts space-separated tokens (recommended), "
            "or a single quoted string containing tokens separated by spaces/commas/newlines. "
            "Also supports the shortcut: --tokens /path/to/tokens.txt"
        ),
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help="text file containing tokens (one per line; supports # comments)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="dataset split name (common: test/trainval/mini). If you pass val/train and only trainval exists, the script auto-falls-back. Also tolerates typo: text->test.",
    )

    parser.add_argument("--baseline-ckpt", required=True, help="path to baseline checkpoint")
    parser.add_argument("--film-ckpt", required=True, help="path to film03 checkpoint")

    # Controller bundle alignment (matches tool/evaluate/eval_ckpts_20260225_valstyles.sh)
    parser.add_argument(
        "--controller-ref-traj-path",
        default=None,
        help="(optional) override agent.config.controller_ref_traj_path",
    )
    parser.add_argument(
        "--controller-exec-traj-path",
        default=None,
        help="(optional) override agent.config.controller_exec_traj_path",
    )
    parser.add_argument(
        "--controller-response-predictor-path",
        default=None,
        help="(optional) override agent.config.controller_response_predictor_path",
    )

    # --- Config alignment knobs (to match eval scripts) ---
    parser.add_argument(
        "--sim-post-apply-mode",
        default="online",
        choices=["auto", "online", "offline"],
        help="PDMSimulator post apply mode. Eval scripts often use 'online'.",
    )
    parser.add_argument(
        "--baseline-ctrl-wm",
        type=int,
        default=0,
        choices=[0, 1],
        help="Override agent.config.controller_condition_on_world_model for baseline run (default: 0 like eval script).",
    )
    parser.add_argument(
        "--film-ctrl-wm",
        type=int,
        default=1,
        choices=[0, 1],
        help="Override agent.config.controller_condition_on_world_model for film run (default: 1 like eval script).",
    )
    parser.add_argument(
        "--film-wm-fusion",
        default="film03",
        help="Override agent.config.controller_world_model_fusion for film run (default: film03).",
    )
    parser.add_argument(
        "--film-wm-strength",
        type=float,
        default=0.3,
        help="Override agent.config.controller_world_model_strength for film run (default: 0.3).",
    )
    parser.add_argument(
        "--style-idx",
        type=int,
        default=None,
        help="If set, fixes controller+sim style idx (sets WOTE_CTRL_STYLE_IDX and PDM_SIM_STYLE_IDX).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for python/random/numpy/torch (best-effort).",
    )

    parser.add_argument(
        "--compute-isolated-scores",
        action="store_true",
        help=(
            "If set, computes isolated fast-mode PDM score for every candidate trajectory "
            "(evaluate_all_trajectories=false; baseline + that candidate) and saves a .npy next to the outputs."
        ),
    )

    parser.add_argument(
        "--noScore",
        action="store_true",
        help="(Deprecated) Ignored. This script now always outputs both score and noscore SVG variants.",
    )

    parser.add_argument(
        "--format",
        default="svg",
        choices=["png", "svg", "both"],
        help="(Deprecated) Output format for the focused BEV figure. This script now saves SVG only.",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Also save the original detailed per-ckpt plots (all candidates + full score box).",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory to save images (default: $NAVSIM_EXP_ROOT/token_viz)",
    )

    parser.add_argument(
        "--navsim-log-path",
        default=None,
        help="path to navsim logs (default: $OPENSCENE_DATA_ROOT/navsim_logs/<split>)",
    )
    parser.add_argument(
        "--sensor-blobs-path",
        default=None,
        help="path to sensor blobs (default: $OPENSCENE_DATA_ROOT/sensor_blobs/<split>)",
    )
    parser.add_argument(
        "--metric-cache-path",
        default=None,
        help="path to metric_cache dir (default: $NAVSIM_EXP_ROOT/metric_cache)",
    )

    parser.add_argument(
        "--cluster-file-path",
        default=None,
        help="(optional) override agent.config.cluster_file_path",
    )
    parser.add_argument(
        "--sim-reward-dict-path",
        default=None,
        help="(optional) override agent.config.sim_reward_dict_path",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # 2026-03 update: always write SVG only.
    fixed_out_format = "svg"
    try:
        if str(getattr(args, "format", "")).lower() != "svg":
            print(
                f"[WARN] --format {args.format!r} is ignored; this script now saves SVG only.",
                file=sys.stderr,
            )
    except Exception:
        pass

    # Must happen before importing navsim/nuplan modules.
    _ensure_local_devkit_on_path()

    # Collect tokens (single, list, and/or from file).
    def _split_tokens_string(s: str) -> list[str]:
        """Split a single CLI string into tokens.

        Supports separators: whitespace, comma, semicolon, newline.
        """

        if s is None:
            return []
        raw = str(s).strip()
        if not raw:
            return []
        for sep in [",", ";", "\n", "\t", "\r"]:
            raw = raw.replace(sep, " ")
        return [p for p in raw.split(" ") if p]

    tokens_raw = []
    if args.token:
        tokens_raw.append(str(args.token))
    if args.tokens:
        # Convenience: allow `--tokens /path/to/tokens.txt`.
        if len(args.tokens) == 1:
            maybe_path = Path(str(args.tokens[0])).expanduser()
            if maybe_path.suffix.lower() in {".txt", ".list"} and maybe_path.is_file():
                try:
                    for line in maybe_path.read_text().splitlines():
                        s = line.strip()
                        if not s or s.startswith("#"):
                            continue
                        tokens_raw.append(s)
                except Exception as e:
                    print(f"ERROR: failed to read --tokens file: {maybe_path} ({e})", file=sys.stderr)
                    return 2
            else:
                # Allow a single quoted string that contains multiple tokens.
                # Examples:
                #   --tokens "tok1,tok2,tok3"
                #   --tokens "tok1 tok2 tok3"
                single = str(args.tokens[0])
                parts = _split_tokens_string(single)
                # If splitting yields exactly one item, preserve original behavior.
                tokens_raw.extend(parts if len(parts) > 1 else [single])
        else:
            tokens_raw.extend([str(t) for t in args.tokens])
    if args.token_file:
        p = Path(args.token_file).expanduser().resolve()
        if not p.exists():
            print(f"ERROR: --token-file not found: {p}", file=sys.stderr)
            return 2
        try:
            for line in p.read_text().splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                tokens_raw.append(s)
        except Exception as e:
            print(f"ERROR: failed to read --token-file: {p} ({e})", file=sys.stderr)
            return 2

    tokens: list[str] = []
    seen = set()
    for t in tokens_raw:
        s = (t or "").strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            tokens.append(s)

    if not tokens:
        print(
            "ERROR: no tokens provided. Use --token <TOK>, or --tokens <TOK1> <TOK2>..., or --token-file <path>.",
            file=sys.stderr,
        )
        return 2

    split_base = (args.split or "").strip()
    if split_base.lower() == "text":
        print("[WARN] split='text' looks like a typo; using 'test'.", file=sys.stderr)
        split_base = "test"

    baseline_ckpt = Path(args.baseline_ckpt).expanduser().resolve()
    film_ckpt = Path(args.film_ckpt).expanduser().resolve()

    if not baseline_ckpt.exists():
        print(f"ERROR: baseline ckpt not found: {baseline_ckpt}", file=sys.stderr)
        return 2
    if not film_ckpt.exists():
        print(f"ERROR: film ckpt not found: {film_ckpt}", file=sys.stderr)
        return 2

    # These roots are shared across tokens; per-token split detection may adjust log/blob paths.
    metric_cache_path_str = args.metric_cache_path or _default_metric_cache_path()

    if not metric_cache_path_str:
        print(
            "ERROR: --metric-cache-path is required if NAVSIM_EXP_ROOT is not set",
            file=sys.stderr,
        )
        return 2
    metric_cache_path = Path(metric_cache_path_str).expanduser().resolve()
    if not metric_cache_path.exists():
        print(f"ERROR: metric_cache_path not found: {metric_cache_path}", file=sys.stderr)
        return 2

    out_dir_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir_root is None:
        navsim_exp_root = (os.environ.get("NAVSIM_EXP_ROOT") or "").strip()
        if not navsim_exp_root:
            print(
                "ERROR: --out-dir is required if NAVSIM_EXP_ROOT is not set",
                file=sys.stderr,
            )
            return 2
        out_dir_root = (Path(navsim_exp_root) / "token_viz").resolve()

    # We'll create per-token output dirs inside the loop.

    cluster_file_path = (
        Path(args.cluster_file_path).expanduser().resolve() if args.cluster_file_path else None
    )
    sim_reward_dict_path = (
        Path(args.sim_reward_dict_path).expanduser().resolve() if args.sim_reward_dict_path else None
    )

    # Controller paths: default to the same artifacts used by the batch eval script.
    repo_root = Path(__file__).resolve().parents[2]
    default_ctrl_ref = (repo_root / "ControllerExp" / "Anchors_Original_256_centered.npy").resolve()
    default_ctrl_exec = (repo_root / "ControllerExp" / "generated" / "controller_styles.npz").resolve()
    default_ctrl_rp = (repo_root / "ControllerExp" / "generated" / "controller_response_predictor.pt").resolve()

    controller_ref_traj_path = (
        Path(args.controller_ref_traj_path).expanduser().resolve()
        if args.controller_ref_traj_path
        else (default_ctrl_ref if default_ctrl_ref.exists() else None)
    )
    controller_exec_traj_path = (
        Path(args.controller_exec_traj_path).expanduser().resolve()
        if args.controller_exec_traj_path
        else (default_ctrl_exec if default_ctrl_exec.exists() else None)
    )
    controller_response_predictor_path = (
        Path(args.controller_response_predictor_path).expanduser().resolve()
        if args.controller_response_predictor_path
        else (default_ctrl_rp if default_ctrl_rp.exists() else None)
    )

    config_dir = (
        Path(__file__).resolve().parents[2]
        / "navsim"
        / "planning"
        / "script"
        / "config"
        / "pdm_scoring"
    )

    if not config_dir.exists():
        print(f"ERROR: hydra config dir not found: {config_dir}", file=sys.stderr)
        return 2

    # Ensure Hydra is in a clean state (important for scripts invoked multiple times).
    try:
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
    except Exception:
        pass

    try:
        from hydra import initialize_config_dir
    except Exception as e:
        print(f"ERROR: failed to import hydra: {e}", file=sys.stderr)
        return 2

    # Fail early with a more actionable error if the current env is missing deps.
    try:
        import torch  # noqa: F401
        import pytorch_lightning  # noqa: F401
    except Exception as e:
        print(
            "ERROR: python env is missing required ML deps (torch/pytorch_lightning/etc).\n"
            "Hint: run this script with the same conda env you use for training/eval (e.g. wotenewnew),\n"
            "and/or install the missing pip/conda packages.\n"
            f"Root error: {e}",
            file=sys.stderr,
        )
        return 2

    # Determinism (best-effort). Note: simulator/controller may still have non-determinism depending on GPU/cuDNN.
    try:
        import numpy as np

        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    except Exception:
        pass

    # Optional: fix controller + simulator style index to match eval logs.
    if args.style_idx is not None:
        idx = int(args.style_idx)
        os.environ.setdefault("WOTE_CTRL_STYLE_SPLIT", "val")
        os.environ.setdefault("WOTE_CTRL_EVAL_SAMPLE", "0")
        os.environ["WOTE_CTRL_STYLE_IDX"] = str(idx)
        # Simulator bundle env (PDMSimulator reads these)
        # If you want a different bundle, set PDM_SIM_BUNDLE_PATH in your shell.
        os.environ.setdefault(
            "PDM_SIM_BUNDLE_PATH",
            str(controller_exec_traj_path) if controller_exec_traj_path is not None else str(default_ctrl_exec),
        )
        os.environ.setdefault("PDM_SIM_BUNDLE_APPLY", "all")
        os.environ.setdefault("PDM_SIM_EVAL_SAMPLE", "0")
        os.environ["PDM_SIM_STYLE_IDX"] = str(idx)

    print("\n=== Run config summary (for score alignment) ===")
    print(f"seed: {args.seed}")
    print(f"sim.post_params.apply_mode: {args.sim_post_apply_mode}")
    print(f"baseline: ctrl_wm={int(args.baseline_ctrl_wm)}")
    print(
        "film:     "
        f"ctrl_wm={int(args.film_ctrl_wm)}, "
        f"wm_fusion={args.film_wm_fusion}, "
        f"wm_strength={args.film_wm_strength}"
    )
    if args.style_idx is not None:
        print(
            "style_idx fixed via env: "
            f"WOTE_CTRL_STYLE_IDX={os.environ.get('WOTE_CTRL_STYLE_IDX')}, "
            f"PDM_SIM_STYLE_IDX={os.environ.get('PDM_SIM_STYLE_IDX')}, "
            f"PDM_SIM_BUNDLE_PATH={os.environ.get('PDM_SIM_BUNDLE_PATH')}"
        )
    else:
        print(
            "style_idx env not forced (scores may differ vs CSV if eval used fixed style): "
            f"WOTE_CTRL_STYLE_IDX={os.environ.get('WOTE_CTRL_STYLE_IDX')}, "
            f"PDM_SIM_STYLE_IDX={os.environ.get('PDM_SIM_STYLE_IDX')}"
        )
    if cluster_file_path is not None:
        print(f"cluster_file_path: {cluster_file_path}")
    if sim_reward_dict_path is not None:
        print(f"sim_reward_dict_path: {sim_reward_dict_path}")
    if controller_ref_traj_path is not None:
        print(f"controller_ref_traj_path: {controller_ref_traj_path}")
    if controller_exec_traj_path is not None:
        print(f"controller_exec_traj_path: {controller_exec_traj_path}")
    if controller_response_predictor_path is not None:
        print(f"controller_response_predictor_path: {controller_response_predictor_path}")
    print("=== End summary ===\n")

    def _hydra_init_ctx(p: Path):
        """Hydra API compatibility shim.

        Some Hydra versions don't support `version_base` for initialize_config_dir.
        """

        try:
            return initialize_config_dir(config_dir=str(p), version_base=None)
        except TypeError:
            return initialize_config_dir(config_dir=str(p))

    failures = 0
    with _hydra_init_ctx(config_dir):
        for token in tokens:
            split = split_base

            navsim_log_path_str = args.navsim_log_path or _default_navsim_log_path(split)
            sensor_blobs_path_str = args.sensor_blobs_path or _default_sensor_blobs_path(split)
            if not navsim_log_path_str:
                print(
                    "ERROR: --navsim-log-path is required if OPENSCENE_DATA_ROOT is not set",
                    file=sys.stderr,
                )
                return 2
            if not sensor_blobs_path_str:
                print(
                    "ERROR: --sensor-blobs-path is required if OPENSCENE_DATA_ROOT is not set",
                    file=sys.stderr,
                )
                return 2

            navsim_log_path = Path(navsim_log_path_str).expanduser().resolve()
            sensor_blobs_path = Path(sensor_blobs_path_str).expanduser().resolve()

            if not navsim_log_path.exists():
                root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
                hint = ""
                if root:
                    hint = f" (available splits under navsim_logs: {_list_subdirs(Path(root) / 'navsim_logs')})"
                print(f"ERROR: navsim_log_path not found: {navsim_log_path}{hint}", file=sys.stderr)
                return 2
            if not sensor_blobs_path.exists():
                root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
                hint = ""
                if root:
                    hint = f" (available splits under sensor_blobs: {_list_subdirs(Path(root) / 'sensor_blobs')})"
                print(f"ERROR: sensor_blobs_path not found: {sensor_blobs_path}{hint}", file=sys.stderr)
                return 2

            # Use metric_cache metadata to infer which log (and split) this token belongs to.
            log_name = None
            openscene_root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
            if openscene_root:
                try:
                    mc_file = _get_metric_cache_file(metric_cache_path, token)
                    log_name = _infer_log_name_from_metric_cache_file(mc_file)
                    if log_name:
                        detected = _detect_split_for_log(
                            openscene_root=Path(openscene_root),
                            preferred_split=split,
                            log_name=log_name,
                        )
                        if detected and detected != split:
                            print(
                                f"[WARN] token appears to belong to split '{detected}' (log={log_name}); overriding split '{split}' -> '{detected}'.",
                                file=sys.stderr,
                            )
                            split = detected
                            # Recompute default paths for the detected split (only if user didn't pass explicit paths).
                            if args.navsim_log_path is None:
                                navsim_log_path = Path(_default_navsim_log_path(split)).expanduser().resolve()
                            if args.sensor_blobs_path is None:
                                sensor_blobs_path = Path(_default_sensor_blobs_path(split)).expanduser().resolve()
                except Exception as e:
                    print(f"[WARN] failed to infer log/split from metric_cache: {e}", file=sys.stderr)

            # Per-token output directory + per-run timestamped base name.
            out_dir = _resolve_token_out_dir(out_dir_root, token)
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_base = f"{token}_{timestamp}"

            try:
                cfg_baseline = _compose_cfg(
                    config_dir=config_dir,
                    split=split,
                    token=token,
                    output_dir=out_dir,
                    navsim_log_path=navsim_log_path,
                    sensor_blobs_path=sensor_blobs_path,
                    metric_cache_path=metric_cache_path,
                    ckpt_path=baseline_ckpt,
                    cluster_file_path=cluster_file_path,
                    sim_reward_dict_path=sim_reward_dict_path,
                    controller_ref_traj_path=controller_ref_traj_path,
                    controller_exec_traj_path=controller_exec_traj_path,
                    controller_response_predictor_path=controller_response_predictor_path,
                    simulator_post_apply_mode=args.sim_post_apply_mode,
                    ctrl_enable_world_model=bool(args.baseline_ctrl_wm),
                    ctrl_world_model_fusion=None,
                    ctrl_world_model_strength=None,
                )
                cfg_film = _compose_cfg(
                    config_dir=config_dir,
                    split=split,
                    token=token,
                    output_dir=out_dir,
                    navsim_log_path=navsim_log_path,
                    sensor_blobs_path=sensor_blobs_path,
                    metric_cache_path=metric_cache_path,
                    ckpt_path=film_ckpt,
                    cluster_file_path=cluster_file_path,
                    sim_reward_dict_path=sim_reward_dict_path,
                    controller_ref_traj_path=controller_ref_traj_path,
                    controller_exec_traj_path=controller_exec_traj_path,
                    controller_response_predictor_path=controller_response_predictor_path,
                    simulator_post_apply_mode=args.sim_post_apply_mode,
                    ctrl_enable_world_model=bool(args.film_ctrl_wm),
                    ctrl_world_model_fusion=str(args.film_wm_fusion),
                    ctrl_world_model_strength=float(args.film_wm_strength),
                )

                res_baseline = _run_one(
                    cfg=cfg_baseline,
                    token=token,
                    out_dir=out_dir,
                    out_name=f"{out_base}_baseline.png",
                    log_name=log_name,
                    compute_isolated_scores=bool(args.compute_isolated_scores),
                    skip_detailed_plot=not bool(args.detailed),
                )
                res_film = _run_one(
                    cfg=cfg_film,
                    token=token,
                    out_dir=out_dir,
                    out_name=f"{out_base}_worldcap.png",
                    log_name=log_name,
                    compute_isolated_scores=bool(args.compute_isolated_scores),
                    skip_detailed_plot=not bool(args.detailed),
                )

                # Save the two model-selected trajectories as .npy in the same folder.
                try:
                    import numpy as np

                    def _to_poses_array(traj: Any) -> "np.ndarray":
                        poses = getattr(traj, "poses", None)
                        if poses is not None:
                            return np.asarray(poses, dtype=np.float32)
                        if hasattr(traj, "detach"):
                            return traj.detach().cpu().numpy().astype(np.float32)
                        return np.asarray(traj, dtype=np.float32)

                    np.save(out_dir / f"{out_base}_baseline_traj.npy", _to_poses_array(res_baseline["chosen_traj"]))
                    np.save(out_dir / f"{out_base}_worldcap_traj.npy", _to_poses_array(res_film["chosen_traj"]))
                except Exception as e:
                    print(f"[WARN] failed to save chosen trajectories as npy: {e}", file=sys.stderr)

                # Always produce 4 SVGs per token:
                #   - no sim: with score / no score
                #   - with sim: with score / no score
                baseline_score = res_baseline["pdm_pred"].get("score")
                film_score = res_film["pdm_pred"].get("score")

                focus_score_path = _plot_bev_focus_compare(
                    scene=res_baseline["scene"],
                    human_traj=res_baseline["human_traj"],
                    baseline_traj=res_baseline["chosen_traj"],
                    film_traj=res_film["chosen_traj"],
                    baseline_score=baseline_score,
                    film_score=film_score,
                    no_score=False,
                    out_format=fixed_out_format,
                    out_dir=out_dir,
                    out_name=f"{out_base}.svg",
                )
                focus_noscore_path = _plot_bev_focus_compare(
                    scene=res_baseline["scene"],
                    human_traj=res_baseline["human_traj"],
                    baseline_traj=res_baseline["chosen_traj"],
                    film_traj=res_film["chosen_traj"],
                    baseline_score=baseline_score,
                    film_score=film_score,
                    no_score=True,
                    out_format=fixed_out_format,
                    out_dir=out_dir,
                    out_name=f"{out_base}_noscore.svg",
                )

                focus_sim_score_path = _plot_bev_focus_compare_with_rollout(
                    scene=res_baseline["scene"],
                    human_traj=res_baseline["human_traj"],
                    baseline_traj=res_baseline["chosen_traj"],
                    film_traj=res_film["chosen_traj"],
                    baseline_rollout=res_baseline.get("chosen_rollout"),
                    film_rollout=res_film.get("chosen_rollout"),
                    baseline_score=baseline_score,
                    film_score=film_score,
                    no_score=False,
                    out_format=fixed_out_format,
                    out_dir=out_dir,
                    out_name=f"{out_base}_sim.svg",
                )
                focus_sim_noscore_path = _plot_bev_focus_compare_with_rollout(
                    scene=res_baseline["scene"],
                    human_traj=res_baseline["human_traj"],
                    baseline_traj=res_baseline["chosen_traj"],
                    film_traj=res_film["chosen_traj"],
                    baseline_rollout=res_baseline.get("chosen_rollout"),
                    film_rollout=res_film.get("chosen_rollout"),
                    baseline_score=baseline_score,
                    film_score=film_score,
                    no_score=True,
                    out_format=fixed_out_format,
                    out_dir=out_dir,
                    out_name=f"{out_base}_sim_noscore.svg",
                )

                print("=== token_viz done ===")
                print(f"token: {token}")
                print(f"out_dir: {out_dir}")
                print(f"focus(score):        {focus_score_path}")
                print(f"focus(noscore):      {focus_noscore_path}")
                print(f"focus+sim(score):    {focus_sim_score_path}")
                print(f"focus+sim(noscore):  {focus_sim_noscore_path}")
            except Exception as e:
                failures += 1
                print(f"[ERR] token failed: {token} ({e})", file=sys.stderr)
                continue

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
