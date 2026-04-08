#!/usr/bin/env python
"""Plot S01-S10 simulator-style executed anchors vs original anchors.

This script matches the scenario matrix in:
- tool/evaluate/eval_3ckpts_s01_s10_no_base.sh

For each scenario (S01..S10):
- Red: original anchor trajectories (ref anchors, typically 256x8x3 or 256x41x3)
- Blue: executed trajectories after running PDMSimulator(tracker_style, post_style, post_params)

Outputs 10 images under --out-dir.

Example:
  python tool/evaluate/plot_s01_s10_anchor_vs_exec.py \
    --ref ControllerExp/Anchors_Original_256_centered.npy \
    --out-dir ControllerInTheLoop/step0_validationOfSimulation/plots_s01_s10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List

import numpy as np

# Make local packages importable without requiring a pre-set PYTHONPATH.
# Matches the environment assumptions in tool/evaluate/eval_3ckpts_s01_s10_no_base.sh
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nuplan-devkit"))

# Use a non-interactive backend for headless servers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


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

    return np.stack([x_new, y_new, yaw_new], axis=-1)


def resample_anchors_to_41(anchors: np.ndarray) -> np.ndarray:
    """Resample anchors [256,T,>=3] to [256,41,3]."""
    b, t, d = anchors.shape
    if b != 256:
        raise ValueError(f"Expected 256 anchors, got {b}")
    if d < 3:
        raise ValueError("Anchors last dim must be >=3 (x,y,yaw)")

    out = np.zeros((b, 41, 3), dtype=np.float32)
    for i in range(b):
        out[i] = _resample_traj_xyz(anchors[i, :, :3], 41)
    return out


def ego_from_anchor_pair(p0: np.ndarray, p1: np.ndarray, dt: float = 0.1) -> EgoState:
    """Construct EgoState using first two anchor poses to estimate initial velocity/yaw_rate."""
    x0, y0, yaw0 = float(p0[0]), float(p0[1]), float(p0[2])
    x1, y1, yaw1 = float(p1[0]), float(p1[1]), float(p1[2])
    dx, dy, dyaw = (x1 - x0), (y1 - y0), (yaw1 - yaw0)

    vx_ego, vy_ego = dx / dt, dy / dt
    cos0, sin0 = np.cos(yaw0), np.sin(yaw0)
    vx = vx_ego * cos0 - vy_ego * sin0
    vy = vx_ego * sin0 + vy_ego * cos0
    yaw_rate = dyaw / dt

    vec9 = [0, x0, y0, yaw0, vx, vy, 0.0, 0.0, yaw_rate]
    return EgoState.deserialize(vec9, get_pacifica_parameters())


def _global_to_ego_xyyaw_all(global_states: np.ndarray, initial_ego_state: EgoState) -> np.ndarray:
    """Convert global (B,T,>=3) to ego-frame (B,T,3) using initial ego pose."""
    xy_yaw = global_states[..., :3]

    x0 = float(initial_ego_state.rear_axle.x)
    y0 = float(initial_ego_state.rear_axle.y)
    yaw0 = float(initial_ego_state.rear_axle.heading)

    dx = xy_yaw[..., 0] - x0
    dy = xy_yaw[..., 1] - y0

    cos0, sin0 = np.cos(yaw0), np.sin(yaw0)
    x_ego = dx * cos0 + dy * sin0
    y_ego = -dx * sin0 + dy * cos0
    yaw_ego = _wrap_angle(xy_yaw[..., 2] - yaw0)

    return np.stack([x_ego, y_ego, yaw_ego], axis=-1).astype(np.float32)


def _global_to_ego_vxy_all(global_states: np.ndarray, initial_ego_state: EgoState) -> np.ndarray:
    """Convert global (B,T,>=5) velocities to ego-frame (B,T,2)."""
    vx = global_states[..., 3]
    vy = global_states[..., 4]

    yaw0 = float(initial_ego_state.rear_axle.heading)
    cos0, sin0 = np.cos(yaw0), np.sin(yaw0)

    vx_ego = vx * cos0 + vy * sin0
    vy_ego = -vx * sin0 + vy * cos0
    return np.stack([vx_ego, vy_ego], axis=-1).astype(np.float32)


def _ego_to_global_states_11(ego_xyyaw_41: np.ndarray, initial_ego_state: EgoState) -> np.ndarray:
    """Build (T,11) state array in global frame from ego-frame x,y,yaw (T,3)."""
    x0 = float(initial_ego_state.rear_axle.x)
    y0 = float(initial_ego_state.rear_axle.y)
    yaw0 = float(initial_ego_state.rear_axle.heading)
    cos0, sin0 = np.cos(yaw0), np.sin(yaw0)

    xe = ego_xyyaw_41[:, 0]
    ye = ego_xyyaw_41[:, 1]
    yawe = ego_xyyaw_41[:, 2]

    xg = x0 + xe * cos0 - ye * sin0
    yg = y0 + xe * sin0 + ye * cos0
    yawg = yaw0 + yawe

    states = np.zeros((ego_xyyaw_41.shape[0], 11), dtype=np.float64)
    states[:, 0] = xg
    states[:, 1] = yg
    states[:, 2] = yawg
    return states


@dataclass(frozen=True)
class Scenario:
    tag: str
    tracker_style: str
    post_style: str
    heading_scale: float
    speed_scale: float
    heading_bias: float
    speed_bias: float
    noise_std: float


SCENARIOS: List[Scenario] = [
    # Keep same ordering as eval_3ckpts_s01_s10_no_base.sh
    Scenario("S07_unstable_none", "unstable", "none", 1.0, 1.0, 0.0, 0.0, 0.0),
    Scenario("S08_yaw_scale_12", "default", "yaw_scale", 1.2, 1.0, 0.0, 0.0, 0.0),
    Scenario("S09_speed_scale_08", "default", "speed_scale", 1.0, 0.8, 0.0, 0.0, 0.0),
    Scenario("S10_noise_02", "default", "none", 1.0, 1.0, 0.0, 0.0, 0.2),
    Scenario("S01_default_none", "default", "none", 1.0, 1.0, 0.0, 0.0, 0.0),
    Scenario("S02_default_post1515", "default", "yaw_speed_extreme", 1.5, 1.5, 0.0, 0.0, 0.0),
    Scenario("S03_aggressive_none", "aggressive", "none", 1.0, 1.0, 0.0, 0.0, 0.0),
    Scenario("S04_safe_none", "safe", "none", 1.0, 1.0, 0.0, 0.0, 0.0),
    Scenario("S05_sluggish_none", "sluggish", "none", 1.0, 1.0, 0.0, 0.0, 0.0),
    Scenario("S06_high_jitter_none", "high_jitter", "none", 1.0, 1.0, 0.0, 0.0, 0.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ref",
        type=str,
        default="ControllerExp/Anchors_Original_256_centered.npy",
        help="Original anchor .npy (red).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="ControllerInTheLoop/step0_validationOfSimulation/plots_s01_s10",
        help="Directory to write png (and optional npy).",
    )
    p.add_argument(
        "--red",
        type=str,
        default="anchor",
        choices=["anchor", "default_exec"],
        help="What to plot in red: original anchor, or anchor executed by default controller.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    # Note: matplotlib alpha: 1.0 = opaque, 0.0 = fully transparent.
    # Default goal: blue stands out; red stays visible but lighter.
    p.add_argument("--alpha-blue", type=float, default=0.85)
    p.add_argument("--alpha-red", type=float, default=0.08)
    p.add_argument("--lw-blue", type=float, default=1.2)
    p.add_argument("--lw-red", type=float, default=0.8)
    p.add_argument(
        "--xy-mode",
        type=str,
        default="sim",
        choices=["sim", "post"],
        help=(
            "XY source for executed trajectories. "
            "sim: use simulator's x,y (post_style won't change x,y in this repo). "
            "post: rebuild x,y by integrating post-processed yaw+speed (visualization-only)."
        ),
    )
    p.add_argument(
        "--max-trajs",
        type=int,
        default=256,
        help="Plot only first N trajectories (speed).",
    )
    p.add_argument(
        "--save-exec-npy",
        action="store_true",
        help="Also save executed trajectories (blue) as .npy (256,41,3).",
    )
    p.add_argument(
        "--plot-yaw-speed",
        action="store_true",
        help="Also save yaw(t)/speed(t) quantile-band plots (helps visualize post_style effects).",
    )
    return p.parse_args()


DEFAULT_SCEN = Scenario(
    tag="DEFAULT_EXEC",
    tracker_style="default",
    post_style="none",
    heading_scale=1.0,
    speed_scale=1.0,
    heading_bias=0.0,
    speed_bias=0.0,
    noise_std=0.0,
)


def _simulate_exec_global(ref: np.ndarray, scen: Scenario, seed: int) -> tuple[np.ndarray, EgoState, np.ndarray]:
    """Simulate proposals and return (exec_global_states, initial_state, ref_41_ego)."""
    rng = np.random.default_rng(seed)
    # make PDMSimulator noise deterministic if it uses np.random
    np.random.seed(seed)

    ref = ref.astype(np.float32)
    if ref.shape[1] != 41:
        ref_41 = resample_anchors_to_41(ref)
    else:
        ref_41 = ref[:, :, :3]

    proposal_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    initial_state = ego_from_anchor_pair(ref_41[0, 0, :3], ref_41[0, 1, :3], dt=0.1)

    post_params = {
        "style": scen.post_style,
        "heading_scale": float(scen.heading_scale),
        "heading_bias": float(scen.heading_bias),
        "speed_scale": float(scen.speed_scale),
        "speed_bias": float(scen.speed_bias),
        "noise_std": float(scen.noise_std),
        "seed": int(rng.integers(0, 2**31 - 1)),
    }

    simulator = PDMSimulator(
        proposal_sampling=proposal_sampling,
        tracker_style=scen.tracker_style,
        post_style=scen.post_style,
        post_params=post_params,
    )

    # Build proposal reference states in global frame: (256,41,11)
    proposal_states = np.zeros((ref_41.shape[0], 41, 11), dtype=np.float64)
    for a_idx in range(ref_41.shape[0]):
        proposal_states[a_idx] = _ego_to_global_states_11(ref_41[a_idx], initial_state)

    exec_global = simulator.simulate_proposals(proposal_states, initial_state)

    return exec_global, initial_state, ref_41


def _rebuild_xy_from_post_yaw_speed(exec_global: np.ndarray, *, dt: float) -> np.ndarray:
    """Visualization-only: rebuild x,y by integrating velocity aligned with (post) yaw.

    In this codebase, post_style modifies yaw/vx/vy but does NOT update x/y.
    This helper makes yaw_scale/speed_scale effects visible in XY.

    Args:
        exec_global: (B,T,11) global states after simulate_proposals (already post-transformed).
        dt: timestep seconds.

    Returns:
        new_global: (B,T,11) with x,y replaced; vx,vy also aligned with yaw.
    """
    new_global = np.array(exec_global, copy=True)
    x0 = exec_global[:, 0, 0]
    y0 = exec_global[:, 0, 1]
    yaw = exec_global[:, :, 2]
    vx = exec_global[:, :, 3]
    vy = exec_global[:, :, 4]

    speed = np.sqrt(vx**2 + vy**2)
    vx2 = speed * np.cos(yaw)
    vy2 = speed * np.sin(yaw)

    B, T, _ = exec_global.shape
    x = np.zeros((B, T), dtype=np.float64)
    y = np.zeros((B, T), dtype=np.float64)
    x[:, 0] = x0
    y[:, 0] = y0
    for t in range(1, T):
        x[:, t] = x[:, t - 1] + vx2[:, t - 1] * dt
        y[:, t] = y[:, t - 1] + vy2[:, t - 1] * dt

    new_global[:, :, 0] = x
    new_global[:, :, 1] = y
    new_global[:, :, 3] = vx2
    new_global[:, :, 4] = vy2
    return new_global


def generate_exec_ego_41(ref: np.ndarray, scen: Scenario, seed: int) -> np.ndarray:
    """Return executed anchors in ego frame, shape (256,41,3)."""
    exec_global, initial_state, _ = _simulate_exec_global(ref, scen, seed)
    return _global_to_ego_xyyaw_all(exec_global, initial_state)


def generate_exec_ego_41_xy_mode(ref: np.ndarray, scen: Scenario, seed: int, *, xy_mode: str) -> np.ndarray:
    """Return executed anchors in ego frame, with XY source controlled by xy_mode."""
    exec_global, initial_state, _ = _simulate_exec_global(ref, scen, seed)
    if xy_mode == "post":
        dt = float(TrajectorySampling(time_horizon=4.0, interval_length=0.1).interval_length)
        exec_global = _rebuild_xy_from_post_yaw_speed(exec_global, dt=dt)
    return _global_to_ego_xyyaw_all(exec_global, initial_state)


def generate_exec_ego_41_with_speed(ref: np.ndarray, scen: Scenario, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyyaw_ego, speed_ego) where speed comes from simulated vx/vy (post_transform affects it)."""
    exec_global, initial_state, _ = _simulate_exec_global(ref, scen, seed)
    xyyaw = _global_to_ego_xyyaw_all(exec_global, initial_state)
    vxy = _global_to_ego_vxy_all(exec_global, initial_state)
    speed = np.sqrt(vxy[..., 0] ** 2 + vxy[..., 1] ** 2).astype(np.float32)
    return xyyaw, speed


def _plot_yaw_speed_quantiles(
    red_xyyaw: np.ndarray,
    blue_xyyaw: np.ndarray,
    *,
    red_speed: np.ndarray | None,
    blue_speed: np.ndarray | None,
    title: str,
    out_png: Path,
    max_trajs: int,
    dpi: int,
) -> None:
    """Save a compact plot showing yaw(t) and speed(t) quantile bands (q10-q90) + mean."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n = min(max_trajs, red_xyyaw.shape[0], blue_xyyaw.shape[0])
    red = red_xyyaw[:n]
    blue = blue_xyyaw[:n]

    t = np.arange(blue.shape[1], dtype=float)

    def band(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.nanmean(x, axis=0)
        lo = np.nanquantile(x, 0.1, axis=0)
        hi = np.nanquantile(x, 0.9, axis=0)
        return mean, lo, hi

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    # yaw(t): unwrap per trajectory to avoid wrap artifacts
    red_yaw = np.unwrap(red[:, :, 2], axis=1)
    blue_yaw = np.unwrap(blue[:, :, 2], axis=1)
    r_mean, r_lo, r_hi = band(red_yaw)
    b_mean, b_lo, b_hi = band(blue_yaw)
    ax1.fill_between(t, r_lo, r_hi, color="tab:red", alpha=0.18)
    ax1.plot(t, r_mean, color="tab:red", linewidth=1.8, label="red yaw")
    ax1.fill_between(t, b_lo, b_hi, color="tab:blue", alpha=0.18)
    ax1.plot(t, b_mean, color="tab:blue", linewidth=1.8, label="blue yaw")
    ax1.set_title("Yaw(t) mean + q10-q90")
    ax1.set_xlabel("t index")
    ax1.set_ylabel("yaw (rad)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    # speed(t)
    if red_speed is not None and blue_speed is not None:
        red_s = red_speed[:n]
        blue_s = blue_speed[:n]
        r_mean, r_lo, r_hi = band(red_s)
        b_mean, b_lo, b_hi = band(blue_s)
        ax2.fill_between(t, r_lo, r_hi, color="tab:red", alpha=0.18)
        ax2.plot(t, r_mean, color="tab:red", linewidth=1.8, label="red speed")
        ax2.fill_between(t, b_lo, b_hi, color="tab:blue", alpha=0.18)
        ax2.plot(t, b_mean, color="tab:blue", linewidth=1.8, label="blue speed")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "speed unavailable", ha="center", va="center")
    ax2.set_title("Speed(t) mean + q10-q90")
    ax2.set_xlabel("t index")
    ax2.set_ylabel("speed (m/s)")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_overlay(ref_red: np.ndarray, exec_blue: np.ndarray, title: str, out_png: Path, *,
                 alpha_blue: float, alpha_red: float, lw_blue: float, lw_red: float, max_trajs: int, dpi: int) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n_ref = min(ref_red.shape[0], max_trajs)
    n_exec = min(exec_blue.shape[0], max_trajs)

    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")

    # Red first (background)
    for i in range(n_ref):
        plt.plot(
            ref_red[i, :, 0],
            ref_red[i, :, 1],
            color="tab:red",
            linewidth=lw_red,
            alpha=alpha_red,
        )

    # Blue on top
    for i in range(n_exec):
        plt.plot(
            exec_blue[i, :, 0],
            exec_blue[i, :, 1],
            color="tab:blue",
            linewidth=lw_blue,
            alpha=alpha_blue,
        )

    # Legend (avoid repeated labels)
    plt.plot([], [], color="tab:red", label="Anchor (ref)")
    plt.plot([], [], color="tab:blue", label="Executed (sim)")
    plt.legend()

    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()

    ref_path = Path(args.ref)
    out_dir = Path(args.out_dir)

    ref = np.load(str(ref_path))
    if ref.ndim != 3:
        raise ValueError(f"ref anchor must be 3D array, got shape={ref.shape}")
    if ref.shape[0] != 256:
        raise ValueError(f"Expected 256 anchors, got shape={ref.shape}")

    ref_anchor = ref[:, :, :3].astype(np.float32)  # keep original resolution for anchor overlay

    default_exec_red = None
    default_exec_red_speed = None
    if args.red == "default_exec":
        if args.plot_yaw_speed:
            default_exec_red, default_exec_red_speed = generate_exec_ego_41_with_speed(
                ref, DEFAULT_SCEN, seed=args.seed
            )
        else:
            default_exec_red = generate_exec_ego_41_xy_mode(
                ref, DEFAULT_SCEN, seed=args.seed, xy_mode=args.xy_mode
            )

    for scen in SCENARIOS:
        if args.plot_yaw_speed:
            exec_blue, exec_blue_speed = generate_exec_ego_41_with_speed(ref, scen, seed=args.seed)
            if args.xy_mode == "post":
                exec_blue = generate_exec_ego_41_xy_mode(ref, scen, seed=args.seed, xy_mode="post")
                if args.red == "default_exec" and default_exec_red is not None:
                    default_exec_red = generate_exec_ego_41_xy_mode(ref, DEFAULT_SCEN, seed=args.seed, xy_mode="post")
        else:
            exec_blue = generate_exec_ego_41_xy_mode(ref, scen, seed=args.seed, xy_mode=args.xy_mode)
            exec_blue_speed = None

        if args.red == "anchor":
            red = ref_anchor
            red_label = "Anchor (ref)"
        else:
            red = default_exec_red
            red_label = "Executed (default)"

        title = (
            f"{scen.tag} | red={red_label} | blue=Executed (style) "
            f"tracker={scen.tracker_style} post={scen.post_style} "
            f"h={scen.heading_scale}/{scen.heading_bias} s={scen.speed_scale}/{scen.speed_bias} noise={scen.noise_std}"
        )
        out_png = out_dir / f"{scen.tag}.png"

        plot_overlay(
            red,
            exec_blue,
            title,
            out_png,
            alpha_blue=args.alpha_blue,
            alpha_red=args.alpha_red,
            lw_blue=args.lw_blue,
            lw_red=args.lw_red,
            max_trajs=args.max_trajs,
            dpi=args.dpi,
        )

        if args.plot_yaw_speed and args.red == "default_exec":
            stats_png = out_dir / f"{scen.tag}_yaw_speed.png"
            _plot_yaw_speed_quantiles(
                red,
                exec_blue,
                red_speed=default_exec_red_speed,
                blue_speed=exec_blue_speed,
                title=title,
                out_png=stats_png,
                max_trajs=args.max_trajs,
                dpi=args.dpi,
            )

        if args.save_exec_npy:
            out_npy = out_dir / f"{scen.tag}_exec_41.npy"
            np.save(str(out_npy), exec_blue)

        print(f"[OK] {scen.tag} -> {out_png}")


if __name__ == "__main__":
    main()
