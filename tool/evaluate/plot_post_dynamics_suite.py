#!/usr/bin/env python
"""Generate a suite of post-dynamics tracking rollouts and plot XY overlays.

Outputs 10-15 (default: 15) figures under --out-dir.

Red: reference anchor trajectories.
Blue: executed trajectories after running PDMSimulator with post dynamics enabled.

Example:
  python tool/evaluate/plot_post_dynamics_suite.py \
    --ref ControllerExp/Anchors_Original_256_centered.npy \
    --out-dir ControllerInTheLoop/step0_validationOfSimulation/plots_post_dynamics_suite
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

# Make local packages importable without requiring a pre-set PYTHONPATH.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nuplan-devkit"))

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
class PostDynScenario:
    tag: str
    tracker_style: str = "default"
    post_style: str = "post_dynamics"
    post_params: Dict[str, Any] = field(default_factory=dict)


def scenarios() -> List[PostDynScenario]:
    # All scenarios are run in online mode to ensure closed-loop feedback.
    base: Dict[str, Any] = {"apply_mode": "online"}

    def sc(tag: str, *, tracker_style: str = "default", post_style: str = "post_dynamics", **params: Any) -> PostDynScenario:
        merged = dict(base)
        merged.update(params)
        return PostDynScenario(tag=tag, tracker_style=tracker_style, post_style=post_style, post_params=merged)

    deg = np.deg2rad

    return [
        sc("PD00_baseline_none", post_style="none"),
        sc("PD01_steer_rate_gain_090", steer_rate_gain=0.90),
        sc("PD02_steer_rate_gain_070", steer_rate_gain=0.70),
        sc("PD03_accel_gain_080", accel_gain=0.80),
        sc("PD04_accel_bias_m05", accel_bias=-0.5),
        sc("PD05_cmd_delay_1step", command_delay_steps=1),
        sc("PD06_cmd_delay_2step", command_delay_steps=2),
        sc("PD07_cmd_lpf_tau_020", command_lpf_tau=0.20),
        sc("PD08_cmd_lpf_tau_050", command_lpf_tau=0.50),
        sc("PD09_speed_dep_steer_k003", steer_gain_speed_k=0.03),
        sc("PD10_model_steer_tau_x4", steering_angle_time_constant_scale=4.0),
        sc("PD11_model_accel_tau_x3", accel_time_constant_scale=3.0),
        sc("PD12_model_wheelbase_x12", wheelbase_scale=1.2),
        sc(
            "PD13_state_yaw_bias_deg2",
            post_style="yaw_scale",
            heading_scale=1.0,
            heading_bias=float(deg(2.0)),
        ),
        sc(
            "PD14_state_speed_scale_090",
            post_style="speed_scale",
            speed_scale=0.90,
            speed_bias=0.0,
        ),
        sc(
            "PD15_combo_understeer_lag",
            steer_rate_gain=0.85,
            command_delay_steps=1,
            command_lpf_tau=0.20,
            steering_angle_time_constant_scale=3.0,
            wheelbase_scale=1.10,
        ),
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
        default="ControllerInTheLoop/step0_validationOfSimulation/plots_post_dynamics_suite",
        help="Directory to write png.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--red",
        type=str,
        default="default_exec",
        choices=["anchor", "default_exec"],
        help="What to plot in red: original anchor, or anchor executed by default controller.",
    )
    p.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include PD00_baseline_none in outputs (otherwise it is skipped).",
    )

    # Default goal (per request): red is pale but still visible; blue is on top and a bit stronger.
    # Keep alpha gap modest.
    p.add_argument("--alpha-blue", type=float, default=0.65)
    p.add_argument("--alpha-red", type=float, default=0.45)
    p.add_argument("--lw-blue", type=float, default=1.3)
    p.add_argument("--lw-red", type=float, default=1.1)
    p.add_argument("--max-trajs", type=int, default=256)
    return p.parse_args()


def _simulate_exec_global(ref_41: np.ndarray, scen: PostDynScenario, *, seed: int) -> tuple[np.ndarray, EgoState]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    proposal_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    initial_state = ego_from_anchor_pair(ref_41[0, 0, :3], ref_41[0, 1, :3], dt=0.1)

    post_params = dict(scen.post_params)
    post_params.setdefault("seed", int(rng.integers(0, 2**31 - 1)))

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
    return exec_global, initial_state


def plot_overlay(
    ref_red: np.ndarray,
    exec_blue: np.ndarray,
    title: str,
    out_png: Path,
    *,
    alpha_blue: float,
    alpha_red: float,
    lw_blue: float,
    lw_red: float,
    max_trajs: int,
    dpi: int,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n_ref = min(ref_red.shape[0], max_trajs)
    n_exec = min(exec_blue.shape[0], max_trajs)

    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")

    for i in range(n_ref):
        plt.plot(
            ref_red[i, :, 0],
            ref_red[i, :, 1],
            color="tab:red",
            linewidth=lw_red,
            alpha=alpha_red,
        )

    for i in range(n_exec):
        plt.plot(
            exec_blue[i, :, 0],
            exec_blue[i, :, 1],
            color="tab:blue",
            linewidth=lw_blue,
            alpha=alpha_blue,
        )

    plt.plot([], [], color="tab:red", label="Anchor (ref)")
    plt.plot([], [], color="tab:blue", label="Executed (post-dyn)")
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

    ref_41 = ref[:, :, :3].astype(np.float32)
    if ref_41.shape[1] != 41:
        ref_41 = resample_anchors_to_41(ref_41)

    suite = scenarios()

    # Prepare red trajectories
    if args.red == "anchor":
        red_trajs = ref_41
        red_title = "Anchor (ref)"
    else:
        # Default controller executed trajectories (no post). This is what you called "anchor default exec".
        base = PostDynScenario(tag="DEFAULT_EXEC", tracker_style="default", post_style="none", post_params={})
        exec_global, initial_state = _simulate_exec_global(ref_41, base, seed=args.seed)
        red_trajs = _global_to_ego_xyyaw_all(exec_global, initial_state)
        red_title = "Executed (default)"

    # Save scenario config for bookkeeping
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "suite_config.json"
    try:
        import json

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "tag": s.tag,
                        "tracker_style": s.tracker_style,
                        "post_style": s.post_style,
                        "post_params": s.post_params,
                    }
                    for s in suite
                ],
                f,
                indent=2,
                sort_keys=True,
            )
    except Exception:
        pass

    for scen in suite:
        if not args.include_baseline and scen.tag == "PD00_baseline_none":
            continue
        exec_global, initial_state = _simulate_exec_global(ref_41, scen, seed=args.seed)
        exec_ego = _global_to_ego_xyyaw_all(exec_global, initial_state)

        title = f"{scen.tag} | red={red_title} | blue=Executed (post-dyn)"
        out_png = out_dir / f"{scen.tag}.png"
        plot_overlay(
            red_trajs,
            exec_ego,
            title,
            out_png,
            alpha_blue=args.alpha_blue,
            alpha_red=args.alpha_red,
            lw_blue=args.lw_blue,
            lw_red=args.lw_red,
            max_trajs=args.max_trajs,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
