import sys
from pathlib import Path
import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# ===== setup PYTHONPATH =====
ROOT = Path(__file__).resolve().parents[2]
NUPLAN_DIR = ROOT / 'nuplan-devkit'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NUPLAN_DIR))

# ===== imports =====
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

# ========= CONFIG =========
TARGET_LEN = 8            # downsample 到 8 步

# Counts: keep this moderate; each style runs PDMSimulator over all 256 anchors.
N_TRACKER_PER_SUBSTYLE = 8
N_POSTDYN_PER_SUBSTYLE = 6

# Validation split: keep only 2-3 per (big, substyle) group.
# Separate quotas for tracker-level and post-dynamics-level styles.
VAL_PER_SUBSTYLE_TRK = 2
VAL_PER_SUBSTYLE_PD = 2

SEED = 42                 # 固定随机种子
np.random.seed(SEED)
random.seed(SEED)

@dataclass(frozen=True)
class StyleSpec:
    """One style entry in the controller bundle."""

    style_name: str
    tracker_params: Optional[Dict[str, Any]]
    post_style: str
    post_params: Dict[str, Any]
    kind: str  # 'trk' | 'pd'
    group_big: str
    group_sub: str


def _deg(x: float) -> float:
    return float(np.deg2rad(x))


# ===== LQR tracker prototypes (big categories + substyles) =====
TRACKER_SUBSTYLES: Dict[str, Dict[str, Dict[str, Any]]] = {
    # Each substyle provides sampling ranges around a prototype.
    "default": {
        "base": dict(
            q_longitudinal=[10.0],
            r_longitudinal=[1.0],
            q_lateral=[1.0, 10.0, 0.0],
            r_lateral=[1.0],
            tracking_horizon=10,
            jerk_penalty=1e-4,
            curvature_rate_penalty=1e-2,
        )
    },
    "aggressive": {
        "mild": dict(q_lat=[2.0, 100.0, 2.0], r_lat=0.2, horizon=[6, 10], jerk=[1e-6, 1e-4], curvature_rate=[5e-3, 2e-2]),
        "strong": dict(q_lat=[5.0, 500.0, 5.0], r_lat=0.05, horizon=[3, 6], jerk=[0.0, 1e-6], curvature_rate=[1e-2, 5e-2]),
    },
    "conservative": {
        "mild": dict(q_lat=[10.0, 800.0, 10.0], r_lat=5.0, horizon=[10, 18], jerk=[1e-3, 5e-3], curvature_rate=[5e-3, 2e-2]),
        "strong": dict(q_lat=[20.0, 2000.0, 20.0], r_lat=10.0, horizon=[14, 24], jerk=[5e-3, 2e-2], curvature_rate=[5e-3, 2e-2]),
    },
    "precise": {
        "mild": dict(q_lat=[20.0, 2000.0, 0.0], r_lat=0.5, horizon=[8, 12], jerk=[1e-6, 1e-4], curvature_rate=[5e-3, 2e-2]),
        "strong": dict(q_lat=[50.0, 5000.0, 0.0], r_lat=0.1, horizon=[5, 10], jerk=[1e-6, 1e-4], curvature_rate=[5e-3, 2e-2]),
    },
    "sluggish": {
        "mild": dict(q_lat=[0.5, 20.0, 0.0], r_lat=10.0, horizon=[12, 20], jerk=[1e-2, 1e-1], curvature_rate=[5e-3, 2e-2]),
        "strong": dict(q_lat=[0.1, 10.0, 0.0], r_lat=50.0, horizon=[18, 28], jerk=[0.05, 0.5], curvature_rate=[5e-3, 2e-2]),
    },
    "jittery": {
        "mild": dict(q_lat=[1.0, 200.0, 10.0], r_lat=1e-2, horizon=[4, 6], jerk=[0.0, 1e-6], curvature_rate=[5e-3, 2e-2]),
        "strong": dict(q_lat=[1.0, 500.0, 20.0], r_lat=1e-4, horizon=[2, 4], jerk=[0.0, 1e-8], curvature_rate=[1e-2, 5e-2]),
    },
}


def _sample_tracker_params(family: str, substyle: str) -> Dict[str, Any]:
    """Return tracker_params dict accepted by PDMSimulator(tracker_params=...)."""
    if family == "default":
        return dict(TRACKER_SUBSTYLES["default"]["base"])

    cfg = TRACKER_SUBSTYLES[family][substyle]

    q_lat = [float(q * np.random.uniform(0.7, 1.3)) for q in cfg["q_lat"]]
    r_lat = float(cfg["r_lat"] * np.random.uniform(0.7, 1.3))
    horizon = int(np.random.randint(cfg["horizon"][0], cfg["horizon"][1] + 1))
    jerk_penalty = float(np.random.uniform(cfg["jerk"][0], cfg["jerk"][1]))
    curvature_rate_penalty = float(np.random.uniform(cfg["curvature_rate"][0], cfg["curvature_rate"][1]))

    # Longitudinal weights: keep within a reasonable discrete set
    q_long = float(np.random.choice([5.0, 10.0, 50.0, 200.0]))
    r_long = float(np.random.choice([0.01, 0.1, 1.0]))

    return dict(
        q_longitudinal=[q_long],
        r_longitudinal=[r_long],
        q_lateral=q_lat,
        r_lateral=[r_lat],
        tracking_horizon=horizon,
        jerk_penalty=jerk_penalty,
        curvature_rate_penalty=curvature_rate_penalty,
    )


def _normalize_post_params(post_params: Dict[str, Any]) -> Dict[str, Any]:
    """Fill defaults so metadata is uniform."""
    out = dict(post_params)
    out.setdefault("apply_mode", "online")
    out.setdefault("heading_scale", 1.0)
    out.setdefault("heading_bias", 0.0)
    out.setdefault("speed_scale", 1.0)
    out.setdefault("speed_bias", 0.0)
    out.setdefault("noise_std", 0.0)
    return out


# ===== post-dynamics (big categories + substyles) =====
POSTDYN_SUBSTYLES: Dict[str, Dict[str, Dict[str, Any]]] = {
    # Command/model-level dynamics mismatch (recommended; decoupled from tracker params).
    "actuator_steer_gain": {
        "mild": dict(post_style="post_dynamics", steer_rate_gain=(0.92, 0.98)),
        "strong": dict(post_style="post_dynamics", steer_rate_gain=(0.80, 0.92)),
    },
    "actuator_accel_gain": {
        "mild": dict(post_style="post_dynamics", accel_gain=(0.90, 0.98)),
        "strong": dict(post_style="post_dynamics", accel_gain=(0.80, 0.90)),
    },
    "actuator_delay": {
        "1step": dict(post_style="post_dynamics", command_delay_steps=1),
        "2step": dict(post_style="post_dynamics", command_delay_steps=2),
    },
    "actuator_lpf": {
        "tau015": dict(post_style="post_dynamics", command_lpf_tau=(0.12, 0.18)),
        "tau030": dict(post_style="post_dynamics", command_lpf_tau=(0.25, 0.35)),
    },
    "speed_dep_understeer": {
        "k0015": dict(post_style="post_dynamics", steer_gain_speed_k=(0.012, 0.018)),
        "k0030": dict(post_style="post_dynamics", steer_gain_speed_k=(0.025, 0.035)),
    },
    "model_wheelbase": {
        "mild": dict(post_style="post_dynamics", wheelbase_scale=(1.03, 1.07)),
        "strong": dict(post_style="post_dynamics", wheelbase_scale=(1.07, 1.12)),
    },
    "model_steer_lag": {
        "mild": dict(post_style="post_dynamics", steering_angle_time_constant_scale=(1.8, 2.6)),
        "strong": dict(post_style="post_dynamics", steering_angle_time_constant_scale=(2.6, 4.0)),
    },
    "model_accel_lag": {
        "mild": dict(post_style="post_dynamics", accel_time_constant_scale=(1.6, 2.4)),
        "strong": dict(post_style="post_dynamics", accel_time_constant_scale=(2.4, 3.5)),
    },
    # Small state-estimation-like biases (keep reasonable; avoid 1.5x extremes)
    "state_yaw_bias": {
        "pos": dict(post_style="yaw_scale", heading_bias=_deg(1.5)),
        "neg": dict(post_style="yaw_scale", heading_bias=-_deg(1.5)),
    },
    "state_speed_scale": {
        "slow": dict(post_style="speed_scale", speed_scale=(0.92, 0.98)),
        "fast": dict(post_style="speed_scale", speed_scale=(1.02, 1.08)),
    },
    "state_noise": {
        "mild": dict(post_style="gaussian_noise", noise_std=(0.005, 0.015)),
        "strong": dict(post_style="gaussian_noise", noise_std=(0.015, 0.03)),
    },
}


def _sample_from_range(v: Any) -> Any:
    """If v is a (lo,hi) tuple -> sample uniform; else pass through."""
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float, np.floating)) for x in v):
        lo, hi = float(v[0]), float(v[1])
        return float(np.random.uniform(lo, hi))
    return v


def _sample_postdyn_params(big: str, sub: str) -> tuple[str, Dict[str, Any]]:
    cfg = dict(POSTDYN_SUBSTYLES[big][sub])
    post_style = str(cfg.pop("post_style"))
    post_params: Dict[str, Any] = {"style": post_style, "apply_mode": "online"}
    for k, v in cfg.items():
        post_params[k] = _sample_from_range(v)
    return post_style, _normalize_post_params(post_params)

# ===== UTIL FUNCTIONS =====
def downsample(traj):
    """traj: [N, T, D] -> [N, TARGET_LEN, 3]"""
    T = traj.shape[1]
    idx = np.linspace(0, T-1, TARGET_LEN, dtype=int)
    return traj[:, idx, :3]

def ego_from_anchor_pair(p0, p1, dt=0.1):
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

# ===== resampling helpers =====
def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def _resample_traj_xyz(xy_yaw: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a single trajectory [T,3] to [target_len,3] with linear interp on x,y and unwrap+interp+wrap for yaw.
    """
    T = xy_yaw.shape[0]
    t_old = np.arange(T, dtype=float)
    t_new = np.linspace(0, T - 1, target_len, dtype=float)
    x_old, y_old, yaw_old = xy_yaw[:, 0], xy_yaw[:, 1], xy_yaw[:, 2]
    x_new = np.interp(t_new, t_old, x_old)
    y_new = np.interp(t_new, t_old, y_old)
    yaw_unwrap = np.unwrap(yaw_old)
    yaw_new = np.interp(t_new, t_old, yaw_unwrap)
    yaw_new = _wrap_angle(yaw_new)
    return np.stack([x_new, y_new, yaw_new], axis=-1)

def resample_anchors_to_41(anchors: np.ndarray) -> np.ndarray:
    """Resample anchors [256,T,3] to [256,41,3]."""
    B, T, D = anchors.shape
    assert D >= 3, "Anchors last dim must be >=3 (x,y,yaw)"
    out = np.zeros((B, 41, 3), dtype=anchors.dtype)
    for i in range(B):
        out[i] = _resample_traj_xyz(anchors[i, :, :3], 41)
    return out

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


def _build_global_proposal_states(ref_anchors_41: np.ndarray, initial_ego_state: EgoState) -> np.ndarray:
    """Build (256,41,11) global proposal state array from ego-frame anchors (256,41,3)."""
    x0 = float(initial_ego_state.rear_axle.x)
    y0 = float(initial_ego_state.rear_axle.y)
    yaw0 = float(initial_ego_state.rear_axle.heading)
    cos0, sin0 = np.cos(yaw0), np.sin(yaw0)

    B, T, _ = ref_anchors_41.shape
    out = np.zeros((B, T, 11), dtype=np.float64)

    xe = ref_anchors_41[:, :, 0]
    ye = ref_anchors_41[:, :, 1]
    yawe = ref_anchors_41[:, :, 2]

    out[:, :, 0] = x0 + xe * cos0 - ye * sin0
    out[:, :, 1] = y0 + xe * sin0 + ye * cos0
    out[:, :, 2] = yaw0 + yawe
    return out


def build_style_specs() -> List[StyleSpec]:
    """Construct a comprehensive style list.

    Design principles:
    - Index 0 MUST be default/no-post for stable eval behavior.
    - Tracker-level styles vary tracker_params with post_style='none'.
    - Post-dynamics styles use DEFAULT tracker (tracker_params=None) to decouple effects.
    - Post-dynamics magnitudes are mild/realistic (avoid 1.5x extremes).
    """
    specs: List[StyleSpec] = []

    default_tracker_params = dict(TRACKER_SUBSTYLES["default"]["base"])

    # 0) default / none (keep first)
    specs.append(
        StyleSpec(
            style_name="trk_default_base",
            tracker_params=dict(default_tracker_params),
            post_style="none",
            post_params=_normalize_post_params({"style": "none", "apply_mode": "off"}),
            kind="trk",
            group_big="default",
            group_sub="base",
        )
    )

    # 1) tracker-level families (big class + substyles)
    for family in ["aggressive", "conservative", "precise", "sluggish", "jittery"]:
        for substyle in TRACKER_SUBSTYLES[family].keys():
            for k in range(N_TRACKER_PER_SUBSTYLE):
                tracker_params = _sample_tracker_params(family, substyle)
                specs.append(
                    StyleSpec(
                        style_name=f"trk_{family}_{substyle}_{k:02d}",
                        tracker_params=tracker_params,
                        post_style="none",
                        post_params=_normalize_post_params({"style": "none", "apply_mode": "off"}),
                        kind="trk",
                        group_big=str(family),
                        group_sub=str(substyle),
                    )
                )

    # 2) post-dynamics families (big class + substyles)
    for big in [
        "actuator_steer_gain",
        "actuator_accel_gain",
        "actuator_delay",
        "actuator_lpf",
        "speed_dep_understeer",
        "model_wheelbase",
        "model_steer_lag",
        "model_accel_lag",
        "state_yaw_bias",
        "state_speed_scale",
        "state_noise",
    ]:
        for sub in POSTDYN_SUBSTYLES[big].keys():
            for k in range(N_POSTDYN_PER_SUBSTYLE):
                post_style, post_params = _sample_postdyn_params(big, sub)
                # for deterministic substyles (delay steps, yaw bias), k>0 may duplicate; add tiny noise seed anyway.
                post_params.setdefault("seed", int(np.random.randint(0, 2**31 - 1)))
                specs.append(
                    StyleSpec(
                        style_name=f"pd_{big}_{sub}_{k:02d}",
                        tracker_params=None,  # decouple: default tracker
                        post_style=post_style,
                        post_params=post_params,
                        kind="pd",
                        group_big=str(big),
                        group_sub=str(sub),
                    )
                )

    return specs


def split_train_val(specs: List[StyleSpec]) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic split: for each (kind,big,sub) group, take first K as val.

    Notes:
        - K depends on kind: tracker -> VAL_PER_SUBSTYLE_TRK, post-dyn -> VAL_PER_SUBSTYLE_PD.
        - Style index 0 ('trk_default_base') is kept in train for stability.
    """
    groups: Dict[tuple[str, str, str], List[int]] = {}
    for idx, s in enumerate(specs):
        key = (s.kind, s.group_big, s.group_sub)
        groups.setdefault(key, []).append(idx)

    val_indices: List[int] = []
    for _, idxs in sorted(groups.items(), key=lambda kv: kv[0]):
        # Keep idx0 in train.
        idxs2 = [i for i in idxs if i != 0]

        kind = ""
        try:
            kind = specs[idxs[0]].kind
        except Exception:
            kind = ""

        if kind == "trk":
            k = int(VAL_PER_SUBSTYLE_TRK)
        elif kind == "pd":
            k = int(VAL_PER_SUBSTYLE_PD)
        else:
            k = int(VAL_PER_SUBSTYLE_PD)

        take = min(k, len(idxs2))
        val_indices.extend(idxs2[:take])

    val_set = set(val_indices)
    train_indices = [i for i in range(len(specs)) if i not in val_set]
    return np.array(train_indices, dtype=np.int64), np.array(sorted(val_indices), dtype=np.int64)

# ===== MAIN SIMULATION =====
def main():
    anchors_path = ROOT / "ControllerExp/Anchors_Original_1024_centered.npy"
    ref_anchors = np.load(str(anchors_path))
    print(f"[INFO] loaded anchors from {anchors_path}, shape={ref_anchors.shape}")
    # normalize anchors to 41 if needed
    if ref_anchors.shape[1] != 41:
        print(f"[WARN] anchors T={ref_anchors.shape[1]} != 41, resampling to 41...")
        ref_anchors = resample_anchors_to_41(ref_anchors)
        print(f"[INFO] anchors resampled to {ref_anchors.shape}")

    # Use 4.0s horizon @ 0.1s -> num_poses=40; anchors包含t=0，因此轨迹取未来40步
    proposal_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    
    # num_poses = ref_anchors.shape[1]
    # proposal_sampling = TrajectorySampling(num_poses=num_poses, interval_length=0.1)

    # initial state (shared)
    initial_state = ego_from_anchor_pair(ref_anchors[0, 0, :3], ref_anchors[0, 1, :3])

    # Precompute global proposal reference states once (major speedup)
    proposal_states = _build_global_proposal_states(ref_anchors[:, :, :3].astype(np.float32), initial_state)
    print(f"[INFO] proposal_states built once: shape={proposal_states.shape}")

    specs = build_style_specs()
    print(f"[INFO] will generate {len(specs)} styles (index0={specs[0].style_name})")

    train_style_indices, val_style_indices = split_train_val(specs)
    print(
        f"[INFO] split: train_styles={len(train_style_indices)} val_styles={len(val_style_indices)} "
        f"(VAL_PER_SUBSTYLE_TRK={VAL_PER_SUBSTYLE_TRK} VAL_PER_SUBSTYLE_PD={VAL_PER_SUBSTYLE_PD})"
    )

    all_execs: List[np.ndarray] = []
    style_meta: List[Dict[str, Any]] = []

    for kind_id, spec in enumerate(tqdm(specs, desc="Generating controller bundle")):
        # init simulator
        simulator = PDMSimulator(
            proposal_sampling,
            tracker_style='default',
            post_style=spec.post_style,
            post_params=spec.post_params,
            tracker_params=spec.tracker_params,
        )

        exec_41 = simulator.simulate_proposals(proposal_states, initial_state)
        exec_ego = _global_to_ego_xyyaw_all(exec_41, initial_state)
        exec_8 = downsample(exec_ego)

        all_execs.append(exec_8)
        style_meta.append(
            dict(
                kind_id=int(kind_id),
                style=str(spec.style_name),
                lqr=(spec.tracker_params or dict(TRACKER_SUBSTYLES["default"]["base"])),
                post=spec.post_params,
                post_style=str(spec.post_style),
            )
        )

    # stack results
    all_execs = np.stack(all_execs)  # [S,256,8,3]
    style_arr = np.array([m["style"] for m in style_meta], dtype=object)
    lqr_meta = np.array([m["lqr"] for m in style_meta], dtype=object)
    post_meta = np.array([m["post"] for m in style_meta], dtype=object)
    kind_arr = np.array([s.kind for s in specs], dtype=object)
    group_big_arr = np.array([s.group_big for s in specs], dtype=object)
    group_sub_arr = np.array([s.group_sub for s in specs], dtype=object)

    # save
    out_dir = ROOT / "ControllerExp/generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "1024" / "controller_styles_1024.npz"
    np.savez(out_path,
             exec_trajs=all_execs,
             ref_traj=downsample(ref_anchors),
             style_names=style_arr,
             lqr_params=lqr_meta,
             post_params=post_meta,
             style_kind=kind_arr,
             style_group_big=group_big_arr,
             style_group_sub=group_sub_arr,
             train_style_indices=train_style_indices,
             val_style_indices=val_style_indices,
             seed=np.array([SEED], dtype=np.int64),
             version=np.array(["v2_tracker_and_postdyn_decoupled"], dtype=object),
             )
    print(f"✅ Generated bundle: styles={len(all_execs)} -> {out_path}")



def main_debug():
    anchors_path = ROOT / "ControllerExp/Anchors_Original_64_centered.npy"
    ref_anchors = np.load(str(anchors_path))
    print(f"[INFO] loaded anchors from {anchors_path}, shape={ref_anchors.shape}")
    if ref_anchors.shape[1] != 41:
        print(f"[WARN] anchors T={ref_anchors.shape[1]} != 41, resampling to 41...")
        ref_anchors = resample_anchors_to_41(ref_anchors)
        print(f"[INFO] anchors resampled to {ref_anchors.shape}")

    proposal_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    # proposal_sampling = TrajectorySampling(num_poses=41, interval_length=0.1)

    # ==== 只生成一条 default+none 轨迹 ====
    simulator = PDMSimulator(
        proposal_sampling,
        tracker_style='default',
        post_style='none',
        post_params={"style": "none", "apply_mode": "off"},
        tracker_params=dict(TRACKER_SUBSTYLES["default"]["base"]),
    )

    initial_state = ego_from_anchor_pair(ref_anchors[0, 0, :3], ref_anchors[0, 1, :3])
    proposal_states = _build_global_proposal_states(ref_anchors[:, :, :3].astype(np.float32), initial_state)
    print(f"[DEBUG] proposal_states shape: {proposal_states.shape}")

    exec_41 = simulator.simulate_proposals(proposal_states, initial_state)
    print(f"[DEBUG] simulated exec_41 shape: {exec_41.shape}")
    exec_ego = _global_to_ego_xyyaw_all(exec_41, initial_state)
    print(f"[DEBUG] exec_ego shape: {exec_ego.shape}")
    exec_8 = downsample(exec_ego)
    print(f"[DEBUG] exec_8 shape: {exec_8.shape}")

    # 保存或者打印检查
    out_dir = ROOT / "ControllerExp/generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "debug_default_none1024.npy"
    np.save(out_path, exec_8)
    print(f"✅ Saved default+none trajectory: {exec_8.shape} -> {out_path}")

if __name__=="__main__":
    main()
    # main_debug()
