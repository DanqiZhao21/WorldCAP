import os
import sys
from pathlib import Path
import numpy as np
import random
from tqdm import trange

# ===== setup PYTHONPATH =====
ROOT = Path(__file__).resolve().parents[2]
NAVSIM_DIR = ROOT / 'navsim'
NUPLAN_DIR = ROOT / 'nuplan-devkit'
sys.path.insert(0, str(NAVSIM_DIR))
sys.path.insert(0, str(NUPLAN_DIR))

# ===== imports =====
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTracker
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array, global_traj_to_ego_all

# ========= CONFIG =========
TARGET_LEN = 8            # downsample 到 8 步
M_PER_FAMILY = 25         # 每个 family 内扰动数量
SEED = 42                 # 固定随机种子
np.random.seed(SEED)
random.seed(SEED)

# ===== LQR family prototypes =====
LQR_STYLE_PROTOTYPES = {
    "aggressive": dict(q_lat=[5,500,5], r_lat=0.01, horizon=[3,5], jerk=[0.0,1e-6], curvature_rate=[1e-2, 1e-2]),
    "conservative": dict(q_lat=[20,2000,20], r_lat=10.0, horizon=[10,20], jerk=[1e-3,1e-2], curvature_rate=[1e-2, 1e-2]),
    "precise": dict(q_lat=[50,5000,0], r_lat=0.1, horizon=[5,10], jerk=[1e-6,1e-4], curvature_rate=[1e-2, 1e-2]),
    "sluggish": dict(q_lat=[0.1,10,0], r_lat=50.0, horizon=[15,25], jerk=[0.1,1.0], curvature_rate=[1e-2, 1e-2]),
    "jittery": dict(q_lat=[1,500,20], r_lat=1e-4, horizon=[2,3], jerk=[0.0,1e-8], curvature_rate=[1e-2, 1e-2]),
    "default":  dict(
                q_lat=[1.0, 10.0, 0.0],        # 原始 lateral 权重
                r_lat=1.0,                      # 原始 lateral R
                q_longitudinal=[10.0],          # 原始 longitudinal Q
                r_longitudinal=[1.0],           # 原始 longitudinal R
                horizon=[10, 10],               # 固定 tracking_horizon = 10
                jerk=[1e-4, 1e-4],              # 固定 jerk_penalty = 1e-4
                curvature_rate=[1e-2, 1e-2],    # 固定 curvature_rate_penalty = 1e-2
                ),
}

# ===== post styles =====
POST_STYLE_BY_LQR_STYLE = {
    "aggressive":[
        dict(style="yaw_speed_extreme", heading_scale=1.2, speed_scale=1.1),
        dict(style="yaw_speed_extreme", heading_scale=1.3, speed_scale=1.2),
        dict(style="yaw_scale", heading_scale=1.1),
        dict(style="none"),
    ],
    "conservative":[
        dict(style="speed_scale", speed_scale=0.8),
        dict(style="speed_scale", speed_scale=0.9),
        dict(style="none"),
    ],
    "jittery":[
        dict(style="yaw_scale", heading_scale=1.05, noise_std=0.01),
        dict(style="yaw_scale", heading_scale=1.1, noise_std=0.02),
        dict(style="none"),
    ],
    "sluggish":[
        dict(style="speed_scale", speed_scale=0.7),
        dict(style="speed_scale", speed_scale=0.8),
        dict(style="none"),
    ],
    "precise":[
        dict(style="none"),
        dict(style="yaw_scale", heading_scale=1.0)
    ],
    "default": [
        dict(style="none")
    ],
}

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

def sample_lqr_params(style_name):
    if style_name == "default":
        proto = LQR_STYLE_PROTOTYPES["default"]
        return dict(
            style="default",
            q_longitudinal=proto["q_longitudinal"],
            r_longitudinal=proto["r_longitudinal"],
            q_lateral=proto["q_lat"],
            r_lateral=[proto["r_lat"]],
            tracking_horizon=proto["horizon"][0],
            jerk_penalty=proto["jerk"][0],
            curvature_rate_penalty=proto["curvature_rate"][0],
        )
    proto = LQR_STYLE_PROTOTYPES[style_name]
    q_lat = [q*np.random.uniform(0.5,1.5) for q in proto["q_lat"]]
    return dict(
        style=style_name,
        q_longitudinal=[float(np.random.choice([5,10,50,200]))],
        r_longitudinal=[float(np.random.choice([0.01,0.1,1]))],
        q_lateral=q_lat,
        r_lateral=[float(proto["r_lat"])],
        tracking_horizon=int(np.random.randint(*proto["horizon"])),
        jerk_penalty=float(np.random.uniform(*proto["jerk"])),
        curvature_rate_penalty=float(np.random.uniform(*proto["curvature_rate"]))
    )

def sample_post_params(style_name):
    """随机选择与LQR style匹配的 post style"""
    return random.choice(POST_STYLE_BY_LQR_STYLE[style_name]).copy()

# ===== MAIN SIMULATION =====
def main():
    anchors_path = ROOT / "ControllerExp/Anchors_Original_256_centered.npy"
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

    all_execs = []
    style_meta = []
    kind_id = 0

    for style_name in LQR_STYLE_PROTOTYPES.keys():
        for m in range(M_PER_FAMILY):
            lqr_params = sample_lqr_params(style_name)
            post_params = sample_post_params(style_name)

            # init simulator
            lqr_cfg = {k: v for k, v in lqr_params.items() if k != 'style'}
            simulator = PDMSimulator(proposal_sampling,
                                     tracker_style='default',
                                     post_style=post_params.get('style','none'),
                                     post_params=post_params,
                                     tracker_params=lqr_cfg)
            print(f"[DEBUG] style='{style_name}', post='{post_params.get('style','none')}', LQR={lqr_cfg}")

            # initial state
            initial_state = ego_from_anchor_pair(ref_anchors[0,0,:3], ref_anchors[0,1,:3])

            # 生成 proposal states
            all_states = []
            for a_idx in range(ref_anchors.shape[0]):
                rel_poses = ref_anchors[a_idx,1:,:3]  # drop t=0，只用未来40步
                traj = Trajectory(poses=rel_poses, trajectory_sampling=proposal_sampling)
                global_traj = transform_trajectory(traj, initial_state)
                traj_states = get_trajectory_as_array(global_traj, proposal_sampling, initial_state.time_point)
                all_states.append(traj_states)
            proposal_states = np.stack(all_states, axis=0)

            # simulate proposals
            exec_41 = simulator.simulate_proposals(proposal_states, initial_state)
            exec_ego = global_traj_to_ego_all(exec_41, initial_state)
            exec_8 = downsample(exec_ego)

            all_execs.append(exec_8)
            style_meta.append(dict(kind_id=kind_id, style=style_name,
                                   lqr={k:v for k,v in lqr_params.items() if k!='style'},
                                   post=post_params))
            kind_id += 1

    # stack results
    all_execs = np.stack(all_execs)  # [N_total,256,8,3]
    style_arr = np.array([m["style"] for m in style_meta], dtype=object)
    lqr_meta = np.array([m["lqr"] for m in style_meta], dtype=object)
    post_meta = np.array([m["post"] for m in style_meta], dtype=object)

    # save
    out_dir = ROOT / "ControllerExp/generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "controller_styles.npz"
    np.savez(out_path,
             exec_trajs=all_execs,
             ref_traj=downsample(ref_anchors),
             style_names=style_arr,
             lqr_params=lqr_meta,
             post_params=post_meta)
    print(f"✅ Generated {len(all_execs)} exec_trajs with metadata saved!")



def main_debug():
    anchors_path = ROOT / "ControllerExp/Anchors_Original_256_centered.npy"
    ref_anchors = np.load(str(anchors_path))
    print(f"[INFO] loaded anchors from {anchors_path}, shape={ref_anchors.shape}")
    if ref_anchors.shape[1] != 41:
        print(f"[WARN] anchors T={ref_anchors.shape[1]} != 41, resampling to 41...")
        ref_anchors = resample_anchors_to_41(ref_anchors)
        print(f"[INFO] anchors resampled to {ref_anchors.shape}")

    proposal_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    # proposal_sampling = TrajectorySampling(num_poses=41, interval_length=0.1)

    # ==== 只生成一条 default+none 轨迹 ====
    style_name = "default"
    lqr_params = sample_lqr_params(style_name)

    # post style 固定为 "none"
    post_params = {"style": "none"}

    # init simulator
    simulator = PDMSimulator(proposal_sampling,
                             tracker_style='default',
                             post_style=post_params['style'],
                             post_params=post_params)
    # simulator._tracker = BatchLQRTracker(**{k:v for k,v in lqr_params.items() if k != 'style'})

    # initial state
    initial_state = ego_from_anchor_pair(ref_anchors[0,0,:3], ref_anchors[0,1,:3])

    # 生成 proposal states
    all_states = []
    print(f"[DEBUG] anchors shape: {ref_anchors.shape}")
    try:
        print(f"[DEBUG] proposal_sampling.num_poses: {proposal_sampling.num_poses}")
    except Exception as e:
        print(f"[DEBUG] cannot read num_poses from proposal_sampling: {e}")
    for a_idx in range(ref_anchors.shape[0]):
        # 使用未来40步，丢弃t=0，保证长度与sampling一致
        rel_poses = ref_anchors[a_idx,1:,:3]
        if a_idx == 0:
            print(f"[DEBUG] rel_poses[0] shape: {rel_poses.shape}")
        traj = Trajectory(poses=rel_poses, trajectory_sampling=proposal_sampling)
        global_traj = transform_trajectory(traj, initial_state)
        traj_states = get_trajectory_as_array(global_traj, proposal_sampling, initial_state.time_point)
        all_states.append(traj_states)
    proposal_states = np.stack(all_states, axis=0)
    print(f"[DEBUG] proposal_states shape: {proposal_states.shape}")

    # simulate proposals
    exec_41 = simulator.simulate_proposals(proposal_states, initial_state)
    print(f"[DEBUG] simulated exec_41 shape: {exec_41.shape}")
    exec_ego = global_traj_to_ego_all(exec_41, initial_state)
    print(f"[DEBUG] exec_ego shape: {exec_ego.shape}")
    exec_8 = downsample(exec_ego)
    print(f"[DEBUG] exec_8 shape: {exec_8.shape}")

    # 保存或者打印检查
    out_dir = ROOT / "ControllerExp/generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "debug_default_none.npy"
    np.save(out_path, exec_8)
    print(f"✅ Saved default+none trajectory: {exec_8.shape} -> {out_path}")

if __name__=="__main__":
    main()
    # main_debug()
