# import os
# import sys
# from pathlib import Path
# import numpy as np
# import json
# from tqdm import trange

# # ===== setup PYTHONPATH for in-repo packages =====
# ROOT = Path(__file__).resolve().parents[2]
# NAVSIM_DIR = ROOT / 'navsim'
# NUPLAN_DIR = ROOT / 'nuplan-devkit'
# if str(NAVSIM_DIR) not in sys.path:
#     sys.path.insert(0, str(NAVSIM_DIR))
# if str(NUPLAN_DIR) not in sys.path:
#     sys.path.insert(0, str(NUPLAN_DIR))

# # Correct imports based on repo structure
# from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
# from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTracker
# from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
# from nuplan.common.actor_state.ego_state import EgoState
# from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
# from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters


# # ========= config =========
# N_KIND = 128          # 你最终要的 controller style 数量
# TARGET_LEN = 8        # downsample 到 8 步
# # ==========================


# def downsample(traj_41):
#     """ traj_41: [256, 41, D] -> [256, 8, 3] """
#     T = traj_41.shape[1]
#     idx = np.linspace(0, T - 1, TARGET_LEN, dtype=int)
#     return traj_41[:, idx, :3]


# LQR_STYLE_PROTOTYPES = {
#     "aggressive": dict(
#         q_lat=[5, 500, 5],
#         r_lat=0.01,
#         horizon=[3, 5],
#         jerk=[0.0, 1e-6],
#     ),
#     "conservative": dict(
#         q_lat=[20, 2000, 20],
#         r_lat=10.0,
#         horizon=[10, 20],
#         jerk=[1e-3, 1e-2],
#     ),
#     "precise": dict(
#         q_lat=[50, 5000, 0],
#         r_lat=0.1,
#         horizon=[5, 10],
#         jerk=[1e-6, 1e-4],
#     ),
#     "sluggish": dict(
#         q_lat=[0.1, 10, 0],
#         r_lat=50.0,
#         horizon=[15, 25],
#         jerk=[0.1, 1.0],
#     ),
#     "jittery": dict(
#         q_lat=[1, 500, 20],
#         r_lat=1e-4,
#         horizon=[2, 3],
#         jerk=[0.0, 1e-8],
#     ),
# }

    
# def sample_lqr_params(style_name: str) -> dict:
#     proto = LQR_STYLE_PROTOTYPES[style_name]

#     q_lat = [
#         float(proto["q_lat"][0]) * float(np.random.uniform(0.5, 1.5)),
#         float(proto["q_lat"][1]) * float(np.random.uniform(0.5, 1.5)),
#         float(proto["q_lat"][2]) * float(np.random.uniform(0.5, 1.5)),
#     ]

#     params = dict(
#         q_longitudinal=[float(np.random.choice([5, 50, 200]))],
#         r_longitudinal=[float(np.random.choice([0.01, 0.1, 1]))],
#         q_lateral=q_lat,
#         r_lateral=[float(proto["r_lat"])],
#         tracking_horizon=int(np.random.randint(*proto["horizon"])),
#         jerk_penalty=float(np.random.uniform(*proto["jerk"])),
#         curvature_rate_penalty=float(np.random.uniform(*proto["jerk"]))
#     )
#     params["style"] = style_name
#     return params



# POST_STYLE_BY_LQR_STYLE = {
#     "aggressive": [
#         dict(style="yaw_speed_extreme", heading_scale=1.2, speed_scale=1.1),
#     ],
#     "conservative": [
#         dict(style="speed_scale", speed_scale=0.8),
#     ],
#     "jittery": [
#         dict(style="yaw_scale", heading_scale=1.1, noise_std=0.02),
#     ],
#     "sluggish": [
#         dict(style="speed_scale", speed_scale=0.7),
#     ],
#     "precise": [
#         dict(style="none"),
#     ],
# }



# def ego_from_state_array(arr: np.ndarray) -> EgoState:
#     """Adapt old-style `from_state_array` by constructing the 9-dim vector required by nuplan's EgoState.deserialize.
#     Input arr is expected as [x, y, yaw, ...]. Velocity/acc set to 0; steering=0; timestamp=0.
#     """
#     x = float(arr[0]) if len(arr) > 0 else 0.0
#     y = float(arr[1]) if len(arr) > 1 else 0.0
#     yaw = float(arr[2]) if len(arr) > 2 else 0.0
#     vec9 = [0, x, y, yaw, 0.0, 0.0, 0.0, 0.0, 0.0]
#     vehicle = get_pacifica_parameters()
#     return EgoState.deserialize(vec9, vehicle)


# def main():
#     # ===== load ONE fixed anchor set =====
#     anchors_path = ROOT / "ControllerExp/Anchors_Original_256_centered.npy"
#     ref_anchors = np.load(str(anchors_path))  # [256, 41, D]

#     proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)

#     all_execs = []
#     style_meta = []

#     lqr_style_names = list(LQR_STYLE_PROTOTYPES.keys())

#     for k in trange(N_KIND, desc="Generating controller kinds"):
#         style_name = np.random.choice(lqr_style_names)
#         lqr_params = sample_lqr_params(style_name)
#         post_cfg = dict(np.random.choice(POST_STYLE_BY_LQR_STYLE[style_name]))

#         simulator = PDMSimulator(
#             proposal_sampling,
#             tracker_style='default',
#             post_style=post_cfg.get('style', 'none'),
#             post_params=post_cfg
#         )
#         simulator._tracker = BatchLQRTracker(**{k:v for k,v in lqr_params.items() if k != 'style'})

#         initial_state = ego_from_state_array(ref_anchors[0, 0])
#         exec_traj_41 = simulator.simulate_proposals(
#             ref_anchors, initial_state
#         )  # [256, 41, 11]

#         exec_traj_8 = downsample(exec_traj_41)
#         all_execs.append(exec_traj_8)

#         style_meta.append({
#             "kind_id": k,
#             "style": style_name,
#             "lqr": {k:v for k,v in lqr_params.items() if k != 'style'},
#             "post": post_cfg,
#         })

#     all_execs = np.stack(all_execs)  # [N_KIND, 256, 8, 3]

#     # object array for metadata
#     lqr_meta = np.empty(N_KIND, dtype=object)
#     post_meta = np.empty(N_KIND, dtype=object)
#     style_arr = np.empty(N_KIND, dtype=object)

#     for i, meta in enumerate(style_meta):
#         style_arr[i] = meta["style"]
#         lqr_meta[i] = meta["lqr"]
#         post_meta[i] = meta["post"]

#     out_dir = ROOT / "ControllerExp/generated"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / "controller_styles.npz"

#     np.savez(
#         str(out_path),
#         exec_trajs=all_execs,          # [N_KIND, 256, 8, 3]
#         ref_traj=downsample(ref_anchors),  # [256, 8, 3]
#         style_names=style_arr,          # object
#         lqr_params=lqr_meta,            # object
#         post_params=post_meta           # object
#     )

#     print("✅ Controller styles with metadata saved!")

# if __name__ == "__main__":
#     main()



import os
import sys
from pathlib import Path
import numpy as np
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
TARGET_LEN = 8            # downsample到8步
M_PER_FAMILY = 10         # 每个family生成多少条内部扰动样本
SEED = 42                 # 可复现
np.random.seed(SEED)

# LQR family prototypes
LQR_STYLE_PROTOTYPES = {
    "aggressive": dict(q_lat=[5,500,5], r_lat=0.01, horizon=[3,5], jerk=[0.0,1e-6]),
    "conservative": dict(q_lat=[20,2000,20], r_lat=10.0, horizon=[10,20], jerk=[1e-3,1e-2]),
    "precise": dict(q_lat=[50,5000,0], r_lat=0.1, horizon=[5,10], jerk=[1e-6,1e-4]),
    "sluggish": dict(q_lat=[0.1,10,0], r_lat=50.0, horizon=[15,25], jerk=[0.1,1.0]),
    "jittery": dict(q_lat=[1,500,20], r_lat=1e-4, horizon=[2,3], jerk=[0.0,1e-8]),
}

# 对应post style
POST_STYLE_BY_LQR_STYLE = {
    "aggressive": [
        dict(style="yaw_speed_extreme", heading_scale=1.2, speed_scale=1.1),
        dict(style="yaw_speed_extreme", heading_scale=1.3, speed_scale=1.2),
        dict(style="yaw_scale", heading_scale=1.1),
        dict(style="none"),
    ],
    "conservative": [
        dict(style="speed_scale", speed_scale=0.8),
        dict(style="speed_scale", speed_scale=0.9),
        dict(style="none"),
    ],
    "jittery": [
        dict(style="yaw_scale", heading_scale=1.05, noise_std=0.01),
        dict(style="yaw_scale", heading_scale=1.1, noise_std=0.02),
        dict(style="none"),
    ],
    "sluggish": [
        dict(style="speed_scale", speed_scale=0.7),
        dict(style="speed_scale", speed_scale=0.8),
        dict(style="none"),
    ],
    "precise": [
        dict(style="none"),
        dict(style="yaw_scale", heading_scale=1.0)
    ],
}


# ========= UTIL FUNCTIONS =========
def downsample(traj_41):
    """traj_41: [256, 41, D] -> [256, 8, 3]"""
    T = traj_41.shape[1]
    idx = np.linspace(0, T-1, TARGET_LEN, dtype=int)
    return traj_41[:, idx, :3]

def ego_from_anchor_pair(p0, p1, dt=0.1):
    """Construct EgoState using first two anchor poses to estimate initial velocity/yaw_rate.
    p0, p1: [x, y, yaw] in ego-frame; returns EgoState.deserialize vector with vx, vy, yaw_rate.
    """
    x0, y0, yaw0 = float(p0[0]), float(p0[1]), float(p0[2])
    x1, y1, yaw1 = float(p1[0]), float(p1[1]), float(p1[2])
    dx, dy, dyaw = (x1 - x0), (y1 - y0), (yaw1 - yaw0)
    vx_ego, vy_ego = dx / dt, dy / dt
    # rotate ego velocities into global using yaw0
    cos0, sin0 = np.cos(yaw0), np.sin(yaw0)
    vx = vx_ego * cos0 - vy_ego * sin0
    vy = vx_ego * sin0 + vy_ego * cos0
    yaw_rate = dyaw / dt
    vec9 = [0, x0, y0, yaw0, vx, vy, 0.0, 0.0, yaw_rate]
    return EgoState.deserialize(vec9, get_pacifica_parameters())

def sample_lqr_params(style_name):
    proto = LQR_STYLE_PROTOTYPES[style_name]
    q_lat = [q*np.random.uniform(0.5,1.5) for q in proto["q_lat"]]
    return dict(
        style=style_name,
        q_longitudinal=[float(np.random.choice([5,50,200]))],
        r_longitudinal=[float(np.random.choice([0.01,0.1,1]))],
        q_lateral=q_lat,
        r_lateral=[float(proto["r_lat"])],
        tracking_horizon=int(np.random.randint(*proto["horizon"])),
        jerk_penalty=float(np.random.uniform(*proto["jerk"])),
        curvature_rate_penalty=float(np.random.uniform(*proto["jerk"]))
    )
    
import random
def sample_post_params(style_name):
    post_choices = POST_STYLE_BY_LQR_STYLE[style_name]
    return random.choice(post_choices).copy()

# ========= MAIN =========
def main():
    anchors_path = ROOT / "ControllerExp/Anchors_Original_256_centered.npy"
    ref_anchors = np.load(str(anchors_path))  # [256,41,D]
    # Use future 40 poses (drop t=0), sampling 4.0s@0.1s to match PDMSimulator's num_poses semantics
    proposal_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1)

    all_execs = []
    style_meta = []

    kind_id = 0
    for style_name in LQR_STYLE_PROTOTYPES.keys():
        for m in range(M_PER_FAMILY):
            # 1. sample LQR + post
            lqr_params = sample_lqr_params(style_name)
            post_params = sample_post_params(style_name)

            # 2. simulate
            simulator = PDMSimulator(proposal_sampling,
                                     tracker_style='default',
                                     post_style=post_params.get('style','none'),
                                     post_params=post_params)
            simulator._tracker = BatchLQRTracker(**{k:v for k,v in lqr_params.items() if k != 'style'})

            # initial EgoState with estimated velocity/yaw_rate from first two anchor steps
            initial_state = ego_from_anchor_pair(ref_anchors[0,0,:3], ref_anchors[0,1,:3])
            # Build global proposal states [256, 41, 11] from relative anchors (use future 40 poses)
            all_states = []
            for a_idx in range(ref_anchors.shape[0]):
                rel_poses_future = ref_anchors[a_idx, 1:, :3]  # drop t=0
                traj = Trajectory(poses=rel_poses_future, trajectory_sampling=proposal_sampling)
                anchor_traj = transform_trajectory(traj, initial_state)
                anchor_states = get_trajectory_as_array(anchor_traj, proposal_sampling, initial_state.time_point)
                all_states.append(anchor_states)
            proposal_states = np.stack(all_states, axis=0)

            # Simulate proposals and convert to ego-frame offsets
            exec_41 = simulator.simulate_proposals(proposal_states, initial_state)
            exec_41_ego = global_traj_to_ego_all(exec_41, initial_state)
            exec_8 = downsample(exec_41_ego)

            all_execs.append(exec_8)
            style_meta.append(dict(kind_id=kind_id, style=style_name, lqr={k:v for k,v in lqr_params.items() if k!='style'}, post=post_params))
            kind_id += 1

    # stack
    all_execs = np.stack(all_execs)  # [N_total,256,8,3]

    # metadata
    style_arr = np.array([m["style"] for m in style_meta], dtype=object)
    lqr_meta = np.array([m["lqr"] for m in style_meta], dtype=object)
    post_meta = np.array([m["post"] for m in style_meta], dtype=object)

    # save
    out_dir = ROOT / "ControllerExp/generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "controller_styles.npz"
    np.savez(out_path, exec_trajs=all_execs,
             ref_traj=downsample(ref_anchors),
             style_names=style_arr,
             lqr_params=lqr_meta,
             post_params=post_meta)
    print(f"✅ Generated {len(all_execs)} exec_trajs with metadata saved!")

if __name__=="__main__":
    main()
