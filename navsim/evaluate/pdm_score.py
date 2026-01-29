import numpy as np
import numpy.typing as npt

from typing import List
import os
import matplotlib.pyplot as plt

from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)

from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses

from navsim.common.dataclasses import PDMResults, Trajectory

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
    EgoAreaIndex,
    BBCoordsIndex,
)


def transform_trajectory(
    pred_trajectory: Trajectory, initial_ego_state: EgoState
) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(
        initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length
    )

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

    # NOTE: velocity and acceleration ignored by LQR + bicycle model
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    # NOTE: maybe make addition of initial_ego_state optional
    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Interpolated trajectory and return as numpy array
    :param trajectory: nuPlan's InterpolatedTrajectory object
    :param future_sampling: Sampling parameters for interpolation
    :param start_time: TimePoint object of start
    :return: Array of interpolated trajectory states.
    """

    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(trajectory_ego_states)


def pdm_score(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer
) -> PDMResults:
    """
    Runs PDM-Score and saves results in dataclass.
    :param metric_cache: Metric cache dataclass
    :param model_trajectory: Predicted trajectory in ego frame.
    :return: Dataclass of PDM-Subscores.
    """

    initial_ego_state = metric_cache.ego_state

    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
    )

    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    # Refactor & add / modify existing metrics.
    #NOTE: 只计算单条预测轨迹的分数，因此 pred_idx=1
    pred_idx = 1

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]
    driving_direction_compliance = scorer._multi_metrics[
        MultiMetricIndex.DRIVING_DIRECTION, pred_idx
    ]
    # underlying metric value for visualization
    try:
        oncoming_progress_val = float(scorer._oncoming_progress_value[pred_idx])
    except Exception:
        oncoming_progress_val = None

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]

    score = scores[pred_idx]

    # _debug_dump_scores(scorer, simulator, pred_idx)
        

    return PDMResults(
        no_at_fault_collisions,
        drivable_area_compliance,
        driving_direction_compliance,
        ego_progress,
        time_to_collision_within_bound,
        comfort,
        score,
        oncoming_progress=oncoming_progress_val,
    )
    
#TODO:将pdm_score改写成pdm_score_multiTraj

#FIXME:
def pdm_score_multiTraj(
    metric_cache: MetricCache,
    model_trajectories: List[Trajectory],
    trajectory_Anchor,
    model_scores: List[float],
    future_sampling: TrajectorySampling,  # 预测时长以及时间间隔
    simulator: PDMSimulator,
    scorer: PDMScorer,
    anchor_save_dir: str | None = None,
    anchor_save_name: str | None = None,
    anchor_overwrite: bool = False,
) -> PDMResults:
    """
    Runs PDM-Score and saves results in dataclass.
    :param metric_cache: Metric cache dataclass
    :param model_trajectory: Predicted trajectory in ego frame.
    :return: Dataclass of PDM-Subscores.
    """
    # future_sampling=10 #新增————手动修改预测时长in simulation,每次只需要10帧，也就是未来5s
    initial_ego_state = metric_cache.ego_state
    pdm_trajectory = metric_cache.trajectory  # 人类轨迹，为InterpolatedTrajectory对象

    # 将 human baseline 先转为数组，后续与每条预测拼接
    pdm_states = get_trajectory_as_array(
        pdm_trajectory, future_sampling, initial_ego_state.time_point
    )  # (T, D)

    # 逐轨评估：每次只给 [baseline, 当前预测] 两条，取索引1作为该预测分数
    per_traj_scores: List[float] = []
    per_traj_results: List[PDMResults] = []
    per_traj_oncoming_masks: List[np.ndarray] = []
    per_traj_oncoming_prog_vals: List[float | None] = []
    pred_sim_states_list: List[np.ndarray] = []  # 每条预测的模拟结果 (T, D)
    pred_input_states_list: List[np.ndarray] = []  # 每条预测的输入轨迹 (T, D)

    for traj in model_trajectories:
        # 预测轨迹转为全局并插值到与仿真一致
        pred_traj = transform_trajectory(traj, initial_ego_state)
        pred_states = get_trajectory_as_array(
            pred_traj, future_sampling, initial_ego_state.time_point
        )  # (T, D)

        # 构造两条 proposal 的批次：[baseline, pred]
        pair_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

        # 仿真两条
        pair_simulated = simulator.simulate_proposals(pair_states, initial_ego_state)  # (2, T, D)

        # 评分两条
        pair_scores = scorer.score_proposals(
            pair_simulated,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
        )

        # 索引1为当前预测的结果
        pred_idx_local = 1
        score_pred = float(pair_scores[pred_idx_local])
        per_traj_scores.append(score_pred)

        # 提取各指标并打包为 PDMResults（与单轨一致）
        no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx_local]
        drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx_local]
        driving_direction_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, pred_idx_local]
        ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx_local]
        time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx_local]
        comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx_local]
        # try:
        #     oncoming_progress_val = float(scorer._oncoming_progress_value[pred_idx_local])
        # except Exception:
        #     oncoming_progress_val = None

        per_traj_results.append(
            PDMResults(
                no_at_fault_collisions,
                drivable_area_compliance,
                driving_direction_compliance,
                ego_progress,
                time_to_collision_within_bound,
                comfort,
                score_pred,
                # oncoming_progress=oncoming_progress_val,
            )
        )

        # 保存 oncoming mask（用于可视化）
        # try:
        #     oncoming_mask_pred = scorer._ego_areas[pred_idx_local, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        # except Exception:
        #     oncoming_mask_pred = None
        # per_traj_oncoming_masks.append(oncoming_mask_pred)
        # per_traj_oncoming_prog_vals.append(oncoming_progress_val)

        # 累积预测的仿真结果与输入轨迹（均不包含 baseline）
        pred_sim_states_list.append(pair_simulated[pred_idx_local])  # (T, D)
        pred_input_states_list.append(pair_states[pred_idx_local])  # (T, D)

    # 模型选择的索引（基于模型打分）
    best_idx_model = int(np.argmax(model_scores))
    pred_idx = best_idx_model + 1  # 与历史约定保持“+1”代表跳过 baseline

    # PDM 选择的最佳索引（逐轨独立评估之后的 PDM 分数）
    best_pdm_no_offset = int(np.argmax(per_traj_scores))
    best_pdm_idx = best_pdm_no_offset + 1  # 对齐“+1”约定

    # # 打印对比日志
    # chosen_score = per_traj_scores[best_idx_model]
    # best_pdm_score = per_traj_scores[best_pdm_no_offset]
    # # print(
    # #     f"💚PDM_DEBUG_SELECT (per-traj) pred_idx={pred_idx} chosen_score={chosen_score:.4f} | "
    # #     f"💚best_pdm_idx={best_pdm_idx} best_pdm_score={best_pdm_score:.4f}"
    # # )

    # 准备返回的两个结果：模型选择的、PDM 选择的
    pdm_results = per_traj_results[best_idx_model]
    pdm_best_results = per_traj_results[best_pdm_no_offset]

    # Oncoming masks（模型选择 vs PDM 选择）
    # oncoming_mask_pred = per_traj_oncoming_masks[best_idx_model]
    # oncoming_mask_best = per_traj_oncoming_masks[best_pdm_no_offset]

    # 聚合所有预测的仿真与输入（不含 baseline），并转到 ego frame 的前 3 维
    pred_sim_all = np.stack(pred_sim_states_list, axis=0)  # (N, T, D)
    pred_input_all = np.stack(pred_input_states_list, axis=0)  # (N, T, D)

    simulated_states_all_egoframe = global_traj_to_ego_all(pred_sim_all, initial_ego_state)
    pred_states_all_ego_frame = global_traj_to_ego_all(pred_input_all, initial_ego_state)

    # print(f"🦋len of trajectory_Anchor is f{len(trajectory_Anchor)}")#(256 8 3)--》
    # print(f"🦋shape of trajectory_Anchor is f{trajectory_Anchor[0].shape}")#(256 8 3)--》 'Trajectory' object has no attribute 'shape' 

#======================================================
#使用navsim simulator进行anchor的simulation
#======================================================

# #PRINT
#     all_anchors_states = []
#     # print(f"🐕🐕🐕111")
#     for traj in trajectory_Anchor:
#         #transform_trajectory函数功能：Transform trajectory in global frame and return as InterpolatedTrajectory
#         anchor_traj = transform_trajectory(traj, initial_ego_state)#这个🟢pred_traj是插InterpolatedTrajectory对象：从ego_frame转化到global_frame;(ps:InterpolatedTrajectory这一步还没有进行插值,shape仍为（9,3））、
#         anchor_states = get_trajectory_as_array(anchor_traj, future_sampling, initial_ego_state.time_point)#(41,11)完成了插值
#         # print(f"🦋shape of traj is {pred_states.shape}")#(41, 11)
#         all_anchors_states.append(anchor_states)
    
#     #NOTE 这里的anchor就是最初始的标准的256条轨迹
#     trajectoryAnchor_states = np.concatenate([s[None, ...] for s in all_anchors_states], axis=0)

#     # configurable output path for simulated anchors
#     save_Anchor_dir = "/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive"
#     os.makedirs(save_Anchor_dir, exist_ok=True)
#     save_Anchor_path = os.path.join(
#         save_Anchor_dir,
#         "LQRstyle_aggressive.npy",
#     )
#     anchor_overwrite=True
#     # print(f"🐕🐕🐕222")
#     if anchor_overwrite or (not os.path.exists(save_Anchor_path)):
#         simulated_states_of_anchor = simulator.simulate_proposals(#这里得到的就是anchor经过不同风格的post_style以及LQR Style后的结果
#             trajectoryAnchor_states,
#             initial_ego_state
#         )
#         simulated_anchor_all_egoframe=global_traj_to_ego_all(simulated_states_of_anchor, initial_ego_state)
#         np.save(save_Anchor_path, simulated_anchor_all_egoframe)
#         print(f"🐕🐕🐕Simulation saved to: {save_Anchor_path}")
    
#PRINT     


    # 旧的“整批评分”逻辑已移除，以上已按逐轨两条方案生成 pdm_results / pdm_best_results
    # print(f"🦋shape of simulated_states_all is f{simulated_states_all_egoframe.shape}") #(21, 41, 11)[diffusionDrive] (256, 41, 3)[WoTE]
    return (
        pdm_results,
        pdm_best_results,
        simulated_states_all_egoframe,#没有包含人类轨迹
        pred_states_all_ego_frame,#没有包含人类轨迹
        best_pdm_idx,#给出包含baseline偏移的best_idx
        pred_idx,#给出包含baseline偏移的pred_idx
        # oncoming_mask_pred,
        # oncoming_mask_best,
    )

#FIXME:

#TODO:#新增 global-frame到ego-frame

#FIXME:
def global_traj_to_ego(traj_global_xy: np.ndarray, initial_ego_state):
    """
    将 (B, T, state_dim) 的全局轨迹转换为 ego 坐标系轨迹，只保留前三维 (x, y, yaw)
    并去掉第一条 baseline（只保留 diffusion proposals）。
    
    :param traj_global_xy: np.ndarray, shape = (B, T, state_dim)
    :param initial_ego_state: EgoState 对象
    :return: np.ndarray, shape = (B-1, T, 3)，转换后的 ego 坐标轨迹
    """
    # 去掉第一条 baseline
    traj_global_xy = traj_global_xy[1:]  # shape = (B-1, T, 11)
    # print("🦋In global_traj_to_ego, traj_global_xy.shape after drop baseline =", traj_global_xy.shape)

    # 只保留前三维
    traj_global_xy = traj_global_xy[:, :, :3]  # (x, y, yaw)

    # ego 位姿
    x0 = initial_ego_state.rear_axle.x
    y0 = initial_ego_state.rear_axle.y
    yaw0 = initial_ego_state.rear_axle.heading

    # 批量转换 global -> ego
    dx = traj_global_xy[:, :, 0] - x0  # (B-1, T)
    dy = traj_global_xy[:, :, 1] - y0

    x_ego =  dx * np.cos(yaw0) + dy * np.sin(yaw0)
    y_ego = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
    yaw_ego = traj_global_xy[:, :, 2] - yaw0  # 旋转角差

    traj_ego_xyyaw = np.stack([x_ego, y_ego, yaw_ego], axis=-1)  # (B-1, T, 3)
    return traj_ego_xyyaw
#FIXME:




#FIXME:
def global_traj_to_ego_all(traj_global_xy: np.ndarray, initial_ego_state):
    """
    将 (B, T, state_dim) 的全局轨迹转换为 ego 坐标系轨迹，只保留前三维 (x, y, yaw)
    并去掉第一条 baseline（只保留 diffusion proposals）。
    
    :param traj_global_xy: np.ndarray, shape = (B, T, state_dim)
    :param initial_ego_state: EgoState 对象
    :return: np.ndarray, shape = (B-1, T, 3)，转换后的 ego 坐标轨迹
    """
    # 去掉第一条 baseline
    # traj_global_xy = traj_global_xy[1:]  # shape = (B-1, T, 11)
    # print("🦋In global_traj_to_ego, traj_global_xy.shape after drop baseline =", traj_global_xy.shape)

    # 只保留前三维
    traj_global_xy = traj_global_xy[:, :, :3]  # (x, y, yaw)

    # ego 位姿
    x0 = initial_ego_state.rear_axle.x
    y0 = initial_ego_state.rear_axle.y
    yaw0 = initial_ego_state.rear_axle.heading

    # 批量转换 global -> ego
    dx = traj_global_xy[:, :, 0] - x0  # (B-1, T)
    dy = traj_global_xy[:, :, 1] - y0

    x_ego =  dx * np.cos(yaw0) + dy * np.sin(yaw0)
    y_ego = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
    yaw_ego = traj_global_xy[:, :, 2] - yaw0  # 旋转角差

    traj_ego_xyyaw = np.stack([x_ego, y_ego, yaw_ego], axis=-1)  # (B-1, T, 3)
    return traj_ego_xyyaw
#FIXME:

def extract_relative_trajectory(simulated_states: np.ndarray) -> np.ndarray:
    """
    Extract relative trajectory from simulated states.

    Parameters:
    - simulated_states: np.ndarray, shape (num_trajs, 41, 11)
        Array containing simulated trajectory states, where the first 3 dimensions are x, y, heading.

    Returns:
    - relative_trajectory: np.ndarray, shape (num_trajs, 9, 11)
        Array containing relative trajectory after downsampling, with the first 3 dimensions relative to the origin.
    """
    num_trajs, _, state_dim = simulated_states.shape

    # Step 1: Downsample by a factor of 5 (1 + 8 timesteps)
    downsampled_indices = np.arange(0, 41, 5)
    downsampled_states = simulated_states[:, downsampled_indices, :]

    # Step 2: Calculate relative trajectory with respect to the origin (first timestep)
    relative_trajectory = downsampled_states.copy()
    origin = StateSE2(*downsampled_states[:, 0, :3].T)  # Extract the origin state for each trajectory

    # Use convert_absolute_to_relative_se2_array to convert positions to relative coordinates
    for i in range(num_trajs):
        origin_state = StateSE2(*downsampled_states[i, 0, :3])
        relative_positions = convert_absolute_to_relative_se2_array(
            origin_state, np.array(downsampled_states[i, 1:, :3], dtype=np.float64)
        )
        relative_trajectory[i, 1:, :3] = relative_positions

    # Return the relative trajectory
    return relative_trajectory


def visualize_trajectories(trajectory_states, simulated_states, vis_dir='/data2/yingyan_li/repo/WoTE//vis/simulated_trajs'):
    """
    Visualize predicted and simulated trajectories in BEV.

    Parameters:
    - trajectory_states: np.ndarray, shape (num_trajs, 41, 11)
        Array containing predicted trajectory states, where the first 3 dimensions are x, y, heading.
    - simulated_states: np.ndarray, shape (num_trajs, 41, 11)
        Array containing simulated trajectory states, where the first 3 dimensions are x, y, heading.
    - vis_dir: str
        Directory to save visualized images.
    """
    # 创建输出目录
    os.makedirs(vis_dir, exist_ok=True)

    # 开始可视化
    num_trajs = trajectory_states.shape[0]
    for traj_idx in range(num_trajs):
        # 提取每条轨迹的状态
        traj = trajectory_states[traj_idx]
        sim_traj = simulated_states[traj_idx]

        # 获取初始原点作为偏移量
        origin = traj[0, :2]  # 使用第一个时间步的 x, y 作为初始原点

        # 计算相对于原点的偏移
        traj_positions = traj[:, :2] - origin
        sim_positions = sim_traj[:, :2] - origin

        # 创建可视化图像
        plt.figure(figsize=(8, 8))
        plt.plot(traj_positions[:, 0], traj_positions[:, 1], label='Predicted Trajectory', color='b', linestyle='--')
        plt.plot(sim_positions[:, 0], sim_positions[:, 1], label='Simulated Trajectory', color='r', linestyle='-')
        
        # 添加图例和标题
        plt.legend()
        plt.title(f'Trajectory Visualization - Index {traj_idx}')
        plt.xlabel('X (relative to origin)')
        plt.ylabel('Y (relative to origin)')
        plt.axis('equal')

        # 保存图片
        save_path = os.path.join(vis_dir, f'{traj_idx}.png')
        plt.savefig(save_path)
        plt.close()
        print(f'Trajectory visualization saved to {save_path}')

    print(f'All trajectory visualizations have been saved to {vis_dir}')


# ==========================
# Comprehensive debug helper
# ==========================
def _debug_dump_scores(scorer: PDMScorer, simulator: PDMSimulator, pred_idx: int) -> None:
    """
    Print all key metrics and intermediate values involved in scoring for a single proposal index.
    """
    try:
        # Build one-line debug message to avoid noisy logs in distributed runs
        num_poses = scorer.proposal_sampling.num_poses
        dt = scorer.proposal_sampling.interval_length
        post_style = getattr(simulator, "_post_style", "unknown")
        post_params = getattr(simulator, "_post_params", {})
        weights = scorer._config.weighted_metrics_array

        m_no_col = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
        m_drivable = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]
        m_direction = scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, pred_idx]
        m_prod_vec = scorer._multi_metrics.prod(axis=0)
        m_prod = m_prod_vec[pred_idx]

        w_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
        w_ttc = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
        w_comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]

        raw_progress = float(scorer._progress_raw[pred_idx])
        raw_progress_all = scorer._progress_raw * m_prod_vec
        denom = float(np.max(raw_progress_all))
        thresh = float(scorer._config.progress_distance_threshold)
        raw_after_multi = float(raw_progress_all[pred_idx])
        path = "divide_by_max" if denom > thresh else "fallback_ones"

        ttc_time = scorer.time_to_ttc_infraction(pred_idx)
        col_time = scorer.time_to_at_fault_collision(pred_idx)

        ego_areas = scorer._ego_areas[pred_idx]  # [T, numAreas]
        flags_multi = bool(ego_areas[:, EgoAreaIndex.MULTIPLE_LANES].any())
        flags_off = bool(ego_areas[:, EgoAreaIndex.NON_DRIVABLE_AREA].any())
        flags_oncoming = bool(ego_areas[:, EgoAreaIndex.ONCOMING_TRAFFIC].any())

        center = scorer._ego_coords[pred_idx, :, BBCoordsIndex.CENTER]  # [T,2]
        step_prog = np.zeros(center.shape[0], dtype=np.float64)
        step_prog[1:] = np.linalg.norm(center[1:] - center[:-1], axis=-1)
        oncoming_mask = ego_areas[:, EgoAreaIndex.ONCOMING_TRAFFIC]
        oncoming_step = step_prog.copy()
        oncoming_step[~oncoming_mask] = 0.0
        horizon = int(scorer._config.driving_direction_horizon / dt)
        rolling = np.array([
            oncoming_step[max(0, i - horizon) : i + 1].sum() for i in range(oncoming_step.shape[0])
        ], dtype=np.float64)
        oncoming_max = float(rolling.max())

        parts = [
            f" 🐷PDM_DEBUG pred_idx={pred_idx}",
            f" 🐷sampling num_poses={num_poses} dt={dt:.3f}s horizon={num_poses*dt:.2f}s",
            f" 🐷sim.post_style={post_style} post_params={post_params}",
            f" 🐷weights progress={weights[WeightedMetricIndex.PROGRESS]:.2f} ttc={weights[WeightedMetricIndex.TTC]:.2f} comfort={weights[WeightedMetricIndex.COMFORTABLE]:.2f}",
            f" 🐷multi no_at_fault={m_no_col:.4f} drivable={m_drivable:.4f} driving_dir={m_direction:.4f} product={m_prod:.4f}",
            f" 🐷weighted progress={w_progress:.4f} ttc={w_ttc:.4f} comfort={w_comfort:.4f}",
            f" 🐷progress raw={raw_progress:.4f} after_multi={raw_after_multi:.4f} denom={denom:.4f} thresh={thresh:.4f} path={path}",
            f" 🐷times ttc_infraction={ttc_time:.3f}s at_fault_collision={col_time:.3f}s",
            f" 🐷ego_areas multiple_lanes={flags_multi} off_road={flags_off} oncoming={flags_oncoming}",
            f" 🐷driving_dir horizon={horizon} oncoming_progress_max={oncoming_max:.4f} comp_thresh={scorer._config.driving_direction_compliance_threshold:.2f} viol_thresh={scorer._config.driving_direction_violation_threshold:.2f}",
        ]
        print(" | ".join(parts))
    except Exception as e:
        print(f"[PDM DEBUG] comprehensive dump failed: {e}")