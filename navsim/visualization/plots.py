from typing import Any, Callable, List, Optional,Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import io
from typing import Dict, Union, List

from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from navsim.common.dataclasses import Scene
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG
from navsim.agents.abstract_agent import AbstractAgent
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax,add_trajectory_to_bev_ax_with_multiColor,add_simulation_state_to_bev_ax_noScore
from navsim.visualization.camera import (
    add_annotations_to_camera_ax,
    add_lidar_to_camera_ax,
    add_camera_ax,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from navsim.common.dataclasses import Trajectory
from pathlib import Path






#TODO:
print('larger figure margin to 128')
BEV_PLOT_CONFIG["figure_margin"] = (128, 128)

def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def configure_all_ax(ax: List[List[plt.Axes]]) -> List[List[plt.Axes]]:
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax


def plot_bev_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    General plot for birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent(scene: Scene, agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    require_scene = agent.requires_scene if hasattr(agent, "requires_scene") else False
    if require_scene:
        agent_trajectory = agent.compute_trajectory(scene.get_agent_input(), scene)
    else:
        agent_trajectory = agent.compute_trajectory(scene.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax

#FIXME:

#新增————绘制simulation前向仿真的轨迹
def plot_bev_with_agent_and_simulation(
    scene: Scene,
    agent: AbstractAgent,
    best_pdm_idx: Optional[int] = -1,#包含了baseline偏移，使用的时候应该需要-1
    best_pred_idx: Optional[int] = -1,#包含了baseline偏移，使用的时候应该需要-1
    human_trajectory: Optional[Trajectory] = None,
    agent_trajectory: Optional[dict] = None,#注意传入的是一个字典而不是单独的多模态轨迹
    simulation_state: Optional[Trajectory] = None,  # 可选的 simulation rollout
    pdm_result_pred: Optional[Any] = None,          # PDM subscores for model-selected 小分结果
    pdm_result_best: Optional[Any] = None,          # PDM subscores for PDM-best 小分结果
    oncoming_mask_pred: Optional[Any] = None,       # mask(T,) for model-selected
    oncoming_mask_best: Optional[Any] = None,       # mask(T,) for PDM-best
    save_path: Optional[Union[str, Path]] = None,   # 图片保存目录
    file_name: Optional[str] = None                 # 图片文件名
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization.
    Optionally overlays a simulation trajectory.
    """
    #这里的计算可以直接有参数获得
    # human_trajectory = scene.get_future_trajectory()
    # agent_trajectory = agent.compute_trajectory(scene.get_agent_input())
    frame_idx = scene.scene_metadata.num_history_frames - 1
    
    pink_cmap = LinearSegmentedColormap.from_list(
        # "my_pink",
        # ["#eaccd8", "#ed97c5f7", "#ee5293"]
        "my_pink",
        ["#e6bacb", "#ed97c5f7", "#f48ebfff"]
    )
    green_cmap = LinearSegmentedColormap.from_list(
        "my_green",
        ["#b3eea6", "#21df70", "#1d8106"]
    )

    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    #NOTE=== 绘制 human 轨迹 ===
    if human_trajectory is not None:
        add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])

    #NOTE=== 绘制 egoagent预测的 多模态轨迹 ===
    # if isinstance(agent_trajectory, list):#多模态轨迹
    best_candidate_idx = int(best_pdm_idx) - 1 if best_pdm_idx is not None else -1
    if agent_trajectory is not None:
        if isinstance(agent_trajectory, dict) and "trajectories" in agent_trajectory:
            trajectories = agent_trajectory["trajectories"]
            scores = agent_trajectory["trajectory_scores"]
            score_min = np.min(scores)
            score_max = np.max(scores)
            # PDM 返回的 best_pdm_idx 包含 baseline 的偏移（baseline 在 0），
            # 因此需要减 1 映射到模型候选的索引范围 [0, len(scores)-1]
            # best_candidate_idx = int(best_pdm_idx) - 1 if best_pdm_idx is not None else -1
            score_idx = None
            if 0 <= best_candidate_idx < len(scores):
                score_idx = scores[best_candidate_idx]
            
            # import ipdb; ipdb.set_trace()
            #这个是两套不一样的评分因该传入idx然后索取
            for traj, score in zip(trajectories, scores):
                # best_pdmScore=
                add_trajectory_to_bev_ax_with_multiColor(
                    ax,
    
            # ============ 分数信息框 ============
                    traj,
                    TRAJECTORY_CONFIG["agent"],
                    score=score,
                    cmap_name=pink_cmap,
                    score_min=score_min,
                    score_max=score_max,
                    best_pdmScore=score_idx
                )
            norm = Normalize(vmin=score_min, vmax=score_max)
            sm = ScalarMappable(cmap=cm.get_cmap(pink_cmap), norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Trajectory Score")
        else:
            add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])


    # 新增：simulation 轨迹绘制
    #NOTE=== 绘制 simulation 轨迹 ===

    if  simulation_state is not None:
        # print(f"🍠shape of simulation_state is {simulation_state.shape}")#🍠shape of simulation_state is (21, 41, 11)   -->(1, 41, 3)
        if isinstance(simulation_state, list) or isinstance(simulation_state, np.ndarray):
            for idx in range(simulation_state.shape[0]):
                sim_traj_state = simulation_state[idx]  # shape: (time, state_dim)
                traj_xy = sim_traj_state[:, :2]  # 提取 x, y
                
                add_simulation_state_to_bev_ax_noScore(ax, traj_xy)
        else:
            print(f"⚠️ simulation_traj 类型异常: {type(simulation_state)}")
        # 自动叠加：模型选择 vs PDM-best 仿真路径 + 逆行片段覆盖
        try:
            from navsim.visualization.bev import add_simulated_pair_with_oncoming
            if isinstance(agent_trajectory, dict) and "trajectory_scores" in agent_trajectory:
                # 计算仿真数组中的索引（包含 baseline 的偏移）
                pred_idx_sim = int(np.argmax(agent_trajectory["trajectory_scores"]))#这里无需加一也无需减一
                #DEBUG 这里至少验证了上一个函数传进来的best_pred_idx是正确的；
                print(f"🍀🍀🍀🍀🍀 pred_idx_real_idx is {best_pred_idx-1}   pred_idx_sim is {pred_idx_sim}")
                
                best_idx_sim = int(best_pdm_idx)-1 #NOTE这里best_pdm_idx已经包含baseline偏移了但是其实传进来的轨迹是没包含人类轨迹simulation的所以得减1

                sim_xy_model = simulation_state[pred_idx_sim][:, :2]
                sim_xy_pdm = simulation_state[best_idx_sim][:, :2]

                # 相对原点对齐以便于重叠查看
                sim_xy_model = sim_xy_model - sim_xy_model[0]
                sim_xy_pdm = sim_xy_pdm - sim_xy_pdm[0]

                add_simulated_pair_with_oncoming(
                    ax,
                    sim_xy_model,
                    sim_xy_pdm,
                    oncoming_mask_model=oncoming_mask_pred,
                    oncoming_mask_pdm=oncoming_mask_best,
                )
        except Exception as e:
            print(f"[PLOT] failed to overlay simulated pairs: {e}")
    
        # ============ 分数信息框（半透明黑框）===========
    #PRINT
        try:
            # 左上角：模型选择轨迹的 PDM 总分与小分（pred_idx）
            if pdm_result_pred is not None:
                def _gp(attr):
                    return getattr(pdm_result_pred, attr) if hasattr(pdm_result_pred, attr) else pdm_result_pred.get(attr)

                lines_pred = [
                    "Model-Selected (PDM)",
                    f"Total: {_gp('score'):.3f}",
                    f"NoCollision: {_gp('no_at_fault_collisions'):.3f}",
                    f"Drivable: {_gp('drivable_area_compliance'):.3f}",
                    f"DrivingDir: {_gp('driving_direction_compliance'):.3f}",
                    f"Progress: {_gp('ego_progress'):.3f}",
                    f"TTC: {_gp('time_to_collision_within_bound'):.3f}",
                    f"Comfort: {_gp('comfort'):.3f}",
                ]
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(lines_pred),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=10,
                    color="white",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.35, edgecolor="#333333"),
                    zorder=300,
                )

            # 右上角：PDM 评分最高轨迹的总分与小分（best_pdm_idx）
            if pdm_result_best is not None:
                def _gb(attr):
                    return getattr(pdm_result_best, attr) if hasattr(pdm_result_best, attr) else pdm_result_best.get(attr)

                lines_best = [
                    "PDM-Best",
                    f"Total: {_gb('score'):.3f}",
                    f"NoCollision: {_gb('no_at_fault_collisions'):.3f}",
                    f"Drivable: {_gb('drivable_area_compliance'):.3f}",
                    f"DrivingDir: {_gb('driving_direction_compliance'):.3f}",
                    f"Progress: {_gb('ego_progress'):.3f}",
                    f"TTC: {_gb('time_to_collision_within_bound'):.3f}",
                    f"Comfort: {_gb('comfort'):.3f}",
                ]
                ax.text(
                    0.98,
                    0.98,
                    "\n".join(lines_best),
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=10,
                    color="white",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.35, edgecolor="#333333"),
                    zorder=300,
                )
            # 左下角：展示 DrivingDirection 的底层 oncoming_progress（米）
            if (pdm_result_pred is not None) or (pdm_result_best is not None):
                def _gx(obj, attr):
                    return getattr(obj, attr) if hasattr(obj, attr) else obj.get(attr)

                lines_bottom = ["Oncoming Progress (m)"]
                if pdm_result_pred is not None and _gx(pdm_result_pred, 'oncoming_progress') is not None:
                    lines_bottom.append(f"Model: {_gx(pdm_result_pred, 'oncoming_progress'):.3f}")
                if pdm_result_best is not None and _gx(pdm_result_best, 'oncoming_progress') is not None:
                    lines_bottom.append(f"PDM-Best: {_gx(pdm_result_best, 'oncoming_progress'):.3f}")

                if len(lines_bottom) > 1:
                    ax.text(
                        0.02,
                        0.02,
                        "\n".join(lines_bottom),
                        transform=ax.transAxes,
                        va="bottom",
                        ha="left",
                        fontsize=10,
                        color="white",
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.35, edgecolor="#333333"),
                        zorder=300,
                    )
        except Exception as e:
            print(f"[PLOT] score box rendering failed: {e}")
    #PRINT

    configure_bev_ax(ax)
    configure_ax(ax)
#FIXME:保存图片
    if save_path is not None and file_name is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / file_name, bbox_inches="tight")
        plt.close(fig)  # 避免内存泄露
        # print(f"✅ Token {file_name} simulation_bev 已成功保存")




#FIXME:





TAB_10: Dict[int, str] = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
}


NEW_TAB_10: Dict[int, str] = {
    0: "#4e79a7",  # blue
    1: "#f28e2b",  # orange
    2: "#e15759",  # red
    3: "#76b7b2",  # cyan
    4: "#59a14f",  # green
    5: "#edc948",  # yellow
    6: "#b07aa1",  # violet
    7: "#ff9da7",
    8: "#9c755f",
    9: "#bab0ac",
}


ELLIS_5: Dict[int, str] = {
    0: "#DE7061",  # red
    1: "#B0E685",  # green
    2: "#4AC4BD",  # cyan
    3: "#E38C47",  # orange
    4: "#699CDB",  # blue
}

TRAJECTORY_WITH_CANDIDATES_CONFIG: Dict[str, Any] = {
    "human": {
        "fill_color": NEW_TAB_10[4],
        "fill_color_alpha": 1.0,
        "line_color": NEW_TAB_10[4],
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 5,
        "marker_edge_color": "black",
        "zorder": 3,
    },
    "agent": {
        "fill_color": ELLIS_5[0],
        "fill_color_alpha": 1.0,
        "line_color": ELLIS_5[0],
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 5,
        "marker_edge_color": "black",
        "zorder": 3,
    },
    "candidates": {
        "line_color": "yellow",               # 低分轨迹的基础颜色
        "line_color_alpha": 0.5,              # 低分轨迹的透明度
        "line_width": 2.0,
        "line_style": "--",
        "zorder": 2,
        "score_threshold": 0.5,               # 分数阈值，用于筛选轨迹
        "colormap": "plasma",                 # 颜色映射方案
        "line_alpha": 1.0,                    # 高分轨迹的透明度系数
        "low_score_color": "gray",            # 低分轨迹的颜色
        "low_score_alpha": 0.3,               # 低分轨迹的透明度
    }
}

def plot_bev_with_agent_and_traj_candidates(scene: Scene, agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    agent_trajectory, all_traj_candidates, final_scores, im_rewards = agent.compute_trajectory_with_vis(scene.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    
    # Add all trajectory candidates to the plot
    add_trajectory_candidates_to_bev_ax(ax, all_traj_candidates, final_scores, TRAJECTORY_WITH_CANDIDATES_CONFIG["candidates"])
    
    # Add the selected agent trajectory to the plot
    # add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_WITH_CANDIDATES_CONFIG["agent"])
    
    # Add human trajectory to the plot
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_WITH_CANDIDATES_CONFIG["human"])
    
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def add_trajectory_candidates_to_bev_ax(
        ax: plt.Axes, 
        all_trajectories: np.ndarray, 
        scores: np.ndarray, 
        config: Dict[str, Any]
    ) -> plt.Axes:
    """
    Add all trajectory candidates to the plot, with color intensity representing their scores.
    :param ax: matplotlib ax object
    :param all_trajectories: numpy array of shape [256, 8, 3] containing all trajectory candidates
    :param scores: numpy array of shape [256] containing scores for each trajectory
    :param config: dictionary with plot parameters for 'candidates'
    :return: ax with plot
    """
    # 获取阈值
    threshold = config.get("score_threshold", 0.0)
    
    # 筛选分数大于阈值的轨迹
    valid_indices = scores > threshold
    if not np.any(valid_indices):
        print("No trajectories exceed the score threshold.")
        return ax

    # 仅使用分数大于阈值的轨迹来计算归一化分数
    valid_scores = scores[valid_indices]
    normalized_scores = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
    
    # 获取颜色映射
    cmap_name = config.get("colormap", "viridis")
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=valid_scores.min(), vmax=valid_scores.max())
    
    # 遍历所有轨迹
    for i, trajectory in enumerate(all_trajectories):
        poses = np.concatenate([np.array([[0, 0]]), trajectory[:, :2]])

        if scores[i] > threshold:
            # 获取归一化分数并映射到颜色
            color = cmap(norm(scores[i]))
            # 获取归一化分数对应的alpha值
            normalized_score = (scores[i] - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
            alpha = np.clip(normalized_score * config.get("line_alpha", 1.0), 0, 1)
        else:
            # # 对于低于阈值的轨迹，使用更轻微透明的颜色
            # color = "#A3B8C8"  # A soft, light blue color for low-scoring trajectories
            # alpha = config.get("low_score_alpha", 0.2)  # Lower alpha for more transparency

            color = "#d3d3d3"  # 淡灰色
            alpha = 0.4  # 更低的透明度
        
        ax.plot(
            poses[:, 1],
            poses[:, 0],
            color=color,
            alpha=alpha,
            linewidth=config.get("line_width", 1.0),
            linestyle=config.get("line_style", "-"),
            zorder=config.get("zorder", 1),
        )
    
    # 添加颜色条（仅当有高分轨迹时）
    if np.any(valid_indices):
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 仅用于颜色条
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Trajectory Score')
    
    return ax


def plot_cameras_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_lidar(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_lidar_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.lidar)

    add_lidar_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.lidar)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_lidar_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.lidar)

    add_lidar_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.lidar)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
    add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

    add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

    add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
    add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def frame_plot_to_pil(
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = callable_frame_plot(scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images


def frame_plot_to_gif(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    """
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices)
    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)
