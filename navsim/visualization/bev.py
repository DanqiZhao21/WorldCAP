from typing import Any, Dict, List
import matplotlib.pyplot as plt

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.transform import translate_longitudinally

from navsim.common.dataclasses import Frame, Annotations, Trajectory, Lidar
from navsim.common.enums import BoundingBoxIndex, LidarIndex

from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.visualization.lidar import filter_lidar_pc, get_lidar_pc_color
from navsim.visualization.config import (
    BEV_PLOT_CONFIG,
    MAP_LAYER_CONFIG,
    AGENT_CONFIG,
    LIDAR_CONFIG,
)
import matplotlib.cm as cm
from matplotlib.colors import to_rgba


def add_configured_bev_on_ax(ax: plt.Axes, map_api: AbstractMap, frame: Frame) -> plt.Axes:
    """
    Adds birds-eye-view visualization optionally with map, annotations, or lidar
    :param ax: matplotlib ax object
    :param map_api: nuPlans map interface
    :param frame: navsim frame dataclass
    :return: ax with plot
    """    

    if "map" in BEV_PLOT_CONFIG["layers"]:
        add_map_to_bev_ax(ax, map_api, StateSE2(*frame.ego_status.ego_pose))

    if "annotations" in BEV_PLOT_CONFIG["layers"]:
        add_annotations_to_bev_ax(ax, frame.annotations)

    if "lidar" in BEV_PLOT_CONFIG["layers"]:
        add_lidar_to_bev_ax(ax, frame.lidar)

    return ax


def add_annotations_to_bev_ax(
    ax: plt.Axes, annotations: Annotations, add_ego: bool = True
) -> plt.Axes:
    """
    Adds birds-eye-view visualization of annotations (ie. bounding boxes)
    :param ax: matplotlib ax object
    :param annotations: navsim annotations dataclass
    :param add_ego: boolean weather to add ego bounding box, defaults to True
    :return: ax with plot
    """    

    for name_value, box_value in zip(annotations.names, annotations.boxes):
        agent_type = tracked_object_types[name_value]

        x, y, heading = (
            box_value[BoundingBoxIndex.X],
            box_value[BoundingBoxIndex.Y],
            box_value[BoundingBoxIndex.HEADING],
        )
        box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
        agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)

        add_oriented_box_to_bev_ax(ax, agent_box, AGENT_CONFIG[agent_type])

    if add_ego:
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=StateSE2(0, 0, 0),
            vehicle_parameters=get_pacifica_parameters(),
        )
        add_oriented_box_to_bev_ax(
            ax, car_footprint.oriented_box, AGENT_CONFIG[TrackedObjectType.EGO], add_heading=False
        )
    return ax


def add_map_to_bev_ax(ax: plt.Axes, map_api: AbstractMap, origin: StateSE2) -> plt.Axes:
    """
    Adds birds-eye-view visualization of map (ie. polygons / lines)
    TODO: add more layers for visualizations (or flags in config)
    :param ax: matplotlib ax object
    :param map_api: nuPlans map interface
    :param origin: (x,y,θ) dataclass of global ego frame
    :return: ax with plot
    """    

    # layers for plotting complete layers
    polygon_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
    ]

    # layers for plotting complete layers
    polyline_layers: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
    ]

    # query map api with interesting layers
    map_object_dict = map_api.get_proximal_map_objects(
        point=origin.point,
        radius=max(BEV_PLOT_CONFIG["figure_margin"]),
        layers=list(set(polygon_layers + polyline_layers)),
    )

    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """ Helper for transforming shapely geometry in coord-frame """
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y
        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])
        return rotated_geometry

    for polygon_layer in polygon_layers:
        for map_object in map_object_dict[polygon_layer]:
            polygon: Polygon = _geometry_local_coords(map_object.polygon, origin)
            add_polygon_to_bev_ax(ax, polygon, MAP_LAYER_CONFIG[polygon_layer])

    for polyline_layer in polyline_layers:
        for map_object in map_object_dict[polyline_layer]:
            linestring: LineString = _geometry_local_coords(
                map_object.baseline_path.linestring, origin
            )
            add_linestring_to_bev_ax(
                ax, linestring, MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]
            )
    return ax


def add_lidar_to_bev_ax(ax: plt.Axes, lidar: Lidar) -> plt.Axes:
    """
    Add lidar point cloud in birds-eye-view
    :param ax: matplotlib ax object
    :param lidar: navsim lidar dataclass
    :return: ax with plot
    """    

    lidar_pc = filter_lidar_pc(lidar.lidar_pc)
    lidar_pc_colors = get_lidar_pc_color(lidar_pc, as_hex=True)
    ax.scatter(
        lidar_pc[LidarIndex.Y],
        lidar_pc[LidarIndex.X],
        c=lidar_pc_colors,
        alpha=LIDAR_CONFIG["alpha"],
        s=LIDAR_CONFIG["size"],
        zorder=LIDAR_CONFIG["zorder"],
    )
    return ax


def add_trajectory_to_bev_ax(
    ax: plt.Axes, trajectory: Trajectory, config: Dict[str, Any]
) -> plt.Axes:
    """
    Add trajectory poses as lint to plot
    :param ax: matplotlib ax object
    :param trajectory: navsim trajectory dataclass
    :param config: dictionary with plot parameters
    :return: ax with plot
    """    
    poses = np.concatenate([np.array([[0, 0]]), trajectory.poses[:, :2]])
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
    )
    return ax

#FIXME:
def add_trajectory_to_bev_ax_with_multiColor(
    ax: plt.Axes,
    trajectory: Trajectory,
    config: Dict[str, Any],
    score: float = None,
    cmap_name: str = "Blues",
    score_min: float = 0.0,
    score_max: float = 1.0,
    best_pdmScore: float = -1.0,
    ) -> plt.Axes:
    """
    Add a trajectory (or multiple) to BEV plot, optionally using score to control color depth.
    
    :param ax: matplotlib ax object
    :param trajectory: navsim trajectory dataclass
    :param config: dictionary with plot parameters
    :param score: optional float in [0,1] controlling line color depth
    :param cmap_name: matplotlib colormap name (default: 'Blues')
    :param score_min: minimum possible score (for normalization)
    :param score_max: maximum possible score (for normalization)
    :return: ax with plotted trajectory
    """
    poses = np.concatenate([np.array([[0, 0]]), trajectory.poses[:, :2]])
    # print(f"🌈planning traj xy_pose: {poses}")
    min_z = 1
    max_z = 10
    if score is not None:
        zorder_val = min_z + (max_z - min_z) * ((score - score_min) / (score_max - score_min))
        # 自动归一化
        if score_min is None: score_min = score
        if score_max is None: score_max = score

        norm_score = 0.5
        if score_max != score_min:
            norm_score = np.clip((score - score_min) / (score_max - score_min), 0, 1)
            if score == score_max:
                line_color =  "#623BED"  
                alpha = 0.95  # 高分全不透明
                marker_color = "#2ADCEFD7"
                zorder_val = 200  # 最高分置于最上层
            elif best_pdmScore != 0.0 and score == best_pdmScore:
                line_color = "#053BDB"
                alpha = 0.95
                marker_color = "#FFD700"
                zorder_val = 190  # 次高覆盖层级，仅次于最高分
                # print("💚 Best PDM trajectory plotted in gold!")
            else:
                cmap = cm.get_cmap(cmap_name)
                color_rgb = cmap(norm_score)[:3]
                # alpha = 0.75 + 0.2 * norm_score
                alpha =0.2 * norm_score
                line_color = (*color_rgb, alpha)
                marker_color = line_color       # 小圈圈颜色与线条一致
                # 其他保持按分数映射的层级
                zorder_val = zorder_val
            cmap = cm.get_cmap(cmap_name)
        else:
            # 无 score 时使用默认样式与较低层级，避免覆盖重点轨迹
            line_color = config.get("line_color", "#2c7fb8")
            marker_color = line_color
            alpha = config.get("line_color_alpha", 0.9)
            zorder_val = config.get("zorder", 5)
            line_color = (*color_rgb, alpha)
            marker_color = line_color       # 小圈圈颜色与线条一致
     
    # else:
    #     line_color =  "coral" 
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=line_color,
        linewidth=config.get("line_width", 2.0),
        linestyle=config.get("line_style", "-"),
        marker="o",                    # 圆圈
        markersize=3,                  # ⚡ 小圈圈
        markeredgecolor="none",        # ⚡ 去掉黑色边框
        markerfacecolor=marker_color,    # ⚡ 圈圈颜色与线条颜色一致
        
        # zorder=config.get("zorder", 5),
        zorder=zorder_val,  # ⚡ 根据 score 调整绘制顺序
    )
    return ax
##新增————专门为simulation的轨迹绘制没有score信息
def add_simulation_state_to_bev_ax_noScore(ax: plt.Axes, traj_xy: np.ndarray) -> plt.Axes:
    """
    绘制一条轨迹(ndarray)不使用 score
    traj_xy.shape = (time, 2)
    """
    poses = np.concatenate([np.array([[0, 0]]), traj_xy])
    # print(f"🔵 simulation_traj xy: {traj_xy}")

    line_color = to_rgba("#87C489", alpha=0.05)
    marker_color = to_rgba("#065705FF", alpha=0.15)
    ax.plot(
        poses[:, 1],
        poses[:, 0],
        color=line_color,
        linewidth=2.0,
        linestyle="-.",
        marker="o",
        markersize=2,
        markeredgecolor="none",
        markerfacecolor=marker_color,
    )
    return ax


def _plot_mask_segments(
    ax: plt.Axes,
    traj_xy: np.ndarray,
    mask: np.ndarray,
    color: str = "#FF0000",
    alpha: float = 0.7,
    linewidth: float = 1.0,
    zorder: int = 215,
) -> None:
    """
    在轨迹上高亮掩码为 True 的连续片段。
    :param ax: matplotlib ax 对象
    :param traj_xy: ndarray，形状 (T, 2)，仿真后的轨迹 (x,y)
    :param mask: ndarray，形状 (T,)，布尔掩码，True 表示逆行/违规区段
    :param color: 叠加段颜色
    :param alpha: 透明度
    :param linewidth: 线宽
    :param zorder: 绘制层级
    """
    if mask is None:
        return
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)
    if mask.ndim != 1 or mask.shape[0] != traj_xy.shape[0]:
        return

    # 提取连续 True 片段 [start, end]
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, mask.shape[0] - 1))

    # 逐段绘制红色半透明覆盖
    for s, e in runs:
        if e <= s:
            continue
        seg = traj_xy[s : e + 1]
        poses = np.concatenate([np.array([[0, 0]]), seg])
        ax.plot(
            poses[:, 1],
            poses[:, 0],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle="-",
            zorder=zorder,
        )


def add_simulated_pair_with_oncoming(
    ax: plt.Axes,
    sim_xy_model: np.ndarray,
    sim_xy_pdm: np.ndarray,
    oncoming_mask_model: np.ndarray | None = None,
    oncoming_mask_pdm: np.ndarray | None = None,
) -> plt.Axes:
    """
    在 BEV 同时绘制两条仿真后的轨迹（模型选择 vs PDM-best），并高亮逆行掩码片段。

    :param ax: matplotlib ax 对象
    :param sim_xy_model: 模型选择轨迹，形状 (T, 2)
    :param sim_xy_pdm: PDM-best 轨迹，形状 (T, 2)
    :param oncoming_mask_model: 可选，模型选择轨迹的逆行掩码 (T,)
    :param oncoming_mask_pdm: 可选，PDM-best 轨迹的逆行掩码 (T,)
    :return: ax
    """
    # 绘制模型选择（橙色）
    poses_m = np.concatenate([np.array([[0, 0]]), sim_xy_model])
    ax.plot(
        poses_m[:, 1],
        poses_m[:, 0],
        color="#F2AB07",
        linewidth=1.5,
        linestyle="-",
        marker="o",
        markersize=2,
        markeredgecolor="none",
        markerfacecolor="#DB7745",
        zorder=220,
        label="Sim(Model-selected)",
    )
    _plot_mask_segments(ax, sim_xy_model, oncoming_mask_model, color="#FF0000", alpha=0.35, linewidth=3.0, zorder=215)

    # 绘制 PDM-best（蓝色）
    poses_p = np.concatenate([np.array([[0, 0]]), sim_xy_pdm])
    ax.plot(
        poses_p[:, 1],
        poses_p[:, 0],
        color="#2E7FAB",
        linewidth=1.5,
        linestyle="-",
        marker="o",
        markersize=2,
        markeredgecolor="none",
        markerfacecolor="#4384F4",
        zorder=210,
        label="Sim(PDM-best)",
    )
    _plot_mask_segments(ax, sim_xy_pdm, oncoming_mask_pdm, color="#FF0000", alpha=0.35, linewidth=3.0, zorder=216)

    # 简单图例（可选）
    try:
        ax.legend(loc="lower left")
    except Exception:
        pass

    return ax


#FIXME:


def add_oriented_box_to_bev_ax(
    ax: plt.Axes, box: OrientedBox, config: Dict[str, Any], add_heading: bool = True
) -> plt.Axes:
    """
    Adds birds-eye-view visualization of surrounding bounding boxes
    :param ax: matplotlib ax object
    :param box: nuPlan dataclass for 2D bounding boxes
    :param config: dictionary with plot parameters
    :param add_heading: whether to add a heading line, defaults to True
    :return: ax with plot
    """    

    box_corners = box.all_corners()
    corners = [[corner.x, corner.y] for corner in box_corners]
    corners = np.asarray(corners + [corners[0]])

    ax.fill(
        corners[:, 1],
        corners[:, 0],
        color=config["fill_color"],
        alpha=config["fill_color_alpha"],
        zorder=config["zorder"],
    )
    ax.plot(
        corners[:, 1],
        corners[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )

    if add_heading:
        future = translate_longitudinally(box.center, distance=box.length / 2 + 1)
        line = np.array([[box.center.x, box.center.y], [future.x, future.y]])
        ax.plot(
            line[:, 1],
            line[:, 0],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            zorder=config["zorder"],
        )

    return ax


def add_polygon_to_bev_ax(ax: plt.Axes, polygon: Polygon, config: Dict[str, Any]) -> plt.Axes:
    """
    Adds shapely polygon to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param polygon: shapely Polygon 
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """    

    def _add_element_helper(element: Polygon):
        """ Helper to add single polygon to ax """
        exterior_x, exterior_y = element.exterior.xy
        ax.fill(
            exterior_y,
            exterior_x,
            color=config["fill_color"],
            alpha=config["fill_color_alpha"],
            zorder=config["zorder"],
        )
        ax.plot(
            exterior_y,
            exterior_x,
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            zorder=config["zorder"],
        )
        for interior in element.interiors:
            x_interior, y_interior = interior.xy
            ax.fill(
                y_interior,
                x_interior,
                color=BEV_PLOT_CONFIG["background_color"],
                zorder=config["zorder"],
            )
            ax.plot(
                y_interior,
                x_interior,
                color=config["line_color"],
                alpha=config["line_color_alpha"],
                linewidth=config["line_width"],
                linestyle=config["line_style"],
                zorder=config["zorder"],
            )

    if isinstance(polygon, Polygon):
        _add_element_helper(polygon)
    else:
        # NOTE: in rare cases, a map polygon has several sub-polygons.
        for element in polygon:
            _add_element_helper(element)

    return ax


def add_linestring_to_bev_ax(
    ax: plt.Axes, linestring: LineString, config: Dict[str, Any]
) -> plt.Axes:
    """
    Adds shapely linestring (polyline) to birds-eye-view visualization
    :param ax: matplotlib ax object
    :param linestring: shapely LineString
    :param config: dictionary containing plot parameters
    :return: ax with plot
    """    

    x, y = linestring.xy
    ax.plot(
        y,
        x,
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )

    return ax
