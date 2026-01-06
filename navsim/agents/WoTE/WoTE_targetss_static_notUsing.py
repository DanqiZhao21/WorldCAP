from typing import Any, List, Dict, Union
import numpy.typing as npt
import torch
import numpy as np
from torchvision import transforms
import time
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene
import timm, cv2
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from enum import IntEnum

from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
from copy import deepcopy                          

class WoTETargetBuilderStatic(AbstractTargetBuilder):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        slice_indices=[3],
        sim_reward_dict_path=None,
        config=None,
    ):
        self._config = config
        self._trajectory_sampling = trajectory_sampling
        self.slice_indices = slice_indices

        # ===== sim reward（确定性，可 cache）=====
        self.sim_reward_dict_path = sim_reward_dict_path
        if self.sim_reward_dict_path is not None:
            self.sim_reward_dict = np.load(
                self.sim_reward_dict_path, allow_pickle=True
            ).item()
            self.sim_keys = [
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "ego_progress",
                "time_to_collision_within_bound",
                "comfort",
            ]

        self.future_idx = (
            config.future_idx if hasattr(config, "future_idx") else 11
        )

    def get_unique_name(self) -> str:
        return "WoTE_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        result = {}

        # =====================================================
        # 1. Ego future GT trajectory（确定性）
        # =====================================================
        assert len(self.slice_indices) == 1
        index = self.slice_indices[0]
        frame_offset = index - 3

        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses,
            frame_offset=frame_offset,
        )
        result["trajectory"] = torch.tensor(
            future_trajectory.poses
        ).unsqueeze(0)  # [1, T, 3]

        # =====================================================
        # 2. Sim reward（如果有）
        # =====================================================
        if self.sim_reward_dict_path is not None:
            token = scene.frames[index].token
            sim_reward_dict_single = self.sim_reward_dict[token]["trajectory_scores"][0]
            combined_sim_reward = np.vstack(
                [sim_reward_dict_single[k] for k in self.sim_keys]
            )
            result["sim_reward"] = torch.tensor(
                combined_sim_reward, dtype=torch.float32
            ).unsqueeze(0)

        # =====================================================
        # 3. 当前帧 agent targets
        # =====================================================
        if getattr(self._config, "use_agent_loss", True):
            annotations = scene.frames[index].annotations
            agent_states, agent_labels = self._compute_agent_targets(annotations)
            result["agent_states"] = agent_states
            result["agent_labels"] = agent_labels

        # =====================================================
        # 4. 当前帧 BEV（无 ego）
        # =====================================================
        if self._config.use_map_loss:
            ego_pose = StateSE2(*scene.frames[index].ego_status.ego_pose)
            bev_map = self._compute_bev_semantic_map(
                scene.frames[index].annotations,
                scene.map_api,
                ego_pose,
            )
            cur_ego_box = [0., 0., 0., 4.0, 2.0, 1.8, 0.]
            bev_semantic_map = self._add_ego_box_to_bev_map(bev_map, cur_ego_box)
            result["bev_semantic_map"] = bev_semantic_map

        # =====================================================
        # 5. 未来 agent（对齐到当前 ego）
        # =====================================================
        frame_interval = self.future_idx - index - 1
        ref_frame_offset = future_trajectory.poses[frame_interval]

        fut_annotations = scene.frames[self.future_idx].annotations
        fut_boxes_in_cur = self.transform_boxes_from_future_to_current_ego_frame(
            fut_annotations.boxes,
            ref_frame_offset,
        )
        fut_annotations_cur = deepcopy(fut_annotations)
        fut_annotations_cur.boxes = fut_boxes_in_cur

        fut_agent_states, fut_agent_labels = self._compute_agent_targets(
            fut_annotations_cur
        )
        result["fut_agent_states"] = fut_agent_states
        result["fut_agent_labels"] = fut_agent_labels

        # =====================================================
        # 6. 未来 BEV（无 ego、无 anchor）⭐关键==》 result["fut_bev_semantic_map"]还需要进一步完善
        # =====================================================
        ego_pose = StateSE2(*scene.frames[index].ego_status.ego_pose)
        fut_bev_base = self._compute_bev_semantic_map(
            fut_annotations_cur,
            scene.map_api,
            ego_pose,
        )
        result["fut_bev_semantic_map_base"] = fut_bev_base

        return result
    
#差了：result["sampled_trajs_index"] = random_sample_idx；；；


#FIXME:
    def transform_boxes_from_future_to_current_ego_frame(self, boxes_future: np.ndarray, points_rel: np.ndarray) -> np.ndarray:
        """
        将未来ego车坐标系下的盒子转换到当前ego车坐标系下。

        :param boxes_future: 形状为 (N, 7) 的数组，表示未来ego车坐标系下的 N 个盒子。
        :param points_rel: 形状为 (3,) 的数组，表示未来ego车相对于当前ego车的位姿差异 [dx, dy, dtheta]。
        :return: 形状为 (N, 7) 的数组，表示当前ego车坐标系下的 N 个盒子。
        """
        # 提取平移和旋转差异
        dx, dy, dtheta = points_rel  # dtheta 是未来ego车相对于当前ego车的朝向差异

        # 计算旋转矩阵
        cos_theta = np.cos(dtheta)
        sin_theta = np.sin(dtheta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

        # 提取未来ego车坐标系下的 (x, y) 坐标和 heading
        x_future = boxes_future[:, BoundingBoxIndex._X]
        y_future = boxes_future[:, BoundingBoxIndex._Y]
        heading_future = boxes_future[:, BoundingBoxIndex._HEADING]

        # 将坐标堆叠为形状为 (N, 2) 的数组
        coords_future = np.stack((x_future, y_future), axis=-1)

        # 进行坐标转换：旋转 + 平移
        coords_current = coords_future @ rotation_matrix.T + np.array([dx, dy])

        # 计算新的朝向角
        heading_current = heading_future + dtheta

        # 构造转换后的 boxes_current
        boxes_current = boxes_future.copy()
        boxes_current[:, BoundingBoxIndex._X] = coords_current[:, 0]
        boxes_current[:, BoundingBoxIndex._Y] = coords_current[:, 1]
        boxes_current[:, BoundingBoxIndex._HEADING] = heading_current

        return boxes_current

    def _compute_agent_targets(self, annotations):
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x, y, config) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                config.lidar_min_y <= y <= config.lidar_max_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
            self, annotations, map_api, ego_pose
        ) -> torch.Tensor:
        """
        Creates semantic map in BEV excluding ego_box.
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels (excluding ego box)
        """
        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":#静态要素:多边形和车道线等由地图获得
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            elif entity_type == 'box':
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _add_ego_box_to_bev_map(self, bev_semantic_map, ego_box) -> torch.Tensor:
        """
        Adds ego box mask to the existing BEV semantic map.
        :param bev_semantic_map: 2D torch tensor of semantic labels (excluding ego box)
        :param ego_box: ego vehicle box description (x, y, z, length, width, height, yaw)
        :return: 2D torch tensor of semantic labels with ego box added
        """
        entity_mask = self._compute_ego_box_mask(ego_box)
        bev_semantic_map[entity_mask] = self._config.ego_box_map_idx  # Assuming label for ego_box

        return bev_semantic_map

    def _compute_ego_box_mask(
        self, box_value
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
        x, y, heading = box_value[0], box_value[1], box_value[-1]
        box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
        agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
        exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
        exterior = self._coords_to_pixel(exterior)
        cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    def _compute_map_polygon_mask(
        self, map_api, ego_pose, layers
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api, ego_pose, layers
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
        self, annotations, layers
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
        self, map_api, ego_pose, layers
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self, layers=layers
        )
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


class BoundingBox2DIndex(IntEnum):

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
            and not attribute.startswith("__")
            and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
