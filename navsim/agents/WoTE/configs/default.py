from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class WoTEConfig:

    trajectory_sampling: TrajectorySampling = TrajectorySampling(
        time_horizon=4, interval_length=0.5
    )

    resnet34_path = '/home/zhaodanqi/clone/WoTE/extra_data/resnet34.pth'
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    num_keyval = 64 # 8*8
    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    # detect_boxes = True
    # use_bev_semantic = True

    detect_boxes = False
    use_bev_semantic = False
    
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
        7: ("ego_box", [TrackedObjectType.VEHICLE]),  # ego box
    }
    use_ego_box_in_map = True
    ego_box_map_idx = 7

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 8
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])

    # New
    # k-means traj
    num_traj_anchor: int = 256
    
    use_sim_reward: bool = True
    #NOTE
    sim_reward_dict_path: str = f'/home/zhaodanqi/clone/WoTE/extra_data/planning_vb/formatted_pdm_score_256.npy'
    cluster_file_path: str='/home/zhaodanqi/clone/WoTE/extra_data/planning_vb/trajectory_anchors_256.npy' #NOTE 这个一般也不变动
    controller_ref_traj_path: str = "/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" #NOTE这个一般不变动
    controller_exec_traj_path: str = "/home/zhaodanqi/clone/WoTE/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy"
    #NOTE
    num_plan_queries: int = 32

    use_agent_loss: bool = True
    # map loss
    input_target = True
    use_map_loss: bool = True
    use_focal_loss_for_map = True
    bev_semantic_weight: float = 0.5
    future_idx = 11
    fut_bev_semantic_weight: float = 0.5#改到了1
    focal_loss_alpha = 0.5
    focal_loss_gamma = 2.0
    
    # sampled trajs for supervision
    num_sampled_trajs: int = 1

    # recurrent
    num_fut_timestep = 1
    use_traj_offset = True
    #NOTE
    # controller fusion mode (simplified): 'film' | 'add'
    controller_injection_mode: str = 'film'
    # controller injection strength (0.0 ~ 1.0)
    controller_injection_strength: float = 0.25#没有使用了

    # Controller-aware BEV world model conditioning.
    # These are read via getattr() in the model, but MUST exist in the dataclass
    # if we want to override them via Hydra CLI.
    controller_condition_on_world_model: bool = True
    # Fusion mode used in WoTE_model.py for world-model conditioning.
    # - 'attn': cross-attention over controller bank embeddings (default)
    # - 'film': FiLM-style token modulation (scale+shift) blended by controller_world_model_strength
    # - 'add' : legacy additive injection with controller_world_model_strength
    controller_world_model_fusion: str = 'attn'
    controller_world_model_strength: float = 0.3
    # Where to inject controller token in world model input tokens: 'all' | 'ego'
    controller_world_model_inject_target: str = 'all'

    # If True, inject controller only at the first world-model step (step 0).
    # Default False keeps the historical behavior: inject at every world-model step.
    controller_world_model_inject_first_step_only: bool = False

    # Only used when controller_world_model_fusion='attn'
    controller_world_model_attn_heads: int = 8

    # controller feature extraction plugin
    # - 'full': use full controller features
    # - 'lateral_only': focus on lateral-related dynamics
    controller_feature_mode: str = 'full'

    # controller global style pooling (used when controller_exec_traj_path is a bank/bundle)
    # - 'attn': learned attention pooling (default)
    # - 'mean': average pooling
    controller_style_pooling: str = 'attn'

    # CAP: inject controller dynamics into trajectory feature branch.
    controller_condition_on_traj_feature: bool = True
    controller_traj_condition_strength: float = 0.25

    # Also allow controller to affect offset generation and scoring features.
    controller_condition_on_offset: bool = True
    controller_offset_condition_strength: float = 0.25
    # New name used by offset-branch injection (kept separate from historical name above).
    controller_offset_inject_strength: float = 0.3
    controller_condition_on_reward_feature: bool = True
    controller_reward_condition_strength: float = 0.35

    # ===== controller execution (response predictor) =====
    controller_execute_predicted_traj: bool = True
    controller_execute_apply_in_train: bool = True
    controller_execute_apply_in_eval: bool = False
    controller_response_predictor_path: str = "/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_response_predictor.pt"
    controller_response_predictor_trainable: bool = False
    #NOTE
    # optmizer
    use_coslr_opt = True
    lr_steps = [70] # not used
    scheduler_type = "MultiStepLR" # not used
    weight_decay: float = 1e-4
    optimizer_type = "AdamW"
    cfg_lr_mult = 0.1
    opt_paramwise_cfg = {
        "name":{
            "image_encoder":{
                "lr_mult": cfg_lr_mult
            }
        }
    }
    max_epochs = 100
    # learning rate floor and warmup epochs for cosine scheduler
    min_lr: float = 1e-6
    warmup_epochs: int = 3

    # loss weight
    traj_offset_loss_weight: float = 1.0
    offset_im_reward_weight: float = 0.1
    im_loss_weight: float = 1.0
    metric_loss_weight: float = 1.0

    # def __post_init__(self):
    #     # 默认让 cluster_file_path 与 controller_ref_traj_path 保持一致；
    #     # 如需不同，可在 CLI 显式覆盖 cluster_file_path。
    #     if self.cluster_file_path is None:
    #         self.cluster_file_path = self.controller_ref_traj_path
