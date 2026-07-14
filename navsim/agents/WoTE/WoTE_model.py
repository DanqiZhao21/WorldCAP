from re import T
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import timm
import time

from navsim.common.enums import StateSE2Index

import torchvision.models as models
import torch.nn.functional as F

import os
from datetime import datetime
from navsim.agents.WoTE.WoTE_targets import BoundingBox2DIndex
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from typing import Any, List, Dict, Union, Optional

import matplotlib.pyplot as plt


# Controller embedding for BEV world-model conditioning.
from navsim.agents.WoTE.controller import ControllerEmbedding



class ResNet34Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet34Backbone, self).__init__()
        # Load a pre-trained ResNet-34 model
        resnet = models.resnet34(pretrained=pretrained)
        # Remove the fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # Extract features from the last convolutional layer
        res = self.backbone(x)
        return res            
       
class WoTEModel(nn.Module):
    def __init__(self, 
                 config,
                ):
        super().__init__()
        # 保留配置以便运行期根据控制器执行轨迹重建目标（如 BEV 注入）
        self._config = config

        # Define constants as variables
        STATUS_ENCODING_INPUT_DIM = 4 + 2 + 2
        hidden_dim = 256
        NUM_CLUSTERS = config.n_clusters if hasattr(config, 'n_clusters') else 256
        CLUSTER_CENTERS_FEATURE_DIM = 24

        TRANSFORMER_DIM_FEEDFORWARD = 512
        TRANSFORMER_NHEAD = 8
        TRANSFORMER_DROPOUT = 0.1
        TRANSFORMER_NUM_LAYERS = 2

        SCORE_HEAD_HIDDEN_DIM = 128
        SCORE_HEAD_OUTPUT_DIM = 1
        NUM_SCORE_HEADS = 5

        # transfuser backbone
        self._backbone = TransfuserBackbone(config)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)

        self._status_encoding = nn.Linear(STATUS_ENCODING_INPUT_DIM, config.tf_d_model)

        num_poses = config.trajectory_sampling.num_poses
        num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 64
        self.num_plan_queries = num_keyval
        self._keyval_embedding = nn.Embedding(
            num_keyval, config.tf_d_model
        )

        # Load offline trajectories and MLP for planning vb feature
        cluster_file = config.cluster_file_path
        # print(f"💜词表路径是 {cluster_file}")
        self.trajectory_anchors = torch.nn.Parameter(
            torch.tensor(np.load(cluster_file)),
            requires_grad=False
        )

        self.mlp_planning_vb = nn.Sequential(
            nn.Linear(CLUSTER_CENTERS_FEATURE_DIM, SCORE_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM, hidden_dim),
        )

        # Transformer Encoder for trajectory_anchors_feat将anchor特征编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=TRANSFORMER_NHEAD, 
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD, 
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True
        )
        self.cluster_encoder = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_NUM_LAYERS)
        
        # latent world model
        self.num_scenes = self.num_plan_queries + 1  # including the ego feat and action(未来+当前)
        self.scene_position_embedding = nn.Embedding(self.num_scenes, hidden_dim)

        self.encode_ego_feat_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        wm_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=TRANSFORMER_NHEAD, 
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD, 
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True
        )
        self.latent_world_model = nn.TransformerEncoder(wm_encoder_layer, num_layers=TRANSFORMER_NUM_LAYERS)

        # reward conv net
        self.num_fut_timestep = config.num_fut_timestep if hasattr(config, 'num_fut_timestep') else 4
        self.reward_conv_net = RewardConvNet(input_channels=(self.num_fut_timestep+1) * hidden_dim, conv1_out_channels=hidden_dim, conv2_out_channels=hidden_dim)
        self.reward_cat_head = nn.Sequential(
            nn.Linear((self.num_fut_timestep + 1 + 1) * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )#输出统一维度的隐藏向量 hidden_dim；卷积网络之后，MLP 前；中间特征处理模块，不是直接输出分数，而是为后续评分提供更紧凑、整合的特征表示

        # MLP head for scoring#reward_cat_head 之后；单条轨迹的主评分 head；多个评分 head，用于多任务训练或评估不同类型的 reward 信号（比如安全性、效率、舒适性等）
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, SCORE_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM, SCORE_HEAD_OUTPUT_DIM),
        )
        self.sim_reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, SCORE_HEAD_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(SCORE_HEAD_HIDDEN_DIM, SCORE_HEAD_OUTPUT_DIM),
            ) for _ in range(NUM_SCORE_HEADS)
        ])

        # agent   ： 通过 Transformer Decoder 建模 ego 与其他 agents 的交互。
        self.use_agent_loss = config.use_agent_loss if hasattr(config, 'use_agent_loss') else True
        if self.use_agent_loss:
            self.agent_query_embedding = nn.Embedding(config.num_bounding_boxes, hidden_dim)
            self.agent_head = AgentHead(
                num_agents=config.num_bounding_boxes,
                d_ffn=config.tf_d_ffn,
                d_model=config.tf_d_model,
            )

            agent_tf_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )

            self.agent_tf_decoder = nn.TransformerDecoder(agent_tf_decoder_layer, config.tf_num_layers)

        # map
        self.use_map_loss = config.use_map_loss if hasattr(config, 'use_map_loss') else False
        if self.use_map_loss:
            self.bev_upsample_head = BEVUpsampleHead(config)
            self.bev_semantic_head = nn.Sequential(           #输出就是预测的 BEV map。
                nn.Conv2d(
                    config.bev_features_channels,
                    config.bev_features_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    config.bev_features_channels,
                    config.num_bev_classes,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Upsample(
                    size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                    mode="bilinear",
                    align_corners=False,
                ),
            )
        self._bev_upscale = nn.Conv2d(config.tf_d_model, 512, kernel_size=1)
        
        # future agent & map
        self.num_sampled_trajs = config.num_sampled_trajs if hasattr(config, 'num_sampled_trajs') else 1
        self.num_sampled_trajs_NoSample = 256  # Evaluation/debug path keeps all anchors.
        self.new_scene_bev_feature_pos_embed = nn.Embedding(self.num_plan_queries, hidden_dim)

        # offset
        offset_tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self.offset_tf_decoder = nn.TransformerDecoder(offset_tf_decoder_layer, config.tf_num_layers)
        self.offset_head = TrajectoryOffsetHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )
        self.offset_score_head = nn.Sequential(
            nn.Linear(hidden_dim, SCORE_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(SCORE_HEAD_HIDDEN_DIM, SCORE_HEAD_OUTPUT_DIM),
        )
        self.reward_weights = config.reward_weights if hasattr(config, 'reward_weights') else [0.1, 0.5, 0.5, 1.0]  
        
        # ===== controller embedding =====
        self.controller_emb_dim = 64
        self.controller_feature_mode = config.controller_feature_mode
        self.controller_encoder = ControllerEmbedding(
            emb_dim=self.controller_emb_dim,
            feature_mode=self.controller_feature_mode,
        )
        self.use_controller_wm = bool(config.use_controller_wm)
        self.controller_wm_token_scope = str(config.controller_wm_token_scope or "all").lower()
        self.controller_wm_first_step_only = bool(config.controller_wm_first_step_only)
        self.controller_wm_fusion = str(config.controller_wm_fusion or "attn_film").lower()

        self.ctrl_proj = nn.Linear(self.controller_emb_dim, 256)
        self.ctrl_token_ln = nn.LayerNorm(256)
        self.ctrl_bank_proj = nn.Linear(self.controller_emb_dim, 256)
        self.ctrl_bank_ln = nn.LayerNorm(256)

        attn_heads = int(config.controller_wm_attn_heads or 8)
        if 256 % attn_heads != 0:
            attn_heads = 8
        self.ctrl_fuse_attn = nn.MultiheadAttention(embed_dim=256, num_heads=attn_heads, batch_first=True)

        # FiLM-style modulation for world-model tokens.
        self.ctrl_wm_film_scale = nn.Linear(256, 256)
        self.ctrl_wm_film_shift = nn.Linear(256, 256)
        self.ctrl_wm_film_ln = nn.LayerNorm(256)
        self._load_controller_banks(config)

    @staticmethod
    def _controller_style_names(data):
        style_names = data.get('style_names', None)
        if style_names is None:
            return None
        try:
            return [str(name) for name in list(style_names)]
        except Exception:
            return None

    @staticmethod
    def _planner_anchors_in_first_pose_frame(planner_anchors: torch.Tensor) -> torch.Tensor:
        """Apply the controller bundle's first-pose offset convention.

        Bundle generation subtracts the first x/y/yaw values but intentionally
        keeps XY in the current ego axes (it does not apply an SE(2) rotation).
        """
        xy = planner_anchors[..., :2]
        yaw = planner_anchors[..., 2]
        xy0 = xy[:, :1]
        yaw0 = yaw[:, :1]
        rel_yaw = torch.atan2(torch.sin(yaw - yaw0), torch.cos(yaw - yaw0))
        return torch.cat((xy - xy0, rel_yaw.unsqueeze(-1)), dim=-1)

    def _load_controller_banks(self, config):
        """Load independent style-observation and planner-candidate banks.

        The style bank supplies controller tokens to the latent world model and
        may contain any number of trajectories.  The candidate bank supplies a
        style-matched executed rollout for each planner anchor and must preserve
        planner index and endpoint alignment.
        """
        style_ref_path = getattr(config, 'controller_style_ref_bank_path', None)
        style_exec_path = getattr(config, 'controller_style_exec_bank_path', None)
        candidate_exec_path = getattr(config, 'controller_candidate_exec_bank_path', None)
        legacy_ref_path = getattr(config, 'controller_ref_bank_path', None)
        legacy_exec_path = getattr(config, 'controller_exec_bank_path', None)
        style_ref_path = style_ref_path or legacy_ref_path
        style_exec_path = style_exec_path or legacy_exec_path
        candidate_exec_path = candidate_exec_path or legacy_exec_path

        if not style_exec_path:
            raise ValueError('controller_style_exec_bank_path is required')
        if not candidate_exec_path:
            raise ValueError('controller_candidate_exec_bank_path is required')

        style_data = np.load(style_exec_path, allow_pickle=True)
        if not isinstance(style_data, np.lib.npyio.NpzFile):
            if not style_ref_path:
                raise ValueError('controller_style_ref_bank_path is required for a plain .npy style exec bank')
            style_ref = np.load(style_ref_path)
            style_exec = np.asarray(style_data)[None, ...]
            style_names = None
            train_style_indices = None
            val_style_indices = None
        else:
            if 'exec_trajs' not in style_data.files or 'ref_traj' not in style_data.files:
                raise ValueError(
                    f"style controller bundle must contain exec_trajs/ref_traj: {style_exec_path}"
                )
            style_ref = np.asarray(style_data['ref_traj'])
            style_exec = np.asarray(style_data['exec_trajs'])
            style_names = self._controller_style_names(style_data)
            train_style_indices = style_data.get('train_style_indices', None)
            val_style_indices = style_data.get('val_style_indices', None)

        if style_ref.ndim != 3 or style_exec.ndim != 4:
            raise ValueError(
                'style controller bank must have ref [N,T,3] and exec [S,N,T,3], '
                f'got ref={style_ref.shape} exec={style_exec.shape}'
            )
        if style_exec.shape[1:] != style_ref.shape:
            raise ValueError(
                f'style controller ref/exec shapes do not align: ref={style_ref.shape} exec={style_exec.shape}'
            )

        if style_ref_path:
            external_style_ref = np.asarray(np.load(style_ref_path))
            if external_style_ref.shape != style_ref.shape or not np.allclose(
                external_style_ref, style_ref, atol=1e-4, rtol=0.0
            ):
                raise ValueError(
                    'controller_style_ref_bank_path does not match ref_traj in the style bundle: '
                    f'{style_ref_path} vs {style_exec_path}'
                )

        candidate_data = np.load(candidate_exec_path, allow_pickle=True)
        if not isinstance(candidate_data, np.lib.npyio.NpzFile):
            candidate_exec = np.asarray(candidate_data)[None, ...]
            candidate_ref = None
            candidate_style_names = None
        else:
            if 'exec_trajs' not in candidate_data.files or 'ref_traj' not in candidate_data.files:
                raise ValueError(
                    f"candidate controller bundle must contain exec_trajs/ref_traj: {candidate_exec_path}"
                )
            candidate_exec = np.asarray(candidate_data['exec_trajs'])
            candidate_ref = np.asarray(candidate_data['ref_traj'])
            candidate_style_names = self._controller_style_names(candidate_data)

        planner_anchors = self.trajectory_anchors.detach().cpu()
        expected_candidate_shape = tuple(planner_anchors.shape)
        if candidate_exec.ndim != 4 or tuple(candidate_exec.shape[1:]) != expected_candidate_shape:
            raise ValueError(
                'candidate controller exec bank must be [S,K,T,3] and align with planner anchors, '
                f'got exec={candidate_exec.shape} planner={expected_candidate_shape}'
            )
        if candidate_exec.shape[0] != style_exec.shape[0]:
            raise ValueError(
                'style and candidate controller bundles must contain the same number of styles, '
                f'got style={style_exec.shape[0]} candidate={candidate_exec.shape[0]}'
            )
        if style_names is not None and candidate_style_names is not None and style_names != candidate_style_names:
            raise ValueError('style and candidate controller bundles have different style_names ordering')

        if candidate_ref is not None:
            if tuple(candidate_ref.shape) != expected_candidate_shape:
                raise ValueError(
                    f'candidate ref bank must align with planner anchors: ref={candidate_ref.shape} '
                    f'planner={expected_candidate_shape}'
                )
            # Controller simulations use the first planner pose as their local
            # origin. Internal points may use a denser simulator time grid, so
            # the invariant alignment check is on the first and final poses.
            planner_local = self._planner_anchors_in_first_pose_frame(planner_anchors)
            atol = float(getattr(config, 'controller_candidate_alignment_atol', 1e-4))
            candidate_endpoints = torch.as_tensor(candidate_ref)[:, [0, -1], :3]
            planner_endpoints = planner_local[:, [0, -1], :3]
            if not torch.allclose(candidate_endpoints, planner_endpoints, atol=atol, rtol=0.0):
                max_error = float((candidate_endpoints - planner_endpoints).abs().max())
                raise ValueError(
                    'candidate controller ref bank is not index-aligned with planner anchors: '
                    f'max endpoint error={max_error:.6g}, atol={atol}'
                )

        self._controller_bundle_ref = torch.as_tensor(style_ref, dtype=torch.float32)
        self._controller_bundle_exec = torch.as_tensor(style_exec, dtype=torch.float32)
        self._controller_bundle_style_names = style_names
        self._controller_bundle_train_style_indices = train_style_indices
        self._controller_bundle_val_style_indices = val_style_indices
        self._controller_style_split = os.environ.get('WOTE_CTRL_STYLE_SPLIT', '').strip().lower() or None
        self._controller_candidate_bundle_exec = torch.as_tensor(candidate_exec, dtype=torch.float32)

        forced = os.environ.get('WOTE_CTRL_STYLE_IDX', '').strip()
        try:
            initial_idx = int(forced) if forced else 0
        except ValueError:
            initial_idx = 0
        initial_idx = max(0, min(int(self._controller_bundle_exec.shape[0]) - 1, initial_idx))
        self._active_ref_trajs = self._controller_bundle_ref
        self._active_exec_trajs = self._controller_bundle_exec[initial_idx]
        self._active_candidate_exec_trajs = self._controller_candidate_bundle_exec[initial_idx]

        print(
            f"Loaded controller banks: style={style_exec_path} "
            f"(styles={style_exec.shape[0]}, refs={style_exec.shape[1]}) "
            f"candidate={candidate_exec_path} (candidates={candidate_exec.shape[1]})"
        )

    def _maybe_sample_controller_style_for_batch(self):
        """Sample one controller style per forward call during training.

        - Only active when controller exec path is a bundle (.npz) with multiple styles.
        - Disabled in eval to avoid metric jitter.
        - Optional env override:
            WOTE_CTRL_STYLE_IDX=<int> forces a fixed style index.
            WOTE_CTRL_STYLE_DEBUG=1 prints chosen style.
        """
        if getattr(self, 'is_eval', False) and os.environ.get('WOTE_CTRL_EVAL_SAMPLE', '0') != '1':
            return
        if self._controller_bundle_exec is None or self._controller_bundle_ref is None:
            return

        try:
            num_styles = int(self._controller_bundle_exec.shape[0])
        except Exception:
            return
        if num_styles <= 1:
            self._active_ref_trajs = self._controller_bundle_ref
            self._active_exec_trajs = self._controller_bundle_exec[0]
            self._active_candidate_exec_trajs = self._controller_candidate_bundle_exec[0]
            return

        forced = os.environ.get('WOTE_CTRL_STYLE_IDX', None)
        if forced is not None and str(forced).strip() != '':
            try:
                idx = int(forced)
                idx = max(0, min(num_styles - 1, idx))
            except Exception:
                idx = 0
        else:
            # Optional sync with simulator: if simulator already picked a style index for this step,
            # follow it to ensure matched (simulator params <-> controller embedding) evaluation.
            # This is only active when eval sampling is enabled.
            if getattr(self, 'is_eval', False) and os.environ.get('WOTE_CTRL_EVAL_SAMPLE', '0') == '1':
                sim_idx_str = os.environ.get('PDM_SIM_STYLE_IDX', '').strip()
                if sim_idx_str != '':
                    try:
                        sim_idx = int(sim_idx_str)
                        sim_idx = max(0, min(num_styles - 1, sim_idx))
                        idx = sim_idx
                    except Exception:
                        idx = None
                else:
                    idx = None
            else:
                idx = None

            if idx is None:
                # Optional split restriction for training-time sampling.
                split = getattr(self, '_controller_style_split', None)
                if split in {'train', 'val'}:
                    idx_pool = None
                    if split == 'train':
                        idx_pool = getattr(self, '_controller_bundle_train_style_indices', None)
                    else:
                        idx_pool = getattr(self, '_controller_bundle_val_style_indices', None)

                    try:
                        if idx_pool is not None and len(idx_pool) > 0:
                            import random
                            idx = int(random.choice(list(idx_pool)))
                        else:
                            import random
                            idx = random.randrange(num_styles)
                    except Exception:
                        import random
                        idx = random.randrange(num_styles)
                else:
                    import random
                    idx = random.randrange(num_styles)

        self._active_ref_trajs = self._controller_bundle_ref
        self._active_exec_trajs = self._controller_bundle_exec[idx]
        self._active_candidate_exec_trajs = self._controller_candidate_bundle_exec[idx]

        # If eval sampling is enabled, publish the chosen style for the simulator to follow.
        if getattr(self, 'is_eval', False) and os.environ.get('WOTE_CTRL_EVAL_SAMPLE', '0') == '1':
            os.environ['PDM_SIM_STYLE_IDX'] = str(int(idx))

        if os.environ.get('WOTE_CTRL_STYLE_DEBUG', '0') == '1':
            label = None
            if isinstance(self._controller_bundle_style_names, list) and idx < len(self._controller_bundle_style_names):
                label = self._controller_bundle_style_names[idx]
            print(f"💜 controller style sampled: idx={idx}" + (f" name={label}" if label else ""))
            
    def encode_traj_into_ego_feat(self, ego_status_feat: torch.Tensor, init_trajectory_anchor: torch.Tensor, batch_size: int):
        """
        Encode trajectory into ego feature by processing cluster centers and concatenating features.

        Args:
            ego_status_feat (torch.Tensor): Ego status features.
            init_trajectory_anchor (torch.Tensor): Initial cluster centers.
            batch_size (int): Batch size.

        Returns:
            ego_feat_fixed_anchor_WoTE (torch.Tensor): Concatenated ego and trajectory features.
            num_traj (int): Number of trajectories.
        """
        trajectory_anchors_feat, num_traj = self._get_trajectory_anchors_feat(init_trajectory_anchor, batch_size)#将ego_status_feat 和 init_trajectory_anchor 拼，得到 encoded_ego_feature
        ego_feat_encoded = self._concatenate_ego_and_traj_features(ego_status_feat, trajectory_anchors_feat)
        return ego_feat_encoded, num_traj

    def extract_trajectory_feature(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, Any]:
        results = {}

        camera_feature = features["camera_feature"]
        lidar_feature = features["lidar_feature"]
        status_feature = features["status_feature"]
        batch_size = status_feature.shape[0]

        backbone_bev_feature, flatten_bev_feature = self._process_backbone_features(
            camera_feature, lidar_feature
        )
        ego_status_feat = self._get_ego_status_feature(status_feature)

        init_trajectory_anchor = self.trajectory_anchors.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        ego_feat_fixed_anchor_WoTE, num_traj = self.encode_traj_into_ego_feat(ego_status_feat, init_trajectory_anchor, batch_size)
        offset_dict = self._predict_offset(ego_feat_fixed_anchor_WoTE, flatten_bev_feature)
        results.update(offset_dict)

        if self.use_agent_loss:
            agents, _ = self._process_agent(batch_size, flatten_bev_feature)
            results.update(agents)
        if self.use_map_loss:
            bev_semantic_map, upsampled_bev_feature = self._process_map(flatten_bev_feature, batch_size)
            results["bev_semantic_map"] = bev_semantic_map

        if self.is_eval:
            reward_trajectory_anchors = init_trajectory_anchor + offset_dict["trajectory_offset"]
            ego_feat_for_reward_network, _ = self.encode_traj_into_ego_feat(ego_status_feat, reward_trajectory_anchors, batch_size)
        else:
            # Main WorldCAP training keeps WoTE's original raw-anchor target
            # space. Controller information is injected only inside the BEV
            # world-model transition, so planning gains are attributable to
            # controller-aware imagination rather than candidate re-labeling.
            reward_trajectory_anchors = init_trajectory_anchor
            ego_feat_for_reward_network = ego_feat_fixed_anchor_WoTE
        wm_rollout_anchors = self._build_controller_wm_rollout_anchors(reward_trajectory_anchors)#执行后轨迹 从bank里面sample

        return {
            "results": results,
            "batch_size": batch_size,
            "num_traj": num_traj,
            "flatten_bev_feature": flatten_bev_feature,
            "ego_feat": ego_feat_for_reward_network,
            "candidate_anchors": init_trajectory_anchor,
            "reward_trajectory_anchors": reward_trajectory_anchors,
            "wm_rollout_anchors": wm_rollout_anchors,
        }

    
    def extract_reward_feature(self, trajectory_outputs, targets) -> Dict[str, torch.Tensor]:
        results = trajectory_outputs["results"]
        batch_size = trajectory_outputs["batch_size"]
        num_traj = trajectory_outputs["num_traj"]
        flatten_bev_feature = trajectory_outputs["flatten_bev_feature"]  # [B, Nq, C]
        ego_feat = trajectory_outputs["ego_feat"]  # [B, K, 1, C]
        reward_trajectory_anchors = trajectory_outputs.get("reward_trajectory_anchors", None)
        wm_rollout_anchors = trajectory_outputs.get("wm_rollout_anchors", reward_trajectory_anchors)
        results["candidate_anchors"] = trajectory_outputs.get("candidate_anchors", None)

        # Rebuild future BEV targets along the controller-executed rollout.
        # This is the supervision signal that makes controller style visible to
        # the world model; planner candidates and imitation labels remain raw WoTE.
        target_anchors = wm_rollout_anchors
        if target_anchors is not None:
            target_anchors = target_anchors.detach()
        self._compose_future_bev_targets_from_base(
            targets,
            trajectory_anchors=target_anchors,
            force=True,
        )

        # Expand scene BEV features over K trajectory candidates and inject current ego features.
        flatten_bev_feature_multi_trajs = flatten_bev_feature.unsqueeze(1).repeat(
            1, num_traj, 1, 1
        )  # [B, K, Nq, C]
        flatten_bev_feature_multi_trajs = self._inject_cur_ego_into_bev(
            flatten_bev_feature_multi_trajs, ego_feat, num_traj
        )

        ego_feat = ego_feat.reshape(batch_size * num_traj, 1, -1)  # [B*K, 1, C]
        fut_flatten_bev_feature_multi_trajs = flatten_bev_feature_multi_trajs.reshape(
            batch_size * num_traj, self.num_plan_queries, -1
        )  # [B*K, Nq, C]

        num_iterations = self.num_fut_timestep
        interval = 8 // num_iterations
        fut_ego_feat = ego_feat
        ego_feat_list = [fut_ego_feat]
        bev_feat_list = [fut_flatten_bev_feature_multi_trajs]
        
        
        #NOTE WorldCAP相关
        ctrl_enabled = bool(self.use_controller_wm)
        first_step_only = bool(self.controller_wm_first_step_only)

        # Controller bank tokens are step-invariant when injected at every WM step.
        pre_bank_tokens = None
        if ctrl_enabled and (not first_step_only):
            pre_bank_tokens = self._compute_controller_bank_tokens(
                batch_size,
                num_traj,
                fut_flatten_bev_feature_multi_trajs.device,
            )

        had_rollout_anchors = hasattr(self, '_wm_rollout_anchors')
        prev_rollout_anchors = getattr(self, '_wm_rollout_anchors', None)
        self._wm_rollout_anchors = wm_rollout_anchors
        try:
            for i in range(num_iterations):
                step_bank_tokens = pre_bank_tokens
                if ctrl_enabled and first_step_only:#每一步的都不一样
                    step_bank_tokens = (
                        self._compute_controller_bank_tokens(
                            batch_size,
                            num_traj,
                            fut_flatten_bev_feature_multi_trajs.device,
                        )
                        if i == 0
                        else None
                    )

                #往前推一步,没有显式监督:
                #通过最终 future BEV semantic map loss 和 reward scoring loss 来训练
                fut_ego_feat, fut_flatten_bev_feature_multi_trajs = self._latent_world_model_processing(
                    fut_flatten_bev_feature_multi_trajs,
                    fut_ego_feat,
                    batch_size,
                    num_traj,
                    wm_step=i,
                    controller_bank_tokens=step_bank_tokens,
                )
                fut_flatten_bev_feature_multi_trajs = self._inject_fut_ego_into_bev(
                    fut_flatten_bev_feature_multi_trajs,
                    fut_ego_feat,
                    num_traj,
                    fut_idx=(i + 1) * interval,
                )
                ego_feat_list.append(fut_ego_feat)
                bev_feat_list.append(fut_flatten_bev_feature_multi_trajs)
        finally:
            if had_rollout_anchors:
                self._wm_rollout_anchors = prev_rollout_anchors
            else:
                delattr(self, '_wm_rollout_anchors')

        fut_flatten_bev_feature_multi_trajs = bev_feat_list[-1]
        results["reward_feature"] = self._compute_reward_feature(
            ego_feat_list,
            bev_feat_list,
            batch_size,
            num_traj,
        )

        # Optional BEV visualization for debugging world-model rollouts.
        # save_dir = "/home/zhaodanqi/clone/WoTE/trainingResult/bev-pic"
        # fut_bev_semantic_map = self._process_future_map_NoSample(
        #     fut_flatten_bev_feature_multi_trajs,  # [B*K, Nq, C]
        #     batch_size,
        # )
        # if self.is_eval:
        #     for t, fut_bev in enumerate(bev_feat_list):
        #         # fut_bev: [B*K, Nq, C]
        #         fut_bev_sem_map = self._process_future_map_NoSample(fut_bev, batch_size)
        #         bev_map0 = fut_bev_sem_map[0].detach().cpu()
        #         sem = bev_map0.argmax(dim=0).numpy()
        #         plt.figure(figsize=(5, 5))
        #         plt.imshow(sem, cmap='tab20')
        #         plt.axis('off')
        #         save_path = os.path.join(save_dir, f"future_bev_step_{t}.png")
        #         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        #         plt.close()

        # Future BEV supervision uses the same map heads; skip it if map loss is disabled.
        if (targets is not None) and self.use_map_loss:
            sampled_fut_flatten_bev_feature_multi_trajs = self._sample_future_bev_feature(
                fut_flatten_bev_feature_multi_trajs,
                batch_size,
                num_traj,
                targets=targets,
            )
            results["fut_bev_semantic_map"] = self._process_future_map(
                sampled_fut_flatten_bev_feature_multi_trajs,
                batch_size,
            )

        return results

    def process_trajectory_and_reward(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, torch.Tensor]:
        # Sample controller style once per batch (training only) to improve generalization.
        self._maybe_sample_controller_style_for_batch()
        trajectory_outputs = self.extract_trajectory_feature(features, targets)
        final_results = self.extract_reward_feature(trajectory_outputs, targets)
        return final_results

#ADD 剩下的都是一些功能函数
#把 controller bank 里的所有 controller 轨迹 编码成 world model 可以 cross-attention 的 token
# [N_ctrl, T, 3] + [N_ctrl, T, 3]-> [N_ctrl, 64]-> [N_ctrl, 256]



    def _build_controller_wm_rollout_anchors(self, planner_anchors: torch.Tensor) -> torch.Tensor:
        """Use the sampled controller style's executed trajectories as WM targets."""
        if planner_anchors is None:
            return None
        if not self.use_controller_wm:
            return planner_anchors
        if not hasattr(self, '_active_candidate_exec_trajs') or self._active_candidate_exec_trajs is None:
            return planner_anchors

        squeeze_batch = False
        if planner_anchors.ndim == 3:
            planner_anchors = planner_anchors.unsqueeze(0)
            squeeze_batch = True
        if planner_anchors.ndim != 4:
            raise ValueError(
                "planner_anchors for controller WM rollout must be [B,K,T,3] or [K,T,3], "
                f"got {tuple(planner_anchors.shape)}"
            )

        exe = self._active_candidate_exec_trajs.to(
            device=planner_anchors.device, dtype=planner_anchors.dtype
        )
        if exe.ndim != 3:
            raise ValueError(
                "controller exec bank for WM rollout must be [K,T,3], "
                f"got exec={tuple(exe.shape)}"
            )
        if exe.shape[0] != planner_anchors.shape[1] or exe.shape[1] != planner_anchors.shape[2]:
            raise ValueError(
                "controller exec bank must align with planner candidates for WM rollout: "
                "N must equal planner K and T must match, "
                f"got exec={tuple(exe.shape)} planner={tuple(planner_anchors.shape)}"
            )

        # Candidate executions are simulated in the coordinate frame of the
        # first planner pose. Compose them back into the current ego frame so
        # rollout step 0 corresponds to the first future planner pose rather
        # than incorrectly remaining at the origin.
        local = exe.unsqueeze(0).expand(planner_anchors.shape[0], -1, -1, -1)
        first_pose = planner_anchors[:, :, :1, :3]
        world_x = first_pose[..., 0] + local[..., 0]
        world_y = first_pose[..., 1] + local[..., 1]
        world_yaw = torch.atan2(
            torch.sin(first_pose[..., 2] + local[..., 2]),
            torch.cos(first_pose[..., 2] + local[..., 2]),
        )
        rollout = torch.stack((world_x, world_y, world_yaw), dim=-1)
        return rollout.squeeze(0) if squeeze_batch else rollout

    def _compute_controller_bank_tokens(self, batch_size: int, num_traj: int, device: torch.device) -> torch.Tensor:
        """Encode the active controller bank into tokens for attn_film WM conditioning."""
        if not self.use_controller_wm:
            return torch.zeros((batch_size * num_traj, 1, 256), device=device)

        if not hasattr(self, '_active_ref_trajs') or self._active_ref_trajs is None:
            return torch.zeros((batch_size * num_traj, 1, 256), device=device)
        if not hasattr(self, '_active_exec_trajs') or self._active_exec_trajs is None:
            return torch.zeros((batch_size * num_traj, 1, 256), device=device)

        ref_traj = self._active_ref_trajs.to(device)
        exec_traj = self._active_exec_trajs.to(device)

        if ref_traj.ndim != 3 or exec_traj.ndim != 3:
            raise ValueError(
                "controller bank trajectories must have shape [N_ctrl, T, 3], "
                f"got ref={tuple(ref_traj.shape)} exec={tuple(exec_traj.shape)}"
            )
        if ref_traj.shape[0] != exec_traj.shape[0]:
            raise ValueError(
                "controller ref/exec banks must have the same N_ctrl, "
                f"got ref={ref_traj.shape[0]} exec={exec_traj.shape[0]}"
            )

        # ControllerEmbedding returns one embedding per provided controller trajectory.
        bank_emb64 = self.controller_encoder(ref_traj, exec_traj)  # [N_ctrl, emb_dim]
        bank_tokens = self.ctrl_bank_ln(self.ctrl_bank_proj(bank_emb64))  # [N_ctrl, 256]

        # Expand across batch and candidate trajectories.
        bank_bt = bank_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_ctrl, 256]
        bank_bnt = bank_bt[:, None, :, :].expand(batch_size, num_traj, -1, -1)  # [B, num_traj, N_ctrl, 256]
        return bank_bnt.reshape(batch_size * num_traj, bank_tokens.shape[0], 256)

    def _predict_offset(self, ego_feat: torch.Tensor, flatten_bev_feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the offset for the cluster centers.
        """
        offset_dict = {}
        ego_feat = ego_feat.squeeze(2)  # [batch_size, num_traj, C]
        ego_feat = self.offset_tf_decoder(ego_feat, flatten_bev_feature) # [batch_size, num_traj, C]
        trajectory_offset = self.offset_head(ego_feat)
        trajectory_offset_rewards = self.offset_score_head(ego_feat).squeeze(-1)  # [batch_size, num_traj]
        trajectory_offset_rewards = torch.softmax(trajectory_offset_rewards, dim=-1)
        offset_dict["trajectory_offset"] = trajectory_offset
        offset_dict["trajectory_offset_rewards"] = trajectory_offset_rewards
        return offset_dict

    def _process_backbone_features(self, camera_feature: torch.Tensor, lidar_feature: torch.Tensor):
        """
        Process the backbone network and extract BEV features.
        """
        _, backbone_bev_feature, _ = self._backbone(camera_feature, lidar_feature)#[B, C, H, W]（batch, channel, height, width）
        bev_feature = self._bev_downscale(backbone_bev_feature).flatten(-2, -1).permute(0, 2, 1)
        flatten_bev_feature = bev_feature + self._keyval_embedding.weight[None, :, :]#给每个位置加入可学习的 embedding。[B, H*W, C]
        return backbone_bev_feature, flatten_bev_feature

    def _get_ego_status_feature(self, status_feature: torch.Tensor) -> torch.Tensor:
        """
        Obtain the encoded ego vehicle status features.
        """
        status_encoding = self._status_encoding(status_feature)  # [batch_size, C]
        ego_status_feat = status_encoding[:, None, :]  # [batch_size, 1, C]
        return ego_status_feat

    def _get_trajectory_anchors_feat(self, trajectory_anchors, batch_size: int):
        """
        Get the features of the cluster centers and expand them to the batch size.
        """
        num_traj = trajectory_anchors.shape[1] # [bz, num_traj, 时间步T,状态维度D] 
        device = trajectory_anchors.device
        init_traj = trajectory_anchors.reshape(batch_size, num_traj, -1).to(device)  # [bz, num_traj, D]  init_traj把所有时间拼接成一个长向量；
        trajectory_anchors_feat = self.mlp_planning_vb(init_traj)  # [bz, num_traj, hidden_dim]
        trajectory_anchors_feat = self.cluster_encoder(trajectory_anchors_feat)  # [bz, num_traj, encoded_dim]   #对这些轨迹特征进行非线性编码或聚类嵌入。
        return trajectory_anchors_feat, num_traj

    def _concatenate_ego_and_traj_features(self, ego_status_feat: torch.Tensor, trajectory_anchors_feat: torch.Tensor) -> torch.Tensor:
        """
        Concatenate ego features with trajectory features and encode.
        """
        # Repeat ego_status_feat to match the number of trajectories
        ego_status_feat = ego_status_feat.repeat(1, trajectory_anchors_feat.shape[1], 1)  # [batch_size, num_traj, C]  #在轨迹条数维度上重复
        # Concatenate
        ego_feat = torch.cat([ego_status_feat, trajectory_anchors_feat], dim=-1)  # [batch_size, num_traj, C + encoded_dim]  #add ego_stattus_feature和anchor_feature
        # Encode
        ego_feat = self.encode_ego_feat_mlp(ego_feat)  # [batch_size, num_traj, C']
        ego_feat = ego_feat.unsqueeze(-2)  # [batch_size, num_traj, 1, C'] #新增倒数第二个维度
        return ego_feat

    def _inject_cur_ego_into_bev(self, scene_bev_feature: torch.Tensor, ego_feat: torch.Tensor, num_traj: int, h = 8, w = 8) -> torch.Tensor:
        """
        Inject the ego feature into the BEV map.
        """
        bz = scene_bev_feature.shape[0]
        scene_bev_feature = scene_bev_feature.permute(0, 1, 3, 2).reshape(bz*num_traj, -1, h, w)  # [batch_size, num_traj, C, H*W]
        ego_feat = ego_feat.squeeze(2).reshape(bz*num_traj, -1)  # [batch_size*num_traj, C]
         # [batch_size*num_traj, 2]
        coors = torch.zeros(bz*num_traj, 2).to(ego_feat.device)
        scene_bev_feature = self.inject_ego_feat_to_bev_map(scene_bev_feature, ego_feat, coors)
        scene_bev_feature = scene_bev_feature.view(bz, num_traj, -1, h * w)  # [batch_size, num_traj, C, H*W]
        scene_bev_feature = scene_bev_feature.permute(0, 1, 3, 2)  # [batch_size, num_traj, H*W, C]
        return scene_bev_feature
    
    def _inject_fut_ego_into_bev(self, scene_bev_feature: torch.Tensor, ego_feat: torch.Tensor, num_traj: int, fut_idx = 8, h = 8, w = 8) -> torch.Tensor:
        """
        Inject the ego feature into the BEV map.
        """
        bz = int(scene_bev_feature.shape[0] // num_traj)
        scene_bev_feature = scene_bev_feature.permute(0, 2, 1).reshape(bz*num_traj, -1, h, w)  # [batch_size, num_traj, C, H*W]
        ego_feat = ego_feat.squeeze(2).reshape(bz*num_traj, -1)  # [batch_size*num_traj, C]
        # [batch_size*num_traj, 2]
        # Future ego coordinates follow the controller-executed rollout used by
        # the world model target. Planner candidates themselves are still kept
        # separate for final reward scoring and trajectory selection.
        rollout_anchors = getattr(self, '_wm_rollout_anchors', None)
        if rollout_anchors is not None:
            anchors = rollout_anchors.to(device=ego_feat.device, dtype=ego_feat.dtype)
            coors = anchors[:, :, fut_idx - 1, :2].reshape(bz * num_traj, -1)
        else:
            coors = self.trajectory_anchors[:, fut_idx-1, :2].to(ego_feat.device).unsqueeze(0).repeat(bz, 1, 1).reshape(bz*num_traj, -1)

        scene_bev_feature = self.inject_ego_feat_to_bev_map(scene_bev_feature, ego_feat, coors)
        scene_bev_feature = scene_bev_feature.view(bz*num_traj, -1, h * w)  # [batch_size, num_traj, C, H*W]
        scene_bev_feature = scene_bev_feature.permute(0, 2, 1)  # [batch_size*num_traj, H*W, C]
        return scene_bev_feature

    def _latent_world_model_processing(
        self,
        flatten_bev_feature_multi_trajs: torch.Tensor,
        ego_feat: torch.Tensor,
        batch_size: int,
        num_traj: int,
        wm_step: int = -1,
        controller_bank_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process features through the latent world model.
        """
        # Concatenate ego_feat and flatten_bev_feature_multi_trajs
        # Assume concatenation along the feature dimension
        scene_feature = torch.cat([ego_feat, flatten_bev_feature_multi_trajs], dim=1) 
        # print(f"💚shape of initial scene_feature: {scene_feature.shape}")#([4096, 65, 256])

        # Add positional embedding
        #learnable token position embedding
        scene_position_embedding = self.scene_position_embedding.weight.unsqueeze(0).expand(batch_size * num_traj, -1, -1)
        # print(f"💚shape of scene_position_embedding: {scene_position_embedding.shape}")#([4096, 65, 256])
        scene_feature = scene_feature + scene_position_embedding    #没有拼接、没有广播扩展，只是简单地把每个元素加上对应位置的值。这个是逐元素相加
        # print(f"💚shape of scene_feature: {scene_feature.shape}")#torch.Size([4096, 65, 256])

        # Controller-aware transition: controller affects planning only through
        # the BEV latent world model.
        if self.use_controller_wm:
            fusion = self.controller_wm_fusion
            target = self.controller_wm_token_scope
            first_step_only = bool(self.controller_wm_first_step_only)
            step_i = int(wm_step) if wm_step is not None else -1
            inject_now = (not first_step_only) or (step_i in {0, -1})

            if inject_now:
                if fusion != 'attn_film':
                    raise ValueError(
                        "WoTE_model.py has been trimmed to the 5.11+ controller path; "
                        f"controller_wm_fusion must be 'attn_film', got {fusion!r}."
                    )

                bank_tokens = controller_bank_tokens
                if bank_tokens is None:
                    bank_tokens = self._compute_controller_bank_tokens(batch_size, num_traj, scene_feature.device)  # [B*num_traj, N_ctrl, 256]

                #选择注入范围:只有egotoken还是所有scene token
                if target in {'ego', 'ego_only'}:
                    tok = scene_feature[:, 0:1, :]
                    ctrl_ctx, _ = self.ctrl_fuse_attn(query=tok, key=bank_tokens, value=bank_tokens, need_weights=False)
                    scale = self.ctrl_wm_film_scale(ctrl_ctx)
                    shift = self.ctrl_wm_film_shift(ctrl_ctx)
                    scene_feature[:, 0:1, :] = self.ctrl_wm_film_ln(tok * (1.0 + scale) + shift)
                else:
                    ctrl_ctx, _ = self.ctrl_fuse_attn(query=scene_feature, key=bank_tokens, value=bank_tokens, need_weights=False)
                    scale = self.ctrl_wm_film_scale(ctrl_ctx)
                    shift = self.ctrl_wm_film_shift(ctrl_ctx)
                    scene_feature = self.ctrl_wm_film_ln(scene_feature * (1.0 + scale) + shift)

        # Reshape to fit the latent world model
        fut_scene_feature = self.latent_world_model(scene_feature)  
        # print(f"💚shape of fut_scene_feature: {fut_scene_feature.shape}")#shape of fut_scene_feature: ([4096, 65, 256])
        fut_ego_feat = fut_scene_feature[:, 0:1]
        # print(f"💚shape of fut_ego_feat: {fut_ego_feat.shape}")#([4096, 1, 256])
        fut_flatten_bev_feature_multi_trajs = fut_scene_feature[:, 1:]
        return fut_ego_feat, fut_flatten_bev_feature_multi_trajs

    def _compute_reward_feature(self, fut_ego_feat_list, fut_flatten_bev_feature_multi_trajs_list, batch_size, num_traj, h=8, w=8) -> torch.Tensor:
        """
        Compute the scoring features.
        它把 world model 多步 rollout 得到的 BEV 特征序列和 ego 特征序列，压缩/拼接成每条候选轨迹的一个 256 维 reward_feature
        """
        bev_feat_list = []
        for bev_feat in  fut_flatten_bev_feature_multi_trajs_list:
            bev_feat = bev_feat.view(batch_size * num_traj, h, w, -1) 
            bev_feat_list.append(bev_feat)
        all_bev_feature = torch.cat(bev_feat_list, dim=-1)  # [batch_size*num_traj, H, W, C1 + C2]
        all_bev_feature = all_bev_feature.permute(0, 3, 1, 2)  # [batch_size*num_traj, C1 + C2, H, W]
        
        # Apply convolution network
        # 也就是把整个 BEV 空间特征压缩成一个全局 token。
        reward_conv_output = self.reward_conv_net(all_bev_feature).squeeze(-1).permute(0, 2, 1)  # [batch_size*num_traj, 1, C_conv]
        
        # Prepare scoring features
        cat_reward_feature = torch.cat(fut_ego_feat_list + [reward_conv_output], dim=1)  # [batch_size*num_traj, 3, C_total]
        cat_reward_feature = cat_reward_feature.reshape(batch_size * num_traj, -1)  # [batch_size*num_traj, 3*C_total]
        
        # Apply scoring head
        reward_feature = self.reward_cat_head(cat_reward_feature)  # [batch_size*num_traj, 256]
        reward_feature = reward_feature.view(batch_size, num_traj, -1)  # [batch_size, num_traj, 256]
        return reward_feature

    def _process_agent(self, batch_size: int, scene_bev_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process agent.
        """
        agent_query = self.agent_query_embedding.weight[None, :, :].repeat(batch_size, 1, 1)  # [batch_size, num_agents, hidden_dim]
        agent_query_out = self.agent_tf_decoder(agent_query, scene_bev_feature)  # [batch_size, num_agents, hidden_dim]
        agents = self.agent_head(agent_query_out)  # dict containing 'agent_states' and 'agent_labels'
        return agents, agent_query_out

    def _process_map(self, flatten_bev_feature: torch.Tensor, batch_size, h=8, w=8) -> torch.Tensor:
        """
        Process map.
        """
        # Adjust dimensions
        # flatten_bev_feature = flatten_bev_feature.mean(dim=1)  # [batch_size, H*W, C]
        flatten_bev_feature = flatten_bev_feature.permute(0, 2, 1)  # [batch_size, C, H*W]
        bz, C, num_plan_queries = flatten_bev_feature.shape
        flatten_bev_feature = flatten_bev_feature.reshape(bz, C, h, w)  # [batch_size, C, H, W]
        flatten_bev_feature = self._bev_upscale(flatten_bev_feature)  # [batch_size, 2* C, H, W]

        upsampled_bev_feature = self.bev_upsample_head(flatten_bev_feature)  # [batch_size, C, H', W']
        bev_semantic_map = self.bev_semantic_head(upsampled_bev_feature)  # [batch_size, num_classes, H', W']
        return bev_semantic_map, upsampled_bev_feature

    def _sample_future_bev_feature(self, fut_scene_feature: torch.Tensor, batch_size: int, num_traj: int, targets) -> torch.Tensor:
        """
        Prepare future BEV features.
        """
        new_scene_bev_feature_pos_embed = self.new_scene_bev_feature_pos_embed.weight[None, :, :].repeat(batch_size * self.num_sampled_trajs, 1, 1)  # [batch_size*num_sampled_trajs, num_plan_queries, embedding_dim]
        new_scene_bev_feature = fut_scene_feature.view(batch_size, num_traj, self.num_plan_queries, -1)  # [batch_size, num_traj, num_plan_queries, C_new]
        B, T, H, W = new_scene_bev_feature.shape  # B=2, T=256, H=64, W=256
        sampled_trajs_index = targets['sampled_trajs_index']
        K = sampled_trajs_index.shape[1]  # K=3
        batch_indices = torch.arange(B, device=sampled_trajs_index.device).unsqueeze(1).repeat(1, K)  # shape: [B, K]
        selected_features = new_scene_bev_feature[batch_indices, sampled_trajs_index]  # shape: [B, K, H, W]
        new_scene_bev_feature = selected_features.view(batch_size * self.num_sampled_trajs, self.num_plan_queries, -1)  # [batch_size*num_sampled_trajs, num_plan_queries, C_new]
        new_scene_bev_feature_with_pos = new_scene_bev_feature + new_scene_bev_feature_pos_embed  # [batch_size*num_sampled_trajs, num_plan_queries, C_new]
        return new_scene_bev_feature_with_pos

    def _process_future_agents(self, new_scene_bev_feature_with_pos: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Process future agent features.
        """
        fut_agent_query = self.fut_agent_query_embedding.weight[None, :, :].repeat(batch_size * self.num_sampled_trajs, 1, 1)  # [batch_size*num_sampled_trajs, num_agents, hidden_dim]
        fut_agent_query_out = self.fut_agent_tf_decoder(fut_agent_query, new_scene_bev_feature_with_pos)  # [batch_size*num_sampled_trajs, num_agents, hidden_dim]
        fut_agents_dict = self.fut_agent_head(fut_agent_query_out)  # dict containing 'agent_states' and 'agent_labels'
        fut_agents_dict = {
            'fut_agent_states': fut_agents_dict.pop('agent_states'),
            'fut_agent_labels': fut_agents_dict.pop('agent_labels')
        }
        return fut_agents_dict

    def _process_future_map(self, new_scene_bev_feature_with_pos: torch.Tensor, batch_size: int, h=8, w=8) -> torch.Tensor:
        """
        Process future BEV semantic map.
        """
        # Adjust dimensions   
        fut_bev_feature = new_scene_bev_feature_with_pos.permute(0, 2, 1)  # [batch_size*num_sampled_trajs, C_new, num_plan_queries]
        # print(f"💟1_shape of futBevFeature: {fut_bev_feature.shape}")#💟1_shape of futBevFeature: torch.Size([256, 256, 64])==>train([4096, 256, 64])
        fut_bev_feature = fut_bev_feature.reshape(batch_size * self.num_sampled_trajs, -1, h, w)  # [batch_size*num_sampled_trajs, C_new, 4, 8]     ==>train([4096, 256, 8, 8])
        # print(f"💟2_shape of futBevFeature: {fut_bev_feature.shape}")#💟2_shape of futBevFeature: torch.Size([1, 65536, 8, 8])
        fut_bev_feature = self._bev_upscale(fut_bev_feature)  # [batch_size*num_sampled_trajs, C_upscaled, H', W']
        # print(f"💟3_shape of futBevFeature: {fut_bev_feature.shape}")
        upsampled_fut_bev_feature = self.bev_upsample_head(fut_bev_feature)  # [batch_size*num_sampled_trajs, C_upscaled, H'', W'']    train==》([4096, 512, 8, 8])
        # print(f"💟4_shape of futBevFeature: {upsampled_fut_bev_feature.shape}")
        fut_bev_semantic_map = self.bev_semantic_head(upsampled_fut_bev_feature)  # [batch_size*num_sampled_trajs, num_classes, H'', W'']        train==》[4096, 64, 64, 64])
        # print(f"💟5_shape of futBevFeature: {fut_bev_semantic_map.shape}")#  train==》 ([16, 8, 128, 256])
        return fut_bev_semantic_map
    
    
    # Optional debug map renderer without sampled-candidate filtering.
    def _process_future_map_NoSample(self, new_scene_bev_feature_with_pos: torch.Tensor, batch_size: int, h=8, w=8) -> torch.Tensor:
        """
        Process future BEV semantic map.
        💟1_shape of futBevFeature: torch.Size([4096, 256, 64])
        💟2_shape of futBevFeature: torch.Size([4096, 256, 8, 8])
        💟3_shape of futBevFeature: torch.Size([4096, 512, 8, 8])
        💟4_shape of futBevFeature: torch.Size([4096, 64, 64, 64])
        💟5_shape of futBevFeature: torch.Size([4096, 8, 128, 256])
        """
        # Adjust dimensions   
        fut_bev_feature = new_scene_bev_feature_with_pos.permute(0, 2, 1)  # [batch_size*num_sampled_trajs, C_new, num_plan_queries]
        # print(f"💟1_shape of futBevFeature: {fut_bev_feature.shape}")#💟1_shape of futBevFeature: torch.Size([256, 256, 64])
        fut_bev_feature = fut_bev_feature.reshape(batch_size * self.num_sampled_trajs_NoSample, -1, h, w)  # [batch_size*num_sampled_trajs, C_new, 4, 8]
        # print(f"💟2_shape of futBevFeature: {fut_bev_feature.shape}")#💟2_shape of futBevFeature: torch.Size([256, 256, 8, 8])
        fut_bev_feature = self._bev_upscale(fut_bev_feature)  # [batch_size*num_sampled_trajs, C_upscaled, H', W']
        # print(f"💟3_shape of futBevFeature: {fut_bev_feature.shape}")#💟3_shape of futBevFeature: torch.Size([256, 512, 8, 8])
        upsampled_fut_bev_feature = self.bev_upsample_head(fut_bev_feature)  # [batch_size*num_sampled_trajs, C_upscaled, H'', W'']
        # print(f"💟4_shape of futBevFeature: {upsampled_fut_bev_feature.shape}")#💟4_shape of futBevFeature: torch.Size([256, 64（通道由512变成了64）, 64, 64])
        fut_bev_semantic_map = self.bev_semantic_head(upsampled_fut_bev_feature) 
        # print(f"💟5_shape of futBevFeature: {fut_bev_semantic_map.shape}")#💟5_shape of futBevFeature: torch.Size([256, 8, 128, 256])
        return fut_bev_semantic_map
    # BEV ego-feature injection utility.
    def inject_ego_feat_to_bev_map(self, bev_map, new_features, delta_x_y, H=8, W=8):
        """`
        Add a new feature vector in batch to the corresponding location in the BEV feature map, affecting the four pixels around each position.

        Parameters:
        - bev_map (torch.Tensor): BEV feature map with shape (B, C, H, W)
        - delta_x_y (torch.Tensor): x and y coordinates in the ego coordinate system, shape (B, 2)
        - new_features (torch.Tensor): Feature vectors to be added, shape (B, C)
        - H (int): Height (in pixels) of the BEV feature map
        - W (int): Width (in pixels) of the BEV feature map

        Returns:
        - updated_bev_map (torch.Tensor): Updated BEV feature map, shape (B, C, H, W)
        """
        B, C, H_map, W_map = bev_map.shape
        assert H_map == H and W_map == W, f"BEV map dimensions must be ({H}, {W}), but got ({H_map}, {W_map})"
        assert new_features.shape == (B, C), "new_features must have shape (B, C)"

        device = bev_map.device
        dtype = bev_map.dtype

        delta_x, delta_y = delta_x_y[:, 0], delta_x_y[:, 1]
        # Calculate the pixel-to-meter ratio
        pixel_per_meter_x = H / 32.0  # 32 meters covered by H pixels
        pixel_per_meter_y = W / 64.0  # 64 meters covered by W pixels

        # Convert ego coordinates to floating point pixel indices
        h_idx = delta_x * pixel_per_meter_x  # (B,)
        w_idx = delta_y * pixel_per_meter_y + (W / 2.0)  # Origin at (0, W/2)

        # Get four nearby integer pixel indices
        h0 = torch.floor(h_idx).long()  # (B,)
        w0 = torch.floor(w_idx).long()  # (B,)
        h1 = h0 + 1
        w1 = w0 + 1

        # Compute distance weights
        dh = h_idx - h0.float()  # (B,)
        dw = w_idx - w0.float()  # (B,)

        w00 = (1 - dh) * (1 - dw)  # (B,)
        w01 = (1 - dh) * dw
        w10 = dh * (1 - dw)
        w11 = dh * dw

        # Stack indices and weights for all four nearby pixels
        # Each point contributes to four positions
        h_indices = torch.stack([h0, h0, h1, h1], dim=1)  # (B, 4)
        w_indices = torch.stack([w0, w1, w0, w1], dim=1)  # (B, 4)
        weights = torch.stack([w00, w01, w10, w11], dim=1)  # (B, 4)

        # Create batch indices
        batch_indices = torch.arange(B, device=device).view(B, 1).repeat(1, 4)  # (B, 4)

        # Flatten contributions
        h_indices_flat = h_indices.reshape(-1)  # (B*4,)
        w_indices_flat = w_indices.reshape(-1)  # (B*4,)
        weights_flat = weights.reshape(-1)  # (B*4,)
        batch_indices_flat = batch_indices.reshape(-1)  # (B*4,)

        # Create validity mask to ensure indices are within range
        valid = (h_indices_flat >= 0) & (h_indices_flat < H) & (w_indices_flat >= 0) & (w_indices_flat < W)  # (B*4,)

        if not valid.any():
            print("No valid indices")
            return bev_map

        # Filter out invalid indices and weights
        h_indices_valid = h_indices_flat[valid]  # (M,)
        w_indices_valid = w_indices_flat[valid]  # (M,)
        weights_valid = weights_flat[valid].unsqueeze(1)  # (M, 1)
        batch_indices_valid = batch_indices_flat[valid]  # (M,)

        # Get corresponding feature vectors and weights
        # Repeat each new_feature four times corresponding to the four weights
        new_features_expanded = new_features[batch_indices_valid]  # (M, C)
        weighted_features = new_features_expanded * weights_valid  # (M, C)

        # Compute linear indices
        # linear_index = b * C * H * W + c * H * W + h * W + w
        # Compute indices for each channel using broadcasting
        c_indices = torch.arange(C, device=device).view(1, C).repeat(weighted_features.shape[0], 1)  # (M, C)
        linear_indices = (batch_indices_valid.unsqueeze(1) * C * H * W) + (c_indices * H * W) + (h_indices_valid.unsqueeze(1) * W) + w_indices_valid.unsqueeze(1)  # (M, C)
        linear_indices = linear_indices.reshape(-1)  # (M*C,)

        # Flatten weighted features to (M*C,)
        weighted_features_flat = weighted_features.reshape(-1)  # (M*C,)

        # Flatten bev_map to (B*C*H*W,)
        bev_map_flat = bev_map.reshape(-1)  # (B*C*H*W,)

        # Use index_add_ to accumulate weighted features at corresponding positions
        bev_map_flat.index_add_(0, linear_indices, weighted_features_flat)

        # Reshape the BEV feature map back to (B, C, H, W)
        updated_bev_map = bev_map_flat.view(B, C, H, W)

        return updated_bev_map
    
    def select_best_trajectory(self, final_rewards, trajectory_anchors, batch_size):
        best_trajectory_idx = torch.argmax(final_rewards, dim=-1)  # [B]
        if trajectory_anchors.ndim == 3:
            trajectory_anchors = trajectory_anchors.unsqueeze(0).expand(batch_size, -1, -1, -1)
        B, K, T, D = trajectory_anchors.shape
        idx = best_trajectory_idx.view(B, 1, 1, 1).expand(B, 1, T, D)
        poses = trajectory_anchors.gather(dim=1, index=idx).squeeze(1)  # [B, T, D]
        return poses

    def forward_test(self, features, targets=None) -> Dict[str, torch.Tensor]:
        # Extract scene feature encoding
        self.is_eval = True
        encoder_results = self.process_trajectory_and_reward(features)
        cluster_feaure = encoder_results["reward_feature"]
        batch_size = cluster_feaure.shape[0]

        # Reward each trajectory using MLP head
        im_rewards = self.reward_head(cluster_feaure).squeeze(-1)  # Shape: [batch_size, 256]
        im_rewards_softmax = torch.softmax(im_rewards, dim=-1)  # Shape: [batch_size, *, 256]

        # Reward each trajectory using the additional metric reward heads if use_sim_reward is enabled
        sim_rewards = [reward_head(cluster_feaure) for reward_head in self.sim_reward_heads]
        sim_rewards = [reward.sigmoid() for reward in sim_rewards]  # Apply sigmoid to each metric reward
        final_rewards = self.weighted_reward_calculation(im_rewards_softmax, sim_rewards)

        # Select the trajectory
        offset = encoder_results['trajectory_offset']  # [B, 256, 8, 3]
        base_anchors = self.trajectory_anchors.to(device=offset.device, dtype=offset.dtype).unsqueeze(0).expand(batch_size, -1, -1, -1)
        trajectory_anchors = base_anchors + offset
        trajectory_anchors_ori = base_anchors
        poses = self.select_best_trajectory(final_rewards, trajectory_anchors, batch_size)
        # print(f"💙💙💙selected best trajectory poses shape: {poses.shape}") 
        # 💙💙💙selected best trajectory poses shape: torch.Size([1, 8, 3])
        
#NOTE:改成多模态轨迹的可视化效果！！！
        results = {
            "trajectory": poses,#[batch_size, 8, 3]#得分最高的那一条
            "final_rewards": final_rewards,#256条轨迹的最终得分
            "trajectoryAnchor": trajectory_anchors_ori.squeeze(0) if batch_size == 1 else trajectory_anchors_ori,#[num_traj, 8, 3]
            "all_trajectory": trajectory_anchors.squeeze(0) if batch_size == 1 else trajectory_anchors,#预测的所有轨迹
            "im_rewards": im_rewards_softmax,#256条轨迹的imitation 得分
        }
        return results

    def forward_train(self, features, targets=None) -> Dict[str, torch.Tensor]:
        self.is_eval = False
        # Extract scene feature encoding  --> 前推forward得到轨迹和奖励的feature
        result = {}
        encoder_results = self.process_trajectory_and_reward(features, targets)#来自:extract_reward_feature;
        cluster_feaure = encoder_results.pop("reward_feature")#

        # Reward each trajectory using MLP head-->将奖励feature映射成为erward数值;
        im_rewards = self.reward_head(cluster_feaure).squeeze(-1)  # Shape: [batch_size, 256]
        im_rewards_softmax = torch.softmax(im_rewards, dim=-1)  # Shape: [batch_size, *, 256]
        result["im_rewards"] = im_rewards_softmax

        result["trajectory_anchors"] = self.trajectory_anchors
        # Reward each trajectory using the additional metric reward heads if use_sim_reward is enabled
        sim_rewards = [sim_reward_head(cluster_feaure) for sim_reward_head in self.sim_reward_heads]
        sim_rewards = torch.cat(sim_rewards, dim=-1).permute(0, 2, 1).sigmoid()  # Concatenate metric rewards
        result["sim_rewards"] = sim_rewards

        # NOTE: loss computation (WoTE_loss.py) expects offset-related keys regardless of
        # whether agent/map losses are enabled. Always return encoder_results.
        result.update(encoder_results)

        return result

    def weighted_reward_calculation(self, im_rewards, sim_rewards) -> torch.Tensor:
        """
        Calculate the final reward for each trajectory based on the given weights.

        Args:
            im_rewards (torch.Tensor): Imitation rewards for each trajectory. Shape: [batch_size, num_traj]
            sim_rewards (List[torch.Tensor]): List of metric rewards for each trajectory. Each tensor shape: [batch_size, num_traj]
            w (List[float]): List of weights for combining the rewards.

        Returns:
            torch.Tensor: Final weighted reward for each trajectory. Shape: [batch_size, num_traj]
        """
        assert len(sim_rewards) == 5, "Expected 4 metric rewards: S_NC, S_DAC, S_TTC, S_EP, S_COMFORT"
        # Extract metric rewards
        w = self.reward_weights
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = sim_rewards
        S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = S_NC.squeeze(-1), S_DAC.squeeze(-1), S_EP.squeeze(-1), S_TTC.squeeze(-1), S_COMFORT.squeeze(-1)
        #self.metric_keys = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort']
        # Calculate assembled cost based on the provided formula
        assembled_cost = (
            w[0] * torch.log(im_rewards) +
            w[1] * torch.log(S_NC) +
            w[2] * torch.log(S_DAC) +
            w[3] * torch.log(5 * S_TTC + 2 * S_COMFORT + 5 * S_EP)
        )
        return assembled_cost
#ADD 
    #==================== Utilities for future BEV targets (runtime compose) ====================
    def _compose_future_bev_targets_from_base(
        self,
        targets: Dict[str, torch.Tensor],
        trajectory_anchors: Optional[torch.Tensor] = None,
        force: bool = False,
    ):
        """
        Rebuild future BEV semantic maps from cached base map using active candidate
        trajectories and cached indices + frame interval. No-op if already present unless force=True.
        Expects keys: 'fut_bev_semantic_map_base', 'sampled_trajs_index', 'frame_interval'.
        Produces: 'fut_bev_semantic_map' as stacked maps for sampled trajectories.
        """
        if targets is None:
            return
        if "fut_bev_semantic_map_base" not in targets:
            return
        if ("fut_bev_semantic_map" in targets) and (not force):
            return
        if trajectory_anchors is None:
            trajectory_anchors = self.trajectory_anchors
        if trajectory_anchors.ndim == 3:
            candidate_anchors = trajectory_anchors
            batched_anchors = None
        elif trajectory_anchors.ndim == 4:
            candidate_anchors = None
            batched_anchors = trajectory_anchors
        else:
            return

        base_map: torch.Tensor = targets["fut_bev_semantic_map_base"]  # [H, W] 或 [B, H, W]
        sampled_idx = targets.get("sampled_trajs_index", None)
        frame_interval = targets.get("frame_interval", None)
        if sampled_idx is None or frame_interval is None:
            return  # 信息不足；保持原样

        # 规范类型
        if isinstance(sampled_idx, np.ndarray):
            sampled_idx = torch.from_numpy(sampled_idx)
        if not torch.is_tensor(sampled_idx):
            sampled_idx = torch.tensor(sampled_idx, dtype=torch.long)
        anchor_device = trajectory_anchors.device
        sampled_idx = sampled_idx.to(anchor_device)

        # 处理不同维度的 batch 索引
        device = base_map.device
        # 归一化 frame_interval，支持标量或按 batch 提供
        if torch.is_tensor(frame_interval):
            fi_tensor = frame_interval.detach().cpu().long().view(-1)  # [B] 或 [1]
        else:
            fi_tensor = torch.tensor([int(frame_interval)], dtype=torch.long)
        # 限制范围到 [0, T-1]（T 来自 planner anchors 的时间步数）
        T = int(trajectory_anchors.shape[-2])
        fi_tensor = torch.clamp(fi_tensor, min=0, max=max(0, T - 1))

        # 单样本：sampled_idx 为 1D
        if sampled_idx.ndim == 1:
            # 单样本：K 个轨迹，frame_interval 取 fi_tensor[0]
            fi = int(fi_tensor[0].item())
            try:
                if batched_anchors is not None:
                    anchors = batched_anchors[0].index_select(0, sampled_idx)  # [K, T, 3]
                else:
                    anchors = candidate_anchors.index_select(0, sampled_idx)  # [K, T, 3]
            except Exception:
                return
            fut_maps = []
            for i in range(int(anchors.shape[0])):
                off = anchors[i, fi]
                dx, dy, dyaw = float(off[0].item()), float(off[1].item()), float(off[2].item())
                ego_box = [dx, dy, 0.0, 4.0, 2.0, 1.8, dyaw]
                src_map = base_map.clone() if base_map.ndim == 2 else base_map[0].clone()
                fut_map = self._add_ego_box_to_bev_map(src_map, ego_box)
                fut_maps.append(fut_map.to(device))
            if len(fut_maps) > 0:
                targets["fut_bev_semantic_map"] = torch.stack(fut_maps).unsqueeze(0)  # [1, K, H, W]
            return

        # 批量：sampled_idx 为 [B, K]
        if sampled_idx.ndim == 2:
            B, K = sampled_idx.shape
            # 对齐 base_map 维度
            if base_map.ndim == 2:
                base_map = base_map.unsqueeze(0).repeat(B, 1, 1)
            fut_maps_batched = []  # [B, K, H, W]
            # fi_tensor: [1] 或 [B]
            if fi_tensor.numel() == 1:
                fi_list = [int(fi_tensor[0].item())] * B
            else:
                # 若长度不匹配，回退到首元素
                fi_list = [int(fi_tensor[min(b, fi_tensor.shape[0]-1)].item()) for b in range(B)]
            # 限制范围
            fi_list = [max(0, min(f, T - 1)) for f in fi_list]

            for b in range(B):
                idx_b = sampled_idx[b]
                fi_b = fi_list[b]
                try:
                    if batched_anchors is not None:
                        anchors_b = batched_anchors[b].index_select(0, idx_b)  # [K, T, 3]
                    else:
                        anchors_b = candidate_anchors.index_select(0, idx_b)  # [K, T, 3]
                except Exception:
                    return
                fut_maps_b = []
                for i in range(int(anchors_b.shape[0])):
                    off = anchors_b[i, fi_b]
                    dx, dy, dyaw = float(off[0].item()), float(off[1].item()), float(off[2].item())
                    ego_box = [dx, dy, 0.0, 4.0, 2.0, 1.8, dyaw]
                    fut_map = self._add_ego_box_to_bev_map(base_map[b].clone(), ego_box)
                    fut_maps_b.append(fut_map.to(device))
                if len(fut_maps_b) > 0:
                    fut_maps_batched.append(torch.stack(fut_maps_b))  # [K, H, W]
                else:
                    fut_maps_batched.append(torch.empty((K,) + base_map.shape[1:], device=device))

            if len(fut_maps_batched) > 0:
                fut_b = torch.stack(fut_maps_batched)  # [B, K, H, W]
                targets["fut_bev_semantic_map"] = fut_b
            return

        # 其他不支持的形状：跳过
        return

    def _coords_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)

    def _compute_ego_box_mask(self, box_value) -> np.ndarray:
        x, y, heading = box_value[0], box_value[1], box_value[-1]
        box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
        try:
            from nuplan.common.actor_state.oriented_box import OrientedBox
            from nuplan.common.actor_state.state_representation import StateSE2
            import cv2
        except Exception:
            # If geometry libs not available, approximate by a small axis-aligned box in pixel space
            H, W = self._config.bev_semantic_frame
            mask = np.zeros((W, H), dtype=np.uint8)
            px_py = self._coords_to_pixel(np.array([[x, y]]))
            px, py = int(px_py[0, 0]), int(px_py[0, 1])
            rx, ry = max(0, px - 1), max(0, py - 1)
            lx, uy = min(W - 1, px + 1), min(H - 1, py + 1)
            mask[rx:lx + 1, ry:uy + 1] = 255
            mask = np.rot90(mask)[::-1]
            return mask > 0

        agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
        exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
        exterior = self._coords_to_pixel(exterior)
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    def _add_ego_box_to_bev_map(self, bev_semantic_map: torch.Tensor, ego_box) -> torch.Tensor:
        mask = self._compute_ego_box_mask(ego_box)
        # torch indexing with numpy mask
        bev_np = bev_semantic_map.cpu().numpy()
        bev_np[mask] = self._config.ego_box_map_idx
        return torch.tensor(bev_np, dtype=bev_semantic_map.dtype, device=bev_semantic_map.device)

class AgentHead(nn.Module):
    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:

        agent_states = self._mlp_states(agent_queries) # agent_states: torch.Size([32, 30, 5])
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}
    
class BEVUpsampleHead(nn.Module):
    def __init__(self, config, channel=64, c5_chs=512):
        super(BEVUpsampleHead, self).__init__()
        self.config = config
        self.relu = nn.ReLU(inplace=True)

        # Initialize upsampling and convolution layers上采样层（放大特征图的分辨率）
        self.upsample = nn.Upsample(
            scale_factor=self.config.bev_upsample_factor, mode="bilinear", align_corners=False#self.config.bev_upsample_factor 图片放大倍数
        )
        if config.lidar_min_x == 0.:
            self.upsample2 = nn.Upsample(#与前一个不同，这里不是用 scale_factor，而是直接指定输出尺寸。size=(height, width) 明确告诉网络要输出固定大小的 BEV 特征图
                    size=(
                        self.config.lidar_resolution_height // (2 * self.config.bev_down_sample_factor),
                        self.config.lidar_resolution_width // self.config.bev_down_sample_factor,
                    ),
                    mode="bilinear",
                    align_corners=False,
            )
        else:
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_resolution_height // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

        self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

        # Lateral connection
        self.c5_conv = nn.Conv2d(
            c5_chs, channel, (1, 1)
        )

    def forward(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

class RewardConvNet(nn.Module):
    def __init__(self, input_channels: int = 512, conv1_out_channels: int = 256, conv2_out_channels: int = 256):
        """
        Initialize RewardConvNet.

        Args:
            input_channels (int): Number of channels in the input feature map. Default is 512.
            conv1_out_channels (int): Number of output channels for the first convolution. Default is 256.
            conv2_out_channels (int): Number of output channels for the second convolution. Default is 256.
        """
        super(RewardConvNet, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=conv1_out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)  # Batch normalization for the first layer
        self.relu1 = nn.ReLU()
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels, 
            out_channels=conv2_out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(conv2_out_channels)  # Batch normalization for the second layer
        self.relu2 = nn.ReLU()
        
        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Input feature map, shape [batch_size*num_traj, 512, 4, 8]

        Returns:
            torch.Tensor: Scoring features, shape [batch_size*num_traj, 128, 1, 1]
        """
        x = self.conv1(x)         # [batch_size*num_traj, 256, 4, 8]
        x = self.bn1(x)           # Batch normalization
        x = self.relu1(x)         # ReLU activation
        x = self.conv2(x)         # [batch_size*num_traj, 128, 4, 8]
        x = self.bn2(x)           # Batch normalization
        x = self.relu2(x)         # ReLU activation
        x = self.pool(x)          # [batch_size*num_traj, 128, 1, 1]
        return x

class TrajectoryOffsetHead(nn.Module):
    def __init__(self, num_poses: int = 8, d_ffn: int=1024, d_model: int=256):
        super(TrajectoryOffsetHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        bz, num_trajs, _ = object_queries.shape
        poses = self._mlp(object_queries).reshape(bz, -1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi#Ellipsis表示所有前面的维度:poses[:, :, :, StateSE2Index.HEADING]
        return poses
