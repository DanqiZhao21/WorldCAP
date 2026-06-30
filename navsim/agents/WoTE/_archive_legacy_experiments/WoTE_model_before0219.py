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
from typing import Any, List, Dict, Union

import matplotlib.pyplot as plt


#FIXME: 新增embedding模块
from ControllerInTheLoop.step2_Embedding.ControllerEmbedding import ControllerEmbedding

# Differentiable controller execution surrogate
from navsim.agents.WoTE.controller_response_predictor import ControllerResponsePredictor



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
        self.num_sampled_trajs_NoSample=256#FIXME:若非训练阶段则都选上
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
        
        #FIXME:
        # ===== controller embedding =====
        self.controller_emb_dim = 64
        # 新增：controller_feature_mode (默认 full) —— 不覆写原 embedding，只是可选地屏蔽纵向相关特征
        # 取值建议：'full' | 'lateral_only'
        self.controller_feature_mode = getattr(config, 'controller_feature_mode', 'full')
        # 新增：controller_target_offset_mode (默认跟随 controller_feature_mode)
        # 用于动态构建 target BEV 时是否仅使用横向偏移（纵向沿 ref-anchor）
        self.controller_target_offset_mode = getattr(config, 'controller_target_offset_mode', self.controller_feature_mode)

        self.controller_encoder = ControllerEmbedding(
            emb_dim=self.controller_emb_dim,
            feature_mode=self.controller_feature_mode,
        )

        # How to pool controller-bank embeddings into a single global style embedding.
        # Kept for backward compatibility with older configs/scripts.
        # - 'mean': average over controller-bank candidates
        # - 'attn': learned attention pooling (default)
        self.controller_style_pooling = getattr(config, 'controller_style_pooling', 'attn')
        self.ctrl_style_attn = nn.Linear(self.controller_emb_dim, 1)

        # Controller-aware latent transition (BEV world model):
        # We model controller as part of the transition dynamics by conditioning the latent
        # world-model forward on a global controller style token.
        self.controller_condition_on_world_model = bool(getattr(config, 'controller_condition_on_world_model', True))
        self.controller_world_model_strength = float(getattr(config, 'controller_world_model_strength', 0.3) or 0.3)
        # Where to inject controller token in the world model input tokens:
        # - 'all' (default): add to ego + all BEV tokens
        # - 'ego': add only to ego token (index 0)
        self.controller_world_model_inject_target = str(getattr(config, 'controller_world_model_inject_target', 'all') or 'all').lower()

        # projection 把 embedding 投影成 BEV 维度
        self.ctrl_proj = nn.Linear(self.controller_emb_dim, 256)
        self.ctrl_token_ln = nn.LayerNorm(256)
        # Controller token projection into model hidden_dim.
        # NOTE: We intentionally do NOT condition traj/offset/reward branches anymore.

        # ===== controller execution (response predictor) =====
        # If enabled, we approximate online execution for planned trajectories in a differentiable way.
        self.controller_execute_predicted_traj = bool(getattr(config, 'controller_execute_predicted_traj', True))
        self.controller_execute_apply_in_train = bool(getattr(config, 'controller_execute_apply_in_train', True))
        self.controller_execute_apply_in_eval = bool(getattr(config, 'controller_execute_apply_in_eval', False))
        self.controller_response_predictor_path = str(getattr(config, 'controller_response_predictor_path', '') or '')
        self.controller_response_predictor_trainable = bool(getattr(config, 'controller_response_predictor_trainable', False))

        # Controller injection into OFFSET branch.
        # New name: controller_offset_inject_strength; fallback to old controller_offset_condition_strength.
        self.controller_offset_inject_strength = float(
            getattr(config, 'controller_offset_inject_strength', getattr(config, 'controller_offset_condition_strength', 0.0)) or 0.0
        )

        # Load controller bank (ref, exec).
        # Supports:
        # - exec: .npy, ref: .npy
        # - exec: bundle .npz with keys exec_trajs [S,256,8,3], ref_traj [256,8,3] (sample styles during training)
        ref_path = getattr(
            config,
            'controller_ref_traj_path',
            "/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy",
        )
        exec_path = getattr(
            config,
            'controller_exec_traj_path',
            "/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz",
        )

        # If using a bundle, keep the full style set for training-time sampling.
        self._controller_bundle_exec = None  # torch.Tensor [S, 256, 8, 3]
        self._controller_bundle_ref = None   # torch.Tensor [256, 8, 3]
        self._controller_bundle_style_names = None  # Optional[List[str]]

        try:
            if isinstance(exec_path, str) and exec_path.endswith('.npz') and os.path.isfile(exec_path):
                data = np.load(exec_path, allow_pickle=True)
                exec_trajs = data.get('exec_trajs', None)
                ref_traj_np = data.get('ref_traj', None)
                if exec_trajs is None or ref_traj_np is None:
                    raise RuntimeError('Missing exec_trajs/ref_traj in controller bundle')
                self._controller_bundle_exec = torch.tensor(exec_trajs, dtype=torch.float32)  # [S,256,8,3]
                self._controller_bundle_ref = torch.tensor(ref_traj_np, dtype=torch.float32)  # [256,8,3]
                style_names = data.get('style_names', None)
                if style_names is not None:
                    try:
                        self._controller_bundle_style_names = [str(x) for x in list(style_names)]
                    except Exception:
                        self._controller_bundle_style_names = None

                # Initialize active style (can be overridden via env at runtime).
                self._active_ref_trajs = self._controller_bundle_ref
                self._active_exec_trajs = self._controller_bundle_exec[0]
                print(f"💜 Loaded controller bundle: {exec_path} (styles={int(self._controller_bundle_exec.shape[0])})")
            else:
                self._active_ref_trajs = torch.tensor(np.load(ref_path), dtype=torch.float32)
                self._active_exec_trajs = torch.tensor(np.load(exec_path), dtype=torch.float32)
                print(f"💜 Loaded controller bank: ref={ref_path} exec={exec_path}")
        except Exception as e:
            print(f"⚠️ Failed to load controller bank, fallback to zeros: {e}")
            # Fallback to keep training/eval alive (controller injection becomes near-noop).
            self._active_ref_trajs = torch.zeros((256, 8, 3), dtype=torch.float32)
            self._active_exec_trajs = torch.zeros((256, 8, 3), dtype=torch.float32)

        # Instantiate response predictor (optionally load checkpoint).
        self.controller_response_predictor = None
        try:
            self.controller_response_predictor = ControllerResponsePredictor(
                num_poses=int(getattr(config.trajectory_sampling, 'num_poses', 8)),
                traj_dim=3,
                controller_emb_dim=self.controller_emb_dim,
            )

            loaded = False
            if self.controller_response_predictor_path and os.path.isfile(self.controller_response_predictor_path):
                ckpt = torch.load(self.controller_response_predictor_path, map_location='cpu')

                # Accept a few common checkpoint layouts.
                state_dict = None
                if isinstance(ckpt, dict):
                    if isinstance(ckpt.get('state_dict', None), dict):
                        state_dict = ckpt['state_dict']
                    elif isinstance(ckpt.get('response_predictor', None), dict):
                        state_dict = ckpt['response_predictor']
                    elif isinstance(ckpt.get('model', None), dict):
                        state_dict = ckpt['model']
                    else:
                        # Sometimes the whole dict is already a state_dict.
                        if all(isinstance(k, str) for k in ckpt.keys()):
                            state_dict = ckpt

                if isinstance(state_dict, dict):
                    # Strip common prefixes.
                    cleaned = {}
                    for k, v in state_dict.items():
                        nk = k
                        for prefix in ('controller_response_predictor.', 'response_predictor.', 'module.'):
                            if nk.startswith(prefix):
                                nk = nk[len(prefix):]
                        cleaned[nk] = v
                    self.controller_response_predictor.load_state_dict(cleaned, strict=False)
                    loaded = True

                if loaded:
                    print(f"💜 Loaded controller response predictor: {self.controller_response_predictor_path}")
                else:
                    print(f"⚠️ controller response predictor ckpt format not recognized: {self.controller_response_predictor_path}")
            elif self.controller_execute_predicted_traj:
                if self.controller_response_predictor_path:
                    print(f"⚠️ controller response predictor missing: {self.controller_response_predictor_path} (fallback to identity execution)")
                else:
                    print("⚠️ controller response predictor enabled but controller_response_predictor_path is empty (fallback to identity execution)")

            if not self.controller_response_predictor_trainable and self.controller_response_predictor is not None:
                for p in self.controller_response_predictor.parameters():
                    p.requires_grad_(False)
        except Exception as e:
            print(f"⚠️ Failed to init controller response predictor (fallback to identity execution): {e}")
            self.controller_response_predictor = None

    def _maybe_sample_controller_style_for_batch(self):
        """Sample one controller style per forward call during training.

        - Only active when controller exec path is a bundle (.npz) with multiple styles.
        - Disabled in eval to avoid metric jitter.
        - Optional env override:
            WOTE_CTRL_STYLE_IDX=<int> forces a fixed style index.
            WOTE_CTRL_STYLE_DEBUG=1 prints chosen style.
        """
        if getattr(self, 'is_eval', False):
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
            return

        forced = os.environ.get('WOTE_CTRL_STYLE_IDX', None)
        if forced is not None and str(forced).strip() != '':
            try:
                idx = int(forced)
                idx = max(0, min(num_styles - 1, idx))
            except Exception:
                idx = 0
        else:
            import random
            idx = random.randrange(num_styles)

        self._active_ref_trajs = self._controller_bundle_ref
        self._active_exec_trajs = self._controller_bundle_exec[idx]

        # For debugging / logging alignment.
        self._active_style_idx = int(idx)
        self._active_style_name = None
        if isinstance(self._controller_bundle_style_names, list) and idx < len(self._controller_bundle_style_names):
            self._active_style_name = self._controller_bundle_style_names[idx]

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
        """
        Part 1: extract_trajectory_feature
        Perform feature extraction, trajectory center processing, optional offset prediction, etc.
        """
        results = {}

        # Extract input features
        camera_feature = features["camera_feature"]
        lidar_feature = features["lidar_feature"]
        status_feature = features["status_feature"]
        # print(f"✅status_feature shape is :{status_feature.shape}")  #train==>([16, 8])
        # print(f"✅lidar_feature shape is :{lidar_feature.shape}")#train==》torch.Size([16, 1, 256, 256])
        # print(f"✅camera_feature shape is :{camera_feature.shape}")#train==>([16, 3, 256, 1024])
        #test evaluation的时候
        # (wrapped_fn pid=2437463) ✅status_feature shape is :torch.Size([1, 8]) [repeated 60x across cluster]
        # (wrapped_fn pid=2437463) ✅lidar_feature shape is :torch.Size([1, 1, 256, 256]) [repeated 60x across cluster]
        # (wrapped_fn pid=2437463) ✅camera_feature shape is :torch.Size([1, 3, 256, 1024]) [repeated 60x across cluster]
        
        # Get batch size
        batch_size = status_feature.shape[0]

        # Process backbone and BEV features
        backbone_bev_feature, flatten_bev_feature = self._process_backbone_features(
            camera_feature, lidar_feature
        )

        # Get ego status features
        ego_status_feat = self._get_ego_status_feature(status_feature)

        # Get cluster center features
        init_trajectory_anchor = self.trajectory_anchors.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #ego 当前状态特征 + 每个轨迹 anchor  → 编码成 ego-level latent 表征（feature embedding）
        #NOTE此处体现了Option A ;影响了encode traj into ego feature输出的ego feat encoded,采用的controller_injection_mode[film或者add]
        ego_feat_fixed_anchor_WoTE, num_traj = self.encode_traj_into_ego_feat(ego_status_feat, init_trajectory_anchor, batch_size)
        
        # Optional offset prediction
        offset_dict = self._predict_offset(
            ego_feat_fixed_anchor_WoTE,
            flatten_bev_feature,
            init_trajectory_anchor=init_trajectory_anchor,
        )
        results.update(offset_dict)
        
        # Optional losses
        if self.use_agent_loss:
            agents, agents_query = self._process_agent(batch_size, flatten_bev_feature) #flatten_bev_feature.shape torch.Size([32, 32, 256])
            results.update(agents)
        if self.use_map_loss:
            bev_semantic_map, upsampled_bev_feature = self._process_map(flatten_bev_feature, batch_size)
            results["bev_semantic_map"] = bev_semantic_map

        if self.is_eval:
            trajectory_offset = offset_dict["trajectory_offset"]
            trajectory_offset_rewards = offset_dict["trajectory_offset_rewards"]
            offseted_trajectory_anchors = init_trajectory_anchor + trajectory_offset

            ego_feat_for_reward_network, _ = self.encode_traj_into_ego_feat(ego_status_feat, offseted_trajectory_anchors, batch_size)
        else: 
            # training
            ego_feat_for_reward_network = ego_feat_fixed_anchor_WoTE

        # Return intermediate results for subsequent stages
        trajectory_outputs = {
            "results": results,  # May include offset_dict and other intermediate results
            "batch_size": batch_size,
            "num_traj": num_traj,
            "flatten_bev_feature": flatten_bev_feature,
            "ego_feat": ego_feat_for_reward_network,
        }
        return trajectory_outputs

    
    def extract_reward_feature(self, trajectory_outputs, targets) -> Dict[str, torch.Tensor]:
        # Retrieve necessary variables from the previous intermediate results
        results = trajectory_outputs["results"]
        batch_size = trajectory_outputs["batch_size"]
        # 若缓存仅提供基础 BEV 目标，则在运行期基于当前控制器执行轨迹重建具体的未来 BEV 目标
        self._compose_future_bev_targets_from_base(targets)
        # print(f"💜In extract_reward_feature : batch_size is {batch_size}")   
        num_traj = trajectory_outputs["num_traj"]#batch_size is 16;num_traj is256
        # print(f"💜batch_size is {batch_size};num_traj is{num_traj}")
        flatten_bev_feature = trajectory_outputs["flatten_bev_feature"]#当前帧的 BEV feature，已经被编码成扁平的 query 表示。
        ego_feat = trajectory_outputs["ego_feat"]
        # print(f"💜flatten_bev_feature: {flatten_bev_feature.shape}")#flatten_bev_feature: torch.Size([16, 64, 256]) ==》(B, Nq, C)
        # print(f"💜ego_feat: {ego_feat.shape}")#ego_feat: torch.Size([16, 256, 1, 256]) #(B, num_traj, 1, C)

        # Inject ego features into the BEV map
        flatten_bev_feature_multi_trajs = flatten_bev_feature.unsqueeze(1).repeat(
            1, num_traj, 1, 1
        )  # [batch_size, num_traj, H*W, C]
        flatten_bev_feature_multi_trajs = self._inject_cur_ego_into_bev(
            flatten_bev_feature_multi_trajs, ego_feat, num_traj
        )
        # print(f"💜flatten_bev_feature_multi_trajs(afer inject ego-info): {flatten_bev_feature_multi_trajs.shape}")#torch.Size([16, 256, 64, 256])
        
        # Process features through the latent world model
        ego_feat = ego_feat.reshape(batch_size * num_traj, 1, -1)
        flatten_bev_feature_multi_trajs = flatten_bev_feature_multi_trajs.reshape(
            batch_size * num_traj, self.num_plan_queries, -1
        )
        # print(f"💜flatten_bev_feature_multi_trajs(afer reshape): {flatten_bev_feature_multi_trajs.shape}")#torch.Size([4096, 64, 256])
        
        # Controller is injected into the latent transition (world model) inside
        # `_latent_world_model_processing` to model controller-dependent dynamics.
   
        # Multiple iterations
        num_iterations = self.num_fut_timestep
        interval = 8 // num_iterations  #未来8步 num_iterations为迭代几次

        fut_ego_feat = ego_feat
        ego_feat_list = [fut_ego_feat]#这个是第一个元素 初始状态的存在后面会想列表一样加进去

        fut_flatten_bev_feature_multi_trajs = flatten_bev_feature_multi_trajs
        bev_feat_list = [fut_flatten_bev_feature_multi_trajs]
        # print(f"💟💟-1_shape of bev_feat_list[0]: {fut_flatten_bev_feature_multi_trajs.shape}")#([256, 64, 256])==>train：torch.Size([4096, 64, 256])
        for i in range(num_iterations):
            fut_ego_feat, fut_flatten_bev_feature_multi_trajs = self._latent_world_model_processing(
                fut_flatten_bev_feature_multi_trajs, fut_ego_feat, batch_size, num_traj
            )
            # print(f"💟💟0_shape of fut_flatten_bev_feature_multi_trajs: {fut_flatten_bev_feature_multi_trajs.shape}")
            #💟💟0_shape of fut_flatten_bev_feature_multi_trajs: torch.Size([256, 64, 256])==》train([4096, 64, 256])
            fut_flatten_bev_feature_multi_trajs = self._inject_fut_ego_into_bev(  #基本上算是场景feat了  都得加入ego-feat
                fut_flatten_bev_feature_multi_trajs,
                fut_ego_feat,
                num_traj,
                fut_idx=(i + 1) * interval
            )
            # print(f"💟💟1_shape of fut_flatten_bev_feature_multi_trajs: {fut_flatten_bev_feature_multi_trajs.shape}")#💟💟1_shape of fut_flatten_bev_feature_multi_trajs: torch.Size([256, 64, 256])==》train([4096, 64, 256])
            ego_feat_list.append(fut_ego_feat)
            bev_feat_list.append(fut_flatten_bev_feature_multi_trajs)

        # 
        fut_ego_feat = ego_feat_list[-1]
        fut_flatten_bev_feature_multi_trajs = bev_feat_list[-1]#这个应该是有自车信息的

        # Compute reward features
        reward_feature = self._compute_reward_feature(
            ego_feat_list,
            bev_feat_list,
            batch_size,
            num_traj,
        )
        
        results["reward_feature"] = reward_feature
        
        #PRINT:
        # print(f"✅batchsize is {batch_size}")
        # print(f"✅self.num_sample is {self.num_sampled_trajs}")#self.num_sample is 1  ==>train✅batchsize is 16✅self.num_sample is 1
        # print(f"💟💟2_shape of fut_flatten_bev_feature_multi_trajs: {fut_flatten_bev_feature_multi_trajs.shape}")#💟💟2_shape of fut_flatten_bev_feature_multi_trajs: torch.Size([256, 64, 256])
        
        # save_dir = "/home/zhaodanqi/clone/WoTE/trainingResult/bev-pic"
        # fut_bev_semantic_map = self._process_future_map_NoSample(
        #         fut_flatten_bev_feature_multi_trajs,#（256 64 256）
        #         batch_size
        #     )
        
        # # ====== 世界模型每一步 BEV 可视化 ======
        # if self.is_eval:   # 只在 eval 模式下画图

        #     ##[16batchsize*256条轨迹=4096, 64（num_plan_queries）, 256（hidden_dim）]->【4096 64 256】
        #     for t, fut_bev in enumerate(bev_feat_list):#bev_feat_list是一个列表，包含从0到num_iteration的 “fut_flatten_bev_feature_multi_trajs” 
        #         # fut_bev: [B*num_traj, Nq, C]【4096 64 256】
        #         # 先选第一个 trajectory 的 BEV
                
        #         fut_bev_sem_map = self._process_future_map_NoSample(
        #             fut_bev, batch_size
        #         )   # → [B, 1, H, W]  或 [B, num_class, H, W]
        #         bev_map0 = fut_bev_sem_map[0].detach().cpu()#只去了batch=0的                
        #         sem = bev_map0.argmax(dim=0).numpy()  # 转成 [H, W]
        #         plt.figure(figsize=(5, 5))
        #         plt.imshow(sem, cmap='tab20')
        #         plt.axis('off')

        #         save_path = os.path.join(save_dir, f"future_bev_step_{t}.png")
        #         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        #         plt.close()  # 关闭 plt 避免弹窗
               
        #PRINT:
        

#FIXME:bev fut loss 只取最终步数的ego特征和BEV特征
        # NOTE: fut BEV semantic map supervision shares the same heads as map loss.
        # If `use_map_loss` is disabled, those heads are not constructed.
        if (targets is not None) and self.use_map_loss:  # 训练阶段 + 需要 map/fut_map loss 才计算
            # Prepare future BEV features
            #这里的targets的anchor没有用simulator么？
            sampled_fut_flatten_bev_feature_multi_trajs = self._sample_future_bev_feature(#这里根据tragets的采样 对应的采样出latent woard model关于该anchor轨迹的前推的东西
                fut_flatten_bev_feature_multi_trajs,
                batch_size,
                num_traj,
                targets=targets
            )
            fut_bev_semantic_map = self._process_future_map(
                sampled_fut_flatten_bev_feature_multi_trajs,
                batch_size
            )
            # print("💚💚💚计算fut_bev_semantic_map 损失 ❤")
            results["fut_bev_semantic_map"] = fut_bev_semantic_map

        return results
#FIXME:

    def process_trajectory_and_reward(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, torch.Tensor]:
        # Sample controller style once per batch (training only) to improve generalization.
        self._maybe_sample_controller_style_for_batch()
        trajectory_outputs = self.extract_trajectory_feature(features, targets)
        final_results = self.extract_reward_feature(trajectory_outputs, targets)
        return final_results

#ADD 剩下的都试一下功能函数

    def _compute_controller_style_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute a controller *style* token independent of planner candidate trajectories.

        Returns:
            style_token: [batch_size, 256]
        """
        if (not getattr(self, 'controller_condition_on_world_model', False)) or (getattr(self, 'controller_world_model_strength', 0.0) <= 0.0):
            return torch.zeros((batch_size, 256), device=device)

        style_emb = self._compute_controller_style_emb64(batch_size, device)  # [B, emb_dim]
        return self.ctrl_token_ln(self.ctrl_proj(style_emb))

    def _compute_controller_style_emb64(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute controller embedding in controller_emb_dim (default 64).

        Returns:
            style_emb: [B, controller_emb_dim]
        """
        if not hasattr(self, '_active_ref_trajs') or self._active_ref_trajs is None:
            return torch.zeros((batch_size, self.controller_emb_dim), device=device)
        if not hasattr(self, '_active_exec_trajs') or self._active_exec_trajs is None:
            return torch.zeros((batch_size, self.controller_emb_dim), device=device)

        ref_traj = self._active_ref_trajs.to(device)
        exec_traj = self._active_exec_trajs.to(device)

        bank_emb = self.controller_encoder(ref_traj, exec_traj)  # [N_ctrl, emb_dim]

        pooling = str(getattr(self, 'controller_style_pooling', None) or 'attn').lower()
        if pooling in {'mean', 'avg', 'average'}:
            style_emb = bank_emb.mean(dim=0)
        else:
            logits = self.ctrl_style_attn(bank_emb).squeeze(-1)  # [N_ctrl]
            w = torch.softmax(logits, dim=0)
            style_emb = (w.unsqueeze(-1) * bank_emb).sum(dim=0)

        return style_emb.unsqueeze(0).expand(batch_size, -1)

    def _should_execute_predicted_trajectories(self) -> bool:
        if not getattr(self, 'controller_execute_predicted_traj', False):
            return False
        if getattr(self, 'controller_response_predictor', None) is None:
            return False
        if getattr(self, 'is_eval', False):
            return bool(getattr(self, 'controller_execute_apply_in_eval', False))
        return bool(getattr(self, 'controller_execute_apply_in_train', True))

    def _execute_planned_trajectories_with_controller(self, planned_trajs: torch.Tensor) -> torch.Tensor:
        """Differentiable execution surrogate.

        Args:
            planned_trajs: [B,K,T,3] or [B,T,3]

        Returns:
            exec_trajs_hat: same shape as planned_trajs
        """
        if not self._should_execute_predicted_trajectories():
            return planned_trajs

        rp = getattr(self, 'controller_response_predictor', None)
        if rp is None:
            return planned_trajs

        if planned_trajs.ndim == 3:
            B, T, D = planned_trajs.shape
            style_emb = self._compute_controller_style_emb64(B, planned_trajs.device)
            residual = rp(planned_trajs, style_emb)
            return planned_trajs + residual

        if planned_trajs.ndim != 4:
            return planned_trajs

        B, K, T, D = planned_trajs.shape
        flat = planned_trajs.reshape(B * K, T, D)
        style_emb_b = self._compute_controller_style_emb64(B, planned_trajs.device)  # [B,64]
        style_emb = style_emb_b[:, None, :].expand(B, K, -1).reshape(B * K, -1)
        residual = rp(flat, style_emb)
        exec_flat = flat + residual
        return exec_flat.reshape(B, K, T, D)

    def _predict_offset(
        self,
        ego_feat: torch.Tensor,
        flatten_bev_feature: torch.Tensor,
        init_trajectory_anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict the offset for the cluster centers.
        """
        offset_dict = {}

        ego_feat = ego_feat.squeeze(2)  # [batch_size, num_traj, C]

        # Inject controller style token into OFFSET branch features.
        if bool(getattr(self._config, 'controller_condition_on_offset', getattr(self, 'controller_condition_on_offset', False))):
            s_cfg = float(
                getattr(
                    self._config,
                    'controller_offset_inject_strength',
                    getattr(self._config, 'controller_offset_condition_strength', getattr(self, 'controller_offset_inject_strength', 0.0)),
                )
                or 0.0
            )
            s = float(max(0.0, min(1.0, s_cfg)))
            if s > 0.0:
                style_emb = self._compute_controller_style_emb64(ego_feat.shape[0], ego_feat.device)
                style_token = self.ctrl_token_ln(self.ctrl_proj(style_emb))  # [B,256]
                ego_feat = ego_feat + (s * style_token[:, None, :])

        ego_feat = self.offset_tf_decoder(ego_feat, flatten_bev_feature) # [batch_size, num_traj, C]
        trajectory_offset = self.offset_head(ego_feat)
        trajectory_offset_rewards = self.offset_score_head(ego_feat).squeeze(-1)  # [batch_size, num_traj]
        trajectory_offset_rewards = torch.softmax(trajectory_offset_rewards, dim=-1)
        offset_dict["trajectory_offset"] = trajectory_offset
        offset_dict["trajectory_offset_rewards"] = trajectory_offset_rewards

        # Optional: execute planned trajectories and expose executed residuals for losses.
        if init_trajectory_anchor is not None:
            try:
                planned = init_trajectory_anchor + trajectory_offset  # [B,256,8,3]
                exec_trajs_hat = self._execute_planned_trajectories_with_controller(planned)
                offset_dict["trajectory_offset_exec_trajs"] = exec_trajs_hat
                offset_dict["trajectory_offset_executed"] = exec_trajs_hat - init_trajectory_anchor
            except Exception:
                # Keep training robust if shapes mismatch.
                pass
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
        # IMPORTANT: future ego injection coordinates are tied to planner candidate trajectories (anchors),
        # not controller bank trajectories. Controller is treated as a global style / preference.
        coors = self.trajectory_anchors[:, fut_idx-1, :2].to(ego_feat.device).unsqueeze(0).repeat(bz, 1, 1).reshape(bz*num_traj, -1)

        scene_bev_feature = self.inject_ego_feat_to_bev_map(scene_bev_feature, ego_feat, coors)
        scene_bev_feature = scene_bev_feature.view(bz*num_traj, -1, h * w)  # [batch_size, num_traj, C, H*W]
        scene_bev_feature = scene_bev_feature.permute(0, 2, 1)  # [batch_size*num_traj, H*W, C]
        return scene_bev_feature

    def _latent_world_model_processing(self, flatten_bev_feature_multi_trajs: torch.Tensor, ego_feat: torch.Tensor, batch_size: int, num_traj: int) -> torch.Tensor:
        """
        Process features through the latent world model.
        """
        # Concatenate ego_feat and flatten_bev_feature_multi_trajs
        # Assume concatenation along the feature dimension
        scene_feature = torch.cat([ego_feat, flatten_bev_feature_multi_trajs], dim=1) 
        # print(f"💚shape of initial scene_feature: {scene_feature.shape}")#([4096, 65, 256])

        # Add positional embedding
        scene_position_embedding = self.scene_position_embedding.weight[None,  :, :].repeat(batch_size * num_traj, 1, 1)  
        # print(f"💚shape of scene_position_embedding: {scene_position_embedding.shape}")#([4096, 65, 256])
        scene_feature = scene_feature + scene_position_embedding    #没有拼接、没有广播扩展，只是简单地把每个元素加上对应位置的值。这个是逐元素相加
        # print(f"💚shape of scene_feature: {scene_feature.shape}")#torch.Size([4096, 65, 256])

        # Controller-aware transition: condition world-model tokens on controller style.
        if getattr(self._config, 'controller_condition_on_world_model', getattr(self, 'controller_condition_on_world_model', False)):
            s_cfg = float(getattr(self._config, 'controller_world_model_strength', getattr(self, 'controller_world_model_strength', 0.0)) or 0.0)
            s = float(max(0.0, min(1.0, s_cfg)))
            if s > 0.0:
                style_token = self._compute_controller_style_token(batch_size, scene_feature.device)  # [B, 256]
                # expand to (B*num_traj, 1, 256) then broadcast over token dim
                style_bt = style_token[:, None, :].expand(batch_size, num_traj, -1).reshape(batch_size * num_traj, 1, -1)
                target = str(getattr(self._config, 'controller_world_model_inject_target', getattr(self, 'controller_world_model_inject_target', 'all')) or 'all').lower()
                if target in {'ego', 'ego_only'}:
                    scene_feature[:, 0:1, :] = scene_feature[:, 0:1, :] + (s * style_bt)
                else:
                    scene_feature = scene_feature + (s * style_bt)

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
        """
        bev_feat_list = []
        for bev_feat in  fut_flatten_bev_feature_multi_trajs_list:
            bev_feat = bev_feat.view(batch_size * num_traj, h, w, -1) 
            bev_feat_list.append(bev_feat)
        all_bev_feature = torch.cat(bev_feat_list, dim=-1)  # [batch_size*num_traj, H, W, C1 + C2]
        all_bev_feature = all_bev_feature.permute(0, 3, 1, 2)  # [batch_size*num_traj, C1 + C2, H, W]
        
        # Apply convolution network
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
    
    
    #FIXME:
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
    #FIXME:
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
#NOTE:为了打印原始轨迹经过仿真器后的情况，需要取消这里的offset加法
        trajectory_anchors_ori=self.trajectory_anchors
        poses = self.select_best_trajectory(final_rewards, trajectory_anchors, batch_size)
        # print(f"💙💙💙selected best trajectory poses shape: {poses.shape}")
        # 💙💙💙selected best trajectory poses shape: torch.Size([1, 8, 3])
        
#NOTE:改成多模态轨迹的可视化效果！！！
        results = {
            "trajectory": poses,#[batch_size, 8, 3]#得分最高的那一条
            "final_rewards": final_rewards,#256条轨迹的最终得分
            "trajectoryAnchor": trajectory_anchors_ori,#[256, 8, 3]
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

        if hasattr(self, '_active_style_idx'):
            try:
                result["controller_style_idx"] = torch.tensor(int(self._active_style_idx), device=im_rewards_softmax.device)
            except Exception:
                pass

        if hasattr(self, '_active_style_name') and self._active_style_name is not None:
            # Keep as python string for logging/debug (not used in loss).
            result["controller_style_name"] = str(self._active_style_name)

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
    def _compose_future_bev_targets_from_base(self, targets: Dict[str, torch.Tensor]):
        """
        Rebuild future BEV semantic maps from cached base map using planner candidate trajectory
        anchors (self.trajectory_anchors) and cached indices + frame interval. No-op if already present.
        Expects keys: 'fut_bev_semantic_map_base', 'sampled_trajs_index', 'frame_interval'.
        Produces: 'fut_bev_semantic_map' as stacked maps for sampled trajectories.
        """
        if targets is None:
            print("👉👉👉targets is None")
            return
        if ("fut_bev_semantic_map" in targets) or ("fut_bev_semantic_map_base" not in targets):
            print("👉👉👉targets already contain fut_bev_semantic_map or missing fut_bev_semantic_map_base")
            return

        base_map: torch.Tensor = targets["fut_bev_semantic_map_base"]  # [H, W] 或 [B, H, W]
        sampled_idx = targets.get("sampled_trajs_index", None)
        frame_interval = targets.get("frame_interval", None)
        if sampled_idx is None or frame_interval is None:
            print("👉👉👉targets missing sampled_trajs_index or frame_interval")
            return  # 信息不足；保持原样

        # 规范类型
        if isinstance(sampled_idx, np.ndarray):
            sampled_idx = torch.from_numpy(sampled_idx)
        if not torch.is_tensor(sampled_idx):
            sampled_idx = torch.tensor(sampled_idx, dtype=torch.long)
        anchor_device = self.trajectory_anchors.device
        sampled_idx = sampled_idx.to(anchor_device)

        # 处理不同维度的 batch 索引
        device = base_map.device
        # 归一化 frame_interval，支持标量或按 batch 提供
        if torch.is_tensor(frame_interval):
            fi_tensor = frame_interval.detach().cpu().long().view(-1)  # [B] 或 [1]
        else:
            fi_tensor = torch.tensor([int(frame_interval)], dtype=torch.long)
        # 限制范围到 [0, T-1]（T 来自 planner anchors 的时间步数）
        T = int(self.trajectory_anchors.shape[1])
        fi_tensor = torch.clamp(fi_tensor, min=0, max=max(0, T - 1))

        # 单样本：sampled_idx 为 1D
        if sampled_idx.ndim == 1:
            # 单样本：K 个轨迹，frame_interval 取 fi_tensor[0]
            fi = int(fi_tensor[0].item())
            try:
                anchors = self.trajectory_anchors.index_select(0, sampled_idx)  # [K, T, 3]
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
                    anchors_b = self.trajectory_anchors.index_select(0, idx_b)  # [K, T, 3]
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
