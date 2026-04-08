import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Tuple
import torch
import os
from navsim.agents.abstract_agent import AbstractAgent

class AgentLightningModule(pl.LightningModule):
    # def __init__(
    #     self,
    #     agent: AbstractAgent,
    # ):
    #     super().__init__()
    #     self.agent = agent
#FIXME:
    def __init__(self, agent: AbstractAgent, ckpt_path: str = None):
        super().__init__()
        self.agent = agent

        # ---- 加载 ckpt 参数 ----
        if ckpt_path is not None:
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            if "state_dict" in ckpt:   # Lightning 格式
                state_dict = {k.replace("agent.", ""): v for k, v in ckpt["state_dict"].items()}
            else:                      # 纯 state_dict 格式
                state_dict = ckpt

            missing, unexpected = self.agent.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
    
    def setup(self, stage=None):
        print("Setting up: freezing & unfreezing layers...")

        # Training profile switch.
        # - wm_reward_only: train controller-conditioned world-model transition + reward/map heads (test-time controller adaptation)
        # - controller_inject_only: ONLY train controller style token + injection layers (older CAP finetune)
        # - legacy: previous behavior (kept for backward compatibility)
        train_profile = os.getenv("WOTE_TRAIN_PROFILE", "controller_inject_only").strip().lower()

        # Always keep these frozen.
        always_frozen_modules = [
            "_backbone",
            "scene_position_embedding",
            "encode_ego_feat_mlp",
        ]

        # Determine controller injection mode (may not be used by newer profiles/models).
        inj_mode = str(getattr(self.agent.config, 'controller_injection_mode', 'film') or 'film').strip().lower()
        pooling = str(getattr(self.agent.config, 'controller_style_pooling', 'attn') or 'attn').strip().lower()
        wm_fusion = str(getattr(self.agent.config, 'controller_world_model_fusion', 'attn') or 'attn').strip().lower()

        if train_profile == "wm_reward_only":
            # New recommended profile for controller-aware latent transition:
            # - Train controller style token extractor
            # - Train latent world model (transition)
            # - Train reward heads
            # - Train map heads (for fut_bev_semantic_map supervision)
            trainable_modules = [
                # controller style token path
                "controller_encoder",
                "ctrl_proj",
                "ctrl_token_ln",
                # transition model
                "latent_world_model",
                # reward/scoring
                "reward_conv_net",
                "reward_cat_head",
                "reward_head",
                "sim_reward_heads",
                # map heads
                "_bev_upscale",
                "bev_upsample_head",
                "bev_semantic_head",
            ]
            if pooling == 'attn':
                trainable_modules.append("ctrl_style_attn")

            # World-model fusion layers (controller -> latent transition).
            # These must be trainable, otherwise controller conditioning is effectively random/frozen.
            if wm_fusion in {'attn', 'attention', 'cross_attn', 'cross_attention'}:
                trainable_modules += [
                    "ctrl_bank_proj",
                    "ctrl_bank_ln",
                    "ctrl_fuse_attn",
                ]
            elif wm_fusion.startswith('film'):
                trainable_modules += [
                    "ctrl_wm_film_scale",
                    "ctrl_wm_film_shift",
                    "ctrl_wm_film_ln",
                ]

            head_modules = []
        elif train_profile == "controller_inject_only":
            # Train controller embedding + pooling/projection + the actual injection layers that have parameters.
            # This matches the latest WoTE_model.py naming.
            trainable_modules = [
                "controller_encoder",
                "ctrl_proj",
                "ctrl_token_ln",
            ]
            if pooling == 'attn':
                trainable_modules.append("ctrl_style_attn")

            # Injection layers with parameters.
            # In the current model, FiLM is used for trajectory feature branch conditioning.
            if inj_mode == 'film':
                trainable_modules += [
                    "ctrl_traj_film_scale",
                    "ctrl_traj_film_shift",
                    "ctrl_traj_film_ln",
                ]
            else:
                # 'add'/'none': no extra trainable injection params beyond ctrl_proj/ln.
                pass

            # No generic head unfreezing in this profile.
            head_modules = []
        elif train_profile in ("controller_inject_offset_reward", "controller_cap_abc"):
            # Recommended minimal finetune for controller-aware planning (CAP):
            # - Controller style token path (must train)
            # - Injection A trainables (FiLM params when film)
            # - Offset branch (B)
            # - Reward/scoring branch (C)
            # Still keeps perception backbone and positional embeddings frozen.
            trainable_modules = [
                # style token path
                "controller_encoder",
                "ctrl_proj",
                "ctrl_token_ln",
                # offset branch (B)
                "offset_tf_decoder",
                "offset_head",
                "offset_score_head",
                # reward/scoring branch (C)
                "reward_conv_net",
                "reward_cat_head",
                "reward_head",
                "sim_reward_heads",
            ]
            if pooling == 'attn':
                trainable_modules.append("ctrl_style_attn")

            if inj_mode == 'film':
                trainable_modules += [
                    "ctrl_traj_film_scale",
                    "ctrl_traj_film_shift",
                    "ctrl_traj_film_ln",
                ]

            head_modules = []
        else:
            # Legacy behavior (previous default): train a larger set of modules.
            trainable_modules_base = [
                "controller_encoder",
                "ctrl_proj",
                "feat_proj",
                "temporal_conv",
                "transformer",
                "final_proj",
                "latent_world_model",
                "reward_conv_net",
                "reward_cat_head",
                "reward_head",
            ]

            if inj_mode == 'film':
                # NOTE: updated to match latest WoTE_model naming
                injection_trainables = [
                    "ctrl_traj_film_scale",
                    "ctrl_traj_film_shift",
                    "ctrl_traj_film_ln",
                ]
            elif inj_mode == 'attn':
                injection_trainables = ["ctrl_attn"]
            elif inj_mode == 'concat':
                injection_trainables = ["ctrl_concat_proj"]
            else:
                injection_trainables = []

            trainable_modules = trainable_modules_base + injection_trainables
            head_modules = ["head"]

        for name, param in self.agent.named_parameters():
            # 默认先冻结
            param.requires_grad = False

            # 解冻训练模块
            if any(key in name for key in trainable_modules):
                param.requires_grad = True

            # 强制冻结模块
            if any(key in name for key in always_frozen_modules):
                param.requires_grad = False

            # 解冻 head 模块
            if any(key in name for key in head_modules):
                param.requires_grad = True

        # 打印结果检查
        trainable = [n for n, p in self.agent.named_parameters() if p.requires_grad]
        print(f"Train profile: {train_profile} (inj_mode={inj_mode}, pooling={pooling})")
        print(f"Total trainable params: {len(trainable)}")
        if os.getenv('WOTE_PRINT_TRAINABLE', '0') == '1':
            print("Trainable params:")
            for n in trainable:
                print("   ", n)
        # print("Trainable params:")
        # for n in trainable:
        #     print("   ", n)

    
#FIXME:   
    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ):
        features, targets = batch


        input_target = self.agent.config.input_target if hasattr(self.agent.config, 'input_target') else False
        if input_target:
            prediction = self.agent.forward(features, targets)
        else:
            prediction = self.agent.forward(features)

        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if isinstance(loss_dict, Tensor):
            loss_dict = {"traj_loss": loss_dict}
            
        total_loss = 0.0
        for loss_key, loss_value in loss_dict.items():
            self.log(f"{logging_prefix}/{loss_key}", loss_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            if 'acc' in loss_key:
                continue
            total_loss = total_loss + loss_value
        return total_loss
    
    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.agent.get_optimizers()
    
    def backward(self, loss):
        # print('set detect anomaly')
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        
#FIXME:   
    # def on_after_backward(self):
    #     """检查哪些参数拥有梯度（被更新），哪些没有（冻结或未使用）"""
    #     if (self.global_step % 200) != 0:   # 每隔 200 step 打印一次
    #         return

    #     print(f"\n[Gradient Check] Step {self.global_step}")

    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(f"❌ NO GRAD: {name}")
    #         else:
    #             grad_norm = param.grad.norm().item()
    #             if grad_norm < 1e-8:
    #                 print(f"⚠️ ZERO GRAD: {name} (norm={grad_norm:.2e})")
    #             else:
    #                 print(f"✅ UPDATED: {name} (grad_norm={grad_norm:.2e})")