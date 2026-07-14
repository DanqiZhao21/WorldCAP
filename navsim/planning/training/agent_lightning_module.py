import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Tuple
import torch
import os
from navsim.agents.abstract_agent import AbstractAgent

TRAINABLE_GROUP_PREFIXES = {
    "controller_embedding": [
        "WoTE_model.controller_encoder",
        "WoTE_model.ctrl_proj",
        "WoTE_model.ctrl_token_ln",
        "WoTE_model.ctrl_bank_proj",
        "WoTE_model.ctrl_bank_ln",
        "WoTE_model.ctrl_fuse_attn",
        "WoTE_model.ctrl_wm_film_scale",
        "WoTE_model.ctrl_wm_film_shift",
        "WoTE_model.ctrl_wm_film_ln",
    ],
    "latent_world_model": [
        "WoTE_model.latent_world_model",
    ],
    "reward_heads": [
        "WoTE_model.reward_conv_net",
        "WoTE_model.reward_cat_head",
        "WoTE_model.reward_head",
        "WoTE_model.sim_reward_heads",
    ],
    "map_heads": [
        "WoTE_model._bev_upscale",
        "WoTE_model.bev_upsample_head",
        "WoTE_model.bev_semantic_head",
    ],
    "offset_heads": [
        "WoTE_model.offset_tf_decoder",
        "WoTE_model.offset_head",
        "WoTE_model.offset_score_head",
    ],
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


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

        if hasattr(self.agent, "load_post_checkpoint_modules"):
            self.agent.load_post_checkpoint_modules()
    
    def setup(self, stage=None):
        print("Setting up: freezing & unfreezing layers...")

        freeze_all = bool(getattr(self.agent.config, "freeze_all", True))
        trainable_groups = _as_list(getattr(self.agent.config, "trainable_groups", []))
        trainable_prefixes = _as_list(getattr(self.agent.config, "trainable_prefixes", []))
        frozen_prefixes = _as_list(
            getattr(
                self.agent.config,
                "frozen_prefixes",
                [
                    "WoTE_model._backbone",
                    "WoTE_model.scene_position_embedding",
                    "WoTE_model.encode_ego_feat_mlp",
                ],
            )
        )
        strict = bool(getattr(self.agent.config, "freeze_strict", True))

        unknown_groups = [group for group in trainable_groups if group not in TRAINABLE_GROUP_PREFIXES]
        if unknown_groups and strict:
            known = ", ".join(sorted(TRAINABLE_GROUP_PREFIXES))
            raise ValueError(f"Unknown trainable group(s): {unknown_groups}. Known groups: {known}")

        for group in trainable_groups:
            trainable_prefixes.extend(TRAINABLE_GROUP_PREFIXES.get(group, []))

        # These core perception/position modules are frozen by default unless the
        # YAML explicitly replaces frozen_prefixes with an empty list.
        default_frozen_prefixes = [
            "_backbone",
            "scene_position_embedding",
            "encode_ego_feat_mlp",
        ]
        for prefix in default_frozen_prefixes:
            qualified = f"WoTE_model.{prefix}"
            if not any(item == qualified or item == prefix for item in frozen_prefixes):
                frozen_prefixes.append(qualified)

        trainable_matches = {prefix: 0 for prefix in trainable_prefixes}
        frozen_matches = {prefix: 0 for prefix in frozen_prefixes}
        for name, param in self.agent.named_parameters():
            param.requires_grad = not freeze_all

            matched_trainable = [prefix for prefix in trainable_prefixes if name.startswith(prefix)]
            if matched_trainable:
                param.requires_grad = True
                for prefix in matched_trainable:
                    trainable_matches[prefix] += 1

            matched_frozen = [prefix for prefix in frozen_prefixes if name.startswith(prefix)]
            if matched_frozen:
                param.requires_grad = False
                for prefix in matched_frozen:
                    frozen_matches[prefix] += 1

        if strict:
            missing_trainable = [prefix for prefix, count in trainable_matches.items() if count == 0]
            missing_frozen = [prefix for prefix, count in frozen_matches.items() if count == 0]
            if missing_trainable:
                raise ValueError(f"Trainable prefix(es) matched no parameters: {missing_trainable}")
            if missing_frozen:
                raise ValueError(f"Frozen prefix(es) matched no parameters: {missing_frozen}")

        # 打印结果检查
        trainable = [n for n, p in self.agent.named_parameters() if p.requires_grad]
        print(f"Trainable groups: {trainable_groups}")
        print(f"Trainable prefixes: {trainable_prefixes}")
        print(f"Frozen prefixes: {frozen_prefixes}")
        print(f"Total trainable params: {len(trainable)}")
        print_trainable = bool(getattr(self.agent.config, "print_trainable", False)) or os.getenv('WOTE_PRINT_TRAINABLE', '0') == '1'
        if print_trainable:
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
