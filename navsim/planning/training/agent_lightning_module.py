import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Tuple
import torch
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

        # 定义不同模块关键词
        freeze_modules = ["_backbone", "scene_position_embedding", "encode_ego_feat_mlp"]
        # 基础可训练模块
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

        # 根据注入模式，仅解冻被实际使用的控制器注入模块，避免 DDP 未用参数报错
        mode = getattr(self.agent.config, 'controller_injection_mode', 'attn')
        if mode == 'film':
            injection_trainables = [
                "ctrl_film_scale",
                "ctrl_film_shift",
                "ctrl_film_ln",
            ]
        elif mode == 'attn':
            injection_trainables = [
                "ctrl_attn",
            ]
        elif mode == 'concat':
            injection_trainables = [
                "ctrl_concat_proj",
            ]
        else:  # 'sum' 或其他无参数模式
            injection_trainables = []

        trainable_modules = trainable_modules_base + injection_trainables
        untrainable_modules = [
            "_backbone","scene_position_embedding", "encode_ego_feat_mlp"
            ]
        head_modules = ["head"]

        for name, param in self.agent.named_parameters():
            # 默认先冻结
            param.requires_grad = False

            # 解冻训练模块
            if any(key in name for key in trainable_modules):
                param.requires_grad = True
                
            if any(key in name for key in untrainable_modules):
                param.requires_grad = False

            # 解冻 head 模块
            if any(key in name for key in head_modules):
                param.requires_grad = True

        # 打印结果检查
        trainable = [n for n, p in self.agent.named_parameters() if p.requires_grad]
        print(f"Total trainable params: {len(trainable)}")
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