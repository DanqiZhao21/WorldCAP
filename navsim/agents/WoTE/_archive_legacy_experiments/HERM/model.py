from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from navsim.agents.WoTE.HERM.geometry import (
    cumulative_arc_length,
    frenet_to_cartesian,
    project_to_plan_frenet,
    trajectory_intrinsics,
)


@dataclass
class HERMConfig:
    num_poses: int = 8
    dt: float = 0.5
    intrinsic_dim: int = 5
    controller_emb_dim: int = 0
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    max_alpha: float = 0.98
    max_beta: float = 2.0
    max_bias_s: float = 2.0
    max_bias_d: float = 2.0
    max_bias_theta: float = 1.0


@dataclass
class HERMOutput:
    exec_traj: torch.Tensor
    residual: torch.Tensor
    params: torch.Tensor
    intrinsics: Optional[Dict[str, torch.Tensor]] = None


class FrenetErrorDynamicsHERM(nn.Module):
    """Hybrid execution response model with structured Frenet residual rollout."""

    def __init__(self, config: HERMConfig) -> None:
        super().__init__()
        self.config = config

        input_dim = int(config.intrinsic_dim) + int(config.controller_emb_dim)#计划轨迹本身的几何/运动属性 + HERM 接收的条件向量维度
        hidden_dim = int(config.hidden_dim) 
        num_layers = max(1, int(config.num_layers)) #
        dropout = float(config.dropout) if num_layers > 1 else 0.0 #

        self.input_proj = nn.Sequential( 
            nn.Linear(input_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.param_head = nn.Linear(hidden_dim, 9)

    def forward(
        self,
        plan_traj: torch.Tensor,
        controller_emb: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> HERMOutput:
        if plan_traj.ndim != 3 or plan_traj.shape[-1] < 3:
            raise ValueError(f"plan_traj must be [B,T,3], got {tuple(plan_traj.shape)}")
        if int(plan_traj.shape[1]) != int(self.config.num_poses):
            raise ValueError(
                f"plan_traj horizon mismatch: expected {self.config.num_poses}, got {plan_traj.shape[1]}"
            )

        batch_size, horizon, _ = plan_traj.shape
        intrinsics = trajectory_intrinsics(plan_traj, dt=float(self.config.dt))
        intrinsic_feat = torch.stack(
            [
                intrinsics["v"],
                intrinsics["a"],
                intrinsics["kappa"],
                intrinsics["kappa_dot"],
                intrinsics["omega"],#角速度
            ],
            dim=-1,
        )

        if int(self.config.controller_emb_dim) > 0:
            if controller_emb is None:
                raise ValueError("controller_emb is required when controller_emb_dim > 0")
            if controller_emb.ndim != 2:
                raise ValueError(f"controller_emb must be [B,D], got {tuple(controller_emb.shape)}")
            if controller_emb.shape[0] != batch_size or controller_emb.shape[1] != int(self.config.controller_emb_dim):
                raise ValueError(
                    "controller_emb shape mismatch: "
                    f"expected [{batch_size},{self.config.controller_emb_dim}], got {tuple(controller_emb.shape)}"
                )
            ctrl_feat = controller_emb.to(device=plan_traj.device, dtype=plan_traj.dtype)
            ctrl_feat = ctrl_feat.unsqueeze(1).expand(batch_size, horizon, -1)
            model_input = torch.cat([intrinsic_feat, ctrl_feat], dim=-1)
        else:
            model_input = intrinsic_feat

        encoded_input = self.input_proj(model_input[:, :-1, :])
        encoded, _ = self.temporal_encoder(encoded_input)
        params = self._bound_params(self.param_head(encoded))
        residual = self._rollout_residual(params, intrinsics)

        s_plan = cumulative_arc_length(plan_traj)
        s_exec = s_plan + residual[..., 0]
        d_exec = residual[..., 1]
        theta_exec = plan_traj[..., 2] + residual[..., 2]
        exec_traj = frenet_to_cartesian(plan_traj, s_exec, d_exec, theta_exec)

        return HERMOutput(
            exec_traj=exec_traj,
            residual=residual,
            params=params,
            intrinsics=intrinsics if return_debug else None,
        )

    def _bound_params(self, raw: torch.Tensor) -> torch.Tensor:
        alpha = float(self.config.max_alpha) * torch.tanh(raw[..., [0, 3, 6]])
        beta = float(self.config.max_beta) * torch.tanh(raw[..., [1, 4, 7]])
        b_s = float(self.config.max_bias_s) * torch.tanh(raw[..., 2:3])
        b_d = float(self.config.max_bias_d) * torch.tanh(raw[..., 5:6])
        b_theta = float(self.config.max_bias_theta) * torch.tanh(raw[..., 8:9])
        return torch.cat(
            [
                alpha[..., 0:1],
                beta[..., 0:1],
                b_s,
                alpha[..., 1:2],
                beta[..., 1:2],
                b_d,
                alpha[..., 2:3],
                beta[..., 2:3],
                b_theta,
            ],
            dim=-1,
        )

    def _rollout_residual(self, params: torch.Tensor, intrinsics: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, steps, _ = params.shape
        device = params.device
        dtype = params.dtype

        state = torch.zeros((batch_size, 3), device=device, dtype=dtype)
        states = [state]
        a = intrinsics["a"][:, :-1].to(device=device, dtype=dtype)
        omega = intrinsics["omega"][:, :-1].to(device=device, dtype=dtype)

        for t in range(steps):
            p = params[:, t, :]
            ds = p[:, 0] * state[:, 0] + p[:, 1] * a[:, t] + p[:, 2]
            dd = p[:, 3] * state[:, 1] + p[:, 4] * omega[:, t] + p[:, 5]
            dtheta = p[:, 6] * state[:, 2] + p[:, 7] * omega[:, t] + p[:, 8]
            state = torch.stack([ds, dd, dtheta], dim=-1)
            states.append(state)

        return torch.stack(states, dim=1)


class SupportStyleEncoder(nn.Module):
    """Encode support ref/exec pairs into a controller-style embedding."""

    def __init__(self, config: HERMConfig, emb_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.config = config
        self.emb_dim = int(emb_dim)
        feature_dim = int(config.intrinsic_dim) + 3
        hidden_dim = int(hidden_dim)
        self.pair_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
        )
        self.support_pool = SupportAttentionPool(self.emb_dim)

    def forward(self, support_plan: torch.Tensor, support_exec: torch.Tensor) -> torch.Tensor:
        if support_plan.ndim != 4 or support_exec.ndim != 4:
            raise ValueError("support_plan and support_exec must be [B,S,T,3]")
        if support_plan.shape != support_exec.shape:
            raise ValueError(
                f"support shape mismatch: plan={tuple(support_plan.shape)} exec={tuple(support_exec.shape)}"
            )

        batch, support, horizon, dim = support_plan.shape
        flat_plan = support_plan.reshape(batch * support, horizon, dim)
        flat_exec = support_exec.reshape(batch * support, horizon, dim)
        intrinsics = trajectory_intrinsics(flat_plan, dt=float(self.config.dt))
        intrinsic_feat = torch.stack(
            [
                intrinsics["v"],
                intrinsics["a"],
                intrinsics["kappa"],
                intrinsics["kappa_dot"],
                intrinsics["omega"],
            ],
            dim=-1,
        )
        residual = project_to_plan_frenet(flat_plan, flat_exec)
        residual_feat = torch.stack(
            [residual["delta_s"], residual["delta_d"], residual["delta_theta"]],
            dim=-1,
        )
        pair_feat = torch.cat([intrinsic_feat, residual_feat], dim=-1)
        pair_emb = self.pair_encoder(pair_feat.transpose(1, 2)).reshape(batch, support, self.emb_dim)
        return self.support_pool(pair_emb)


class SupportAttentionPool(nn.Module):
    """Learn weighted pooling over support pair embeddings."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(int(emb_dim), int(emb_dim)),
            nn.Tanh(),
            nn.Linear(int(emb_dim), 1),
        )

    def forward(self, pair_emb: torch.Tensor) -> torch.Tensor:
        if pair_emb.ndim != 3:
            raise ValueError(f"pair_emb must be [B,S,D], got {tuple(pair_emb.shape)}")
        attn = self.score(pair_emb)
        weight = torch.softmax(attn, dim=1)
        return torch.sum(weight * pair_emb, dim=1)


class SupportConditionalHERM(nn.Module):
    """HERM conditioned on same-style support trajectory pairs."""

    def __init__(
        self,
        config: HERMConfig,
        style_emb_dim: int = 64,
        style_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if int(config.controller_emb_dim) != int(style_emb_dim):
            config = HERMConfig(
                num_poses=config.num_poses,
                dt=config.dt,
                intrinsic_dim=config.intrinsic_dim,
                controller_emb_dim=int(style_emb_dim),
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
                max_alpha=config.max_alpha,
                max_beta=config.max_beta,
                max_bias_s=config.max_bias_s,
                max_bias_d=config.max_bias_d,
                max_bias_theta=config.max_bias_theta,
            )
        self.config = config
        self.style_encoder = SupportStyleEncoder(config, emb_dim=style_emb_dim, hidden_dim=style_hidden_dim)
        self.herm = FrenetErrorDynamicsHERM(config)

    def forward(
        self,
        support_plan: torch.Tensor,
        support_exec: torch.Tensor,
        query_plan: torch.Tensor,
        return_debug: bool = False,
    ) -> HERMOutput:
        if query_plan.ndim != 4:
            raise ValueError(f"query_plan must be [B,Q,T,3], got {tuple(query_plan.shape)}")
        batch, query, horizon, dim = query_plan.shape
        style_emb = self.style_encoder(support_plan, support_exec)
        flat_query = query_plan.reshape(batch * query, horizon, dim)
        flat_style = style_emb[:, None, :].expand(batch, query, -1).reshape(batch * query, -1)
        flat_out = self.herm(flat_query, controller_emb=flat_style, return_debug=return_debug)
        intrinsics = flat_out.intrinsics if return_debug else None
        return HERMOutput(
            exec_traj=flat_out.exec_traj.reshape(batch, query, horizon, dim),
            residual=flat_out.residual.reshape(batch, query, horizon, 3),
            params=flat_out.params.reshape(batch, query, horizon - 1, 9),
            intrinsics=intrinsics,
        )
