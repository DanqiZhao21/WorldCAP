from __future__ import annotations

from dataclasses import fields
from typing import Optional

import torch

from navsim.agents.WoTE.HERM.model import HERMConfig, FrenetErrorDynamicsHERM


def _config_from_checkpoint(data: dict) -> HERMConfig:
    raw = data.get("config", {})
    valid = {f.name for f in fields(HERMConfig)}
    return HERMConfig(**{k: v for k, v in raw.items() if k in valid})


def load_herm_checkpoint(path: str, device: torch.device | str = "cpu") -> FrenetErrorDynamicsHERM:
    checkpoint = torch.load(path, map_location=device)
    config = _config_from_checkpoint(checkpoint)
    model = FrenetErrorDynamicsHERM(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def execute_with_herm(
    model: FrenetErrorDynamicsHERM,
    planned_trajs: torch.Tensor,
    controller_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute planned trajectories with HERM, preserving input shape."""
    was_training = model.training
    model.eval()
    try:
        if planned_trajs.ndim == 3:
            return model(planned_trajs, controller_emb=controller_emb).exec_traj
        if planned_trajs.ndim != 4:
            raise ValueError(f"planned_trajs must be [B,T,3] or [B,K,T,3], got {tuple(planned_trajs.shape)}")

        batch, num_traj, horizon, dim = planned_trajs.shape
        flat = planned_trajs.reshape(batch * num_traj, horizon, dim)

        flat_controller = None
        if controller_emb is not None:
            if controller_emb.ndim == 2:
                flat_controller = controller_emb[:, None, :].expand(batch, num_traj, -1).reshape(batch * num_traj, -1)
            elif controller_emb.ndim == 3:
                flat_controller = controller_emb.reshape(batch * num_traj, -1)
            else:
                raise ValueError(f"controller_emb must be [B,D] or [B,K,D], got {tuple(controller_emb.shape)}")

        executed = model(flat, controller_emb=flat_controller).exec_traj
        return executed.reshape(batch, num_traj, horizon, dim)
    finally:
        if was_training:
            model.train()
