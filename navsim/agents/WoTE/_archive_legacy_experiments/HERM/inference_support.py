from __future__ import annotations

from dataclasses import fields

import torch

from navsim.agents.WoTE.HERM.model import HERMConfig, SupportConditionalHERM


def _config_from_checkpoint(data: dict) -> HERMConfig:
    raw = data.get("config", {})
    valid = {f.name for f in fields(HERMConfig)}
    return HERMConfig(**{k: v for k, v in raw.items() if k in valid})


def load_support_herm_checkpoint(path: str, device: torch.device | str = "cpu") -> SupportConditionalHERM:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = _config_from_checkpoint(checkpoint)
    model = SupportConditionalHERM(
        config,
        style_emb_dim=int(checkpoint.get("style_emb_dim", config.controller_emb_dim)),
        style_hidden_dim=int(checkpoint.get("style_hidden_dim", 128)),
    )
    state = checkpoint.get("model_state", checkpoint.get("model_state_dict", None))
    if state is None:
        raise KeyError("HERM checkpoint must contain 'model_state' or 'model_state_dict'")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def execute_with_support_herm(
    model: SupportConditionalHERM,
    support_plan: torch.Tensor,
    support_exec: torch.Tensor,
    query_plan: torch.Tensor,
) -> torch.Tensor:
    """Predict executed query trajectories from same-style support pairs."""
    was_training = model.training
    model.eval()
    try:
        return model(support_plan, support_exec, query_plan).exec_traj
    finally:
        if was_training:
            model.train()
