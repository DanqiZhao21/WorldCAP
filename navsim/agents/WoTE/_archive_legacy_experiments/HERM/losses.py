from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from navsim.agents.WoTE.HERM.geometry import wrap_angle


def herm_loss(
    pred_exec: torch.Tensor,
    target_exec: torch.Tensor,
    residual: torch.Tensor,
    params: torch.Tensor,
    w_xy: float = 1.0,
    w_yaw: float = 0.2,
    w_residual_smooth: float = 0.01,
    w_param_smooth: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute HERM trajectory and regularization losses."""
    if pred_exec.shape != target_exec.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred_exec.shape)} vs {tuple(target_exec.shape)}")

    pos_l1 = F.l1_loss(pred_exec[..., :2], target_exec[..., :2], reduction="mean")
    yaw_l1 = torch.mean(torch.abs(wrap_angle(pred_exec[..., 2] - target_exec[..., 2])))

    if residual.shape[-2] > 1:
        residual_smooth = torch.mean(torch.abs(residual[..., 1:, :] - residual[..., :-1, :]))
    else:
        residual_smooth = residual.new_tensor(0.0)

    if params.shape[-2] > 1:
        param_smooth = torch.mean(torch.abs(params[..., 1:, :] - params[..., :-1, :]))
    else:
        param_smooth = params.new_tensor(0.0)

    loss = (
        float(w_xy) * pos_l1
        + float(w_yaw) * yaw_l1
        + float(w_residual_smooth) * residual_smooth
        + float(w_param_smooth) * param_smooth
    )
    metrics = {
        "loss": loss.detach(),
        "pos_l1": pos_l1.detach(),
        "yaw_l1": yaw_l1.detach(),
        "residual_smooth": residual_smooth.detach(),
        "param_smooth": param_smooth.detach(),
    }
    return loss, metrics
