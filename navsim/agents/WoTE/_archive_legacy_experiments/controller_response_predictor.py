import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class ControllerResponsePredictor(nn.Module):
    """Predict tracking residual (exec - ref) for a given controller embedding.

    This is intended as a plug-in module for controller-aware planning:
        g(a, phi) -> \hat{\Delta a}

    Where:
        a: reference trajectory candidate (T, 3) in ego frame (x, y, yaw)
        phi: controller dynamics embedding (D)

    Output:
        residual_hat: (T, 3)

    Notes:
        - This module does *not* require scene/BEV.
        - It can be trained purely from (ref_traj, exec_traj) pairs.
    """

    def __init__(
        self,
        num_poses: int = 8,
        traj_dim: int = 3,
        controller_emb_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_poses = int(num_poses)
        self.traj_dim = int(traj_dim)
        self.controller_emb_dim = int(controller_emb_dim)

        in_dim = self.num_poses * self.traj_dim

        self.traj_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.controller_encoder = nn.Sequential(
            nn.Linear(self.controller_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.fuser = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Linear(hidden_dim, in_dim)

    def forward(self, ref_traj: torch.Tensor, controller_emb: torch.Tensor) -> torch.Tensor:
        """Args:
            ref_traj: [B, T, 3]
            controller_emb: [B, D]

        Returns:
            residual_hat: [B, T, 3]
        """
        if ref_traj.ndim != 3:
            raise ValueError(f"ref_traj must be [B,T,3], got {tuple(ref_traj.shape)}")
        if controller_emb.ndim != 2:
            raise ValueError(f"controller_emb must be [B,D], got {tuple(controller_emb.shape)}")

        B, T, D = ref_traj.shape
        if T != self.num_poses or D != self.traj_dim:
            raise ValueError(
                f"ref_traj shape mismatch: expected [B,{self.num_poses},{self.traj_dim}], got {tuple(ref_traj.shape)}"
            )
        if controller_emb.shape[0] != B:
            raise ValueError("Batch size mismatch between ref_traj and controller_emb")

        x = ref_traj.reshape(B, -1)
        t_feat = self.traj_encoder(x)
        c_feat = self.controller_encoder(controller_emb)
        fused = self.fuser(torch.cat([t_feat, c_feat], dim=-1))
        out = self.head(fused).view(B, self.num_poses, self.traj_dim)
        return out

    @torch.no_grad()
    def compute_risk(
        self,
        residual_hat: torch.Tensor,
        xy_weight: float = 1.0,
        yaw_weight: float = 0.2,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute a scalar risk per trajectory from predicted residual.

        Args:
            residual_hat: [B, T, 3]
            xy_weight: weight for positional error
            yaw_weight: weight for yaw error
            reduction: 'mean' (default) or 'sum'

        Returns:
            risk: [B]
        """
        if residual_hat.ndim != 3 or residual_hat.shape[-1] < 3:
            raise ValueError(f"residual_hat must be [B,T,3], got {tuple(residual_hat.shape)}")

        xy = residual_hat[..., :2]
        yaw = residual_hat[..., 2]

        # per-timestep magnitude
        xy_mag = torch.sqrt((xy ** 2).sum(dim=-1) + 1e-6)  # [B, T]
        yaw_mag = torch.abs(yaw)  # [B, T]

        if reduction == "sum":
            xy_r = xy_mag.sum(dim=-1)
            yaw_r = yaw_mag.sum(dim=-1)
        else:
            xy_r = xy_mag.mean(dim=-1)
            yaw_r = yaw_mag.mean(dim=-1)

        return (float(xy_weight) * xy_r) + (float(yaw_weight) * yaw_r)


def load_controller_response_checkpoint(
    module: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Load a checkpoint that may include both controller_encoder and response_predictor."""
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
    return ckpt
