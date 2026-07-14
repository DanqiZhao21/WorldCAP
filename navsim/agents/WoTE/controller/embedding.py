"""Trajectory-error embedding used to condition the WoTE world model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

# ------------------------
# Utility functions
# ------------------------
def wrap_angle_np(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi

def wrap_angle_torch(a: torch.Tensor) -> torch.Tensor:
    return (a + np.pi) % (2 * np.pi) - np.pi

def compute_speed_from_xy(x: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
    # x,y: (..., T)
    dx = np.diff(x, axis=-1)
    dy = np.diff(y, axis=-1)
    v = np.sqrt(dx**2 + dy**2) / (dt + 1e-9)
    # pad last to keep length T
    v = np.concatenate([v, v[..., -1:]], axis=-1)
    return v

def finite_diff(a: np.ndarray, dt: float, axis=-1) -> np.ndarray:
    da = np.diff(a, axis=axis) / (dt + 1e-9)
    da = np.concatenate([da, np.take(da, indices=[-1], axis=axis)], axis=axis)
    return da

# ------------------------
# ControllerEmbedding Module
# ------------------------
class ControllerEmbedding(nn.Module):
    """
    ControllerEmbedding:
        Input: ref_traj (B, T, 3) and exec_traj (B, T, 3) ; columns: x, y, theta (rad)
        Output: embedding (B, emb_dim)

    Design:
      - compute per-timestep error features (frenet lateral error, heading error, speed error)
      - estimate derived control commands: accel (dv/dt) and steering_rate (d(delta)/dt approximate by dtheta)
      - normalize / aggregate features -> produce per-timestep feature vector
      - temporal encoder (1D conv + transformer pooling) -> final embedding
      - emb_dim is configurable

    Usage:
        encoder = ControllerEmbedding(emb_dim=64)
        embedding = encoder(ref_traj, exec_traj, dt=0.1)  # returns (B, emb_dim)
    """
    def __init__(
        self,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        transformer_layers: int = 2,
        dt: float = 0.1,
        use_transformer_pool: bool = True,
        feature_mode: str = "full",
        # device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.use_transformer_pool = use_transformer_pool
        self.feature_mode = feature_mode
        # self.device = device or torch.device("cpu")

        # per-timestep projector (hand-crafted features -> hidden)
        # we'll compute a vector of size feat_dim per timestep; choose feat_dim ~ 12
        self.feat_dim = 12

        # small MLP to lift per-timestep features
        self.feat_proj = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # temporal encoder: 1D conv stack to get local temporal patterns
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # optional transformer pool (global context)
        if use_transformer_pool:
            enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, batch_first=True)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        else:
            self.transformer = None

        # final pooling & projection to embedding
        self.pool = nn.AdaptiveAvgPool1d(1)  # over time dim
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, emb_dim),
        )

        # 预先构建特征掩码：在不改变网络结构的情况下做“只看横向”
        # feat order:
        # [0 e_y, 1 e_psi, 2 dv, 3 ev, 4 rv, 5 kappa, 6 a_exec, 7 steering_rate_est,
        #  8 (ex-rx), 9 (ey-ry), 10 sin(dtheta), 11 cos(dtheta)]
        mode = (feature_mode or "full").lower()
        if mode in {"lateral_only", "lateral", "lat", "y_only"}:
            keep = torch.zeros(self.feat_dim, dtype=torch.float32)
            for idx in [0, 1, 5, 7, 9, 10, 11]:
                keep[idx] = 1.0
            self.register_buffer("_feat_mask", keep.view(1, 1, -1), persistent=False)
        else:
            self.register_buffer("_feat_mask", torch.ones(1, 1, self.feat_dim, dtype=torch.float32), persistent=False)

    # ------------------------
    # Feature extraction helpers (numpy and torch variants)
    # ------------------------
    def _compute_handcrafted_features_numpy(self, ref_traj: np.ndarray, exec_traj: np.ndarray) -> np.ndarray:
        """
        Args:
            ref_traj: (B, T, 3) numpy
            exec_traj: (B, T, 3) numpy
        Returns:
            feats: (B, T, feat_dim) numpy
        """
        B, T, _ = ref_traj.shape
        feats = np.zeros((B, T, self.feat_dim), dtype=np.float32)

        for b in range(B):
            rx = ref_traj[b, :, 0]
            ry = ref_traj[b, :, 1]
            rtheta = ref_traj[b, :, 2]

            ex = exec_traj[b, :, 0]
            ey = exec_traj[b, :, 1]
            etheta = exec_traj[b, :, 2]

            # speed
            rv = compute_speed_from_xy(rx, ry, self.dt)
            ev = compute_speed_from_xy(ex, ey, self.dt)
            dv = ev - rv  # speed error

            # lateral & heading errors (Frenet using ref pose at same index)
            dx = ex - rx
            dy = ey - ry
            e_y = -dx * np.sin(rtheta) + dy * np.cos(rtheta)  # lateral error (left positive)
            e_psi = wrap_angle_np(etheta - rtheta)

            # curvature of reference (feedforward steering)
            # approximate curvature numerically on ref traj
            kappa = np.zeros(T)
            if T >= 3:
                # simple finite differences
                dx_r = np.gradient(rx, self.dt)
                dy_r = np.gradient(ry, self.dt)
                ddx_r = np.gradient(dx_r, self.dt)
                ddy_r = np.gradient(dy_r, self.dt)
                kappa = (dx_r * ddy_r - dy_r * ddx_r) / np.maximum((dx_r**2 + dy_r**2)**1.5, 1e-9)
                kappa = np.nan_to_num(kappa)

            # estimate accel and steering_rate of exec_traj by finite diff
            a_exec = finite_diff(ev, self.dt)
            # approximate steering rate via heading derivative: dtheta/dt (coarse)
            steering_rate_est = finite_diff(etheta, self.dt)
            steering_rate_est = wrap_angle_np(steering_rate_est)

            # assemble features per timestep (suggested order)
            # [e_y, e_psi, dv, ev, rv, kappa, a_exec, steering_rate_est, ex-rx, ey-ry, theta_diff_sin, theta_diff_cos]
            theta_diff_sin = np.sin(etheta - rtheta)
            theta_diff_cos = np.cos(etheta - rtheta)

            feats[b, :, 0] = e_y
            feats[b, :, 1] = e_psi
            feats[b, :, 2] = dv
            feats[b, :, 3] = ev
            feats[b, :, 4] = rv
            feats[b, :, 5] = kappa
            feats[b, :, 6] = a_exec
            feats[b, :, 7] = steering_rate_est
            feats[b, :, 8] = ex - rx
            feats[b, :, 9] = ey - ry
            feats[b, :, 10] = theta_diff_sin
            feats[b, :, 11] = theta_diff_cos

        return feats

    def _to_tensor_features(self, ref_traj: torch.Tensor, exec_traj: torch.Tensor) -> torch.Tensor:
        """
        Compute features on torch tensors (batch).
        Inputs:
            ref_traj, exec_traj: (B, T, 3) torch.Tensor (float)
        Returns:
            feats: (B, T, feat_dim) torch.Tensor
        """
        # convert to numpy for robust geometry ops when batch small OR do direct torch ops
        # We'll implement in torch for speed / differentiability
        B, T, _ = ref_traj.shape
        rx = ref_traj[..., 0]
        ry = ref_traj[..., 1]
        rtheta = ref_traj[..., 2]

        ex = exec_traj[..., 0]
        ey = exec_traj[..., 1]
        etheta = exec_traj[..., 2]

        # speeds
        dxr = rx[..., 1:] - rx[..., :-1]
        dyr = ry[..., 1:] - ry[..., :-1]
        dxr = torch.cat([dxr, dxr[..., -1:]], dim=-1)
        dyr = torch.cat([dyr, dyr[..., -1:]], dim=-1)
        rv = torch.sqrt(dxr**2 + dyr**2) / (self.dt + 1e-9)

        dxe = ex[..., 1:] - ex[..., :-1]
        dye = ey[..., 1:] - ey[..., :-1]
        dxe = torch.cat([dxe, dxe[..., -1:]], dim=-1)
        dye = torch.cat([dye, dye[..., -1:]], dim=-1)
        ev = torch.sqrt(dxe**2 + dye**2) / (self.dt + 1e-9)

        dv = ev - rv

        # lateral and heading errors (Frenet)
        e_y = - (ex - rx) * torch.sin(rtheta) + (ey - ry) * torch.cos(rtheta)
        e_psi = wrap_angle_torch(etheta - rtheta)

        # curvature approx for ref using gradients (torch.gradient not available; use finite diffs)
        # central differences for interior, forward/back for ends
        # We'll compute dx/dt and ddx/dt twice
        dxr_f = rx[..., 1:] - rx[..., :-1]
        dxr_f = torch.cat([dxr_f, dxr_f[..., -1:]], dim=-1)
        dyr_f = ry[..., 1:] - ry[..., :-1]
        dyr_f = torch.cat([dyr_f, dyr_f[..., -1:]], dim=-1)
        vx = dxr_f / (self.dt + 1e-9)
        vy = dyr_f / (self.dt + 1e-9)
        ddx = (vx[..., 1:] - vx[..., :-1]) / (self.dt + 1e-9) if vx.shape[-1] > 1 else torch.zeros_like(vx)
        ddy = (vy[..., 1:] - vy[..., :-1]) / (self.dt + 1e-9) if vy.shape[-1] > 1 else torch.zeros_like(vy)
        if ddx.shape[-1] > 0:
            ddx = torch.cat([ddx, ddx[..., -1:]], dim=-1)
            ddy = torch.cat([ddy, ddy[..., -1:]], dim=-1)
            ddx = torch.cat([ddx, ddx[..., -1:]], dim=-1) if ddx.shape[-1] < T else ddx
            ddy = torch.cat([ddy, ddy[..., -1:]], dim=-1) if ddy.shape[-1] < T else ddy
        else:
            ddx = torch.zeros_like(vx)
            ddy = torch.zeros_like(vy)
        denom = torch.clamp((vx**2 + vy**2)**1.5, min=1e-9)
        kappa = (vx * ddy - vy * ddx) / denom
        kappa = torch.nan_to_num(kappa)

        # estimate accel and steering_rate for exec traj
        a_exec = (ev[..., 1:] - ev[..., :-1]) / (self.dt + 1e-9)
        a_exec = torch.cat([a_exec, a_exec[..., -1:]], dim=-1)
        steering_rate_est = (etheta[..., 1:] - etheta[..., :-1]) / (self.dt + 1e-9)
        steering_rate_est = torch.cat([steering_rate_est, steering_rate_est[..., -1:]], dim=-1)
        steering_rate_est = wrap_angle_torch(steering_rate_est)

        theta_diff_sin = torch.sin(etheta - rtheta)
        theta_diff_cos = torch.cos(etheta - rtheta)

        feats = torch.stack([
            e_y,
            e_psi,
            dv,
            ev,
            rv,
            kappa,
            a_exec,
            steering_rate_est,
            (ex - rx),
            (ey - ry),
            theta_diff_sin,
            theta_diff_cos
        ], dim=-1)  # (B, T, feat_dim)

        return feats

    # ------------------------
    # Forward / main API
    # ------------------------
    def forward(self, ref_traj: torch.Tensor, exec_traj: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        """
        ref_traj, exec_traj: torch.Tensor, shape (B, T, 3)
        returns: embedding: shape (B, emb_dim)
        """
        if dt is not None:
            self.dt = dt

        # ensure float tensors
        device = ref_traj.device
        ref_traj = ref_traj.to(device=device).float()
        exec_traj = exec_traj.to(device=device).float()

        # compute features (B, T, feat_dim)
        feats = self._to_tensor_features(ref_traj, exec_traj)  # (B, T, feat_dim)
        if hasattr(self, "_feat_mask") and self._feat_mask is not None:
            feats = feats * self._feat_mask.to(device=feats.device, dtype=feats.dtype)

        # project per-timestep
        B, T, D = feats.shape
        feats_proj = self.feat_proj(feats.view(B * T, D)).view(B, T, -1)  # (B, T, hidden_dim)

        # temporal conv expects (B, C, T)
        x = feats_proj.permute(0, 2, 1)  # (B, hidden_dim, T)
        x = self.temporal_conv(x)  # (B, hidden_dim, T)

        # transformer expects (B, T, C)
        x_t = x.permute(0, 2, 1).contiguous()  # (B, T, hidden_dim)
        if self.transformer is not None:
            x_t = self.transformer(x_t)  # (B, T, hidden_dim)

        # pooling & final projection
        x_pool = x_t.permute(0, 2, 1)  # (B, hidden_dim, T)
        pooled = self.pool(x_pool).squeeze(-1)  # (B, hidden_dim)
        emb = self.final_proj(pooled)  # (B, emb_dim)#这里的B就是num_traj
        return emb

# ------------------------
# Simple training / usage helper
# ------------------------
def sample_training_loss(embedding_net: nn.Module, ref_traj: torch.Tensor, exec_traj: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """
    Example self-supervised objective:
      - reconstruct per-timestep residual (exec_traj - ref_traj) from (ref_traj, embedding)
      - minimize L2 between predicted residuals and actual residuals
    This produces a training signal for the embedding network; the reconstructor is a small MLP + upsampler.

    Returns loss and diagnostics dict.
    """
    B, T, _ = ref_traj.shape
    emb = embedding_net(ref_traj, exec_traj)  # (B, emb_dim)

    # simple reconstructor
    recon_mlp = nn.Sequential(
        nn.Linear(embedding_net.emb_dim, 128),
        nn.ReLU(),
        nn.Linear(128, T * 3)
    ).to(emb.device)

    pred = recon_mlp(emb).view(B, T, 3)  # predicted residual
    residual_gt = exec_traj - ref_traj
    loss = F.mse_loss(pred, residual_gt)

    return loss, {"mse": loss.item()}

# ------------------------
# Small example (when module executed directly)
# ------------------------
if __name__ == "__main__":


    # load your trajectories
    ref = np.load("/home/zhaodanqi/clone/DiffusionDrive/extra_data/planning_vb/trajectory_anchors_256.npy")  # shape [B, T, 3]
    exec_t = np.load("/home/zhaodanqi/clone/DiffusionDrive/extra_data/planning_vb/trajectory_exec_256.npy")  # shape [B, T, 3]

    # convert to torch tensors
    ref = torch.from_numpy(ref).float()
    exec_t = torch.from_numpy(exec_t).float()

    encoder = ControllerEmbedding(emb_dim=64)
    emb = encoder(ref, exec_t)
    print("embedding shape:", emb.shape)  # (B, emb_dim)
