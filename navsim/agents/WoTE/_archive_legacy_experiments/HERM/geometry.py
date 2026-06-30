from __future__ import annotations

import math
from typing import Dict

import torch

#HERM 的数学基础，负责轨迹几何、角度处理、Frenet 坐标转换。

def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def unwrap_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Unwrap yaw along the last dimension."""
    if yaw.shape[-1] <= 1:
        return yaw
    delta = wrap_angle(yaw[..., 1:] - yaw[..., :-1])
    first = yaw[..., :1]
    return torch.cat([first, first + torch.cumsum(delta, dim=-1)], dim=-1)

#根据 (x, y) 算每段轨迹长度，再累计成弧长 s
def pairwise_segment_lengths(traj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return XY segment lengths for a trajectory shaped [..., T, 3]."""
    if traj.shape[-1] < 2:
        raise ValueError(f"trajectory last dimension must include xy, got {tuple(traj.shape)}")
    delta_xy = traj[..., 1:, :2] - traj[..., :-1, :2]
    return torch.sqrt(torch.sum(delta_xy * delta_xy, dim=-1).clamp_min(eps))


def cumulative_arc_length(traj: torch.Tensor) -> torch.Tensor:
    """Return cumulative arc length shaped [..., T]."""
    seg = pairwise_segment_lengths(traj)
    zero = torch.zeros(*traj.shape[:-2], 1, device=traj.device, dtype=traj.dtype)
    return torch.cat([zero, torch.cumsum(seg, dim=-1)], dim=-1)


def finite_difference(values: torch.Tensor, step: torch.Tensor | float, eps: float = 1e-6) -> torch.Tensor:
    """Forward finite difference padded to the original length."""
    if values.shape[-1] <= 1:
        return torch.zeros_like(values)
    delta = values[..., 1:] - values[..., :-1]
    if isinstance(step, torch.Tensor):
        denom = step.clamp_min(eps)
    else:
        denom = max(float(step), eps)
    diff = delta / denom
    return torch.cat([diff, diff[..., -1:]], dim=-1)

#从计划轨迹里提取五个运动特征
def trajectory_intrinsics(traj: torch.Tensor, dt: float = 0.5, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Compute trajectory-intrinsic features from [B, T, 3] poses."""
    if traj.ndim < 3 or traj.shape[-1] < 3:
        raise ValueError(f"traj must be shaped [..., T, 3], got {tuple(traj.shape)}")

    s = cumulative_arc_length(traj)
    yaw = unwrap_yaw(traj[..., :, 2])

    if traj.shape[-2] <= 1:
        zeros = torch.zeros_like(s)
        return {"s": s, "v": zeros, "a": zeros, "kappa": zeros, "kappa_dot": zeros, "omega": zeros}

    ds = (s[..., 1:] - s[..., :-1]).clamp_min(eps)
    dtheta = yaw[..., 1:] - yaw[..., :-1]

    v_seg = ds / max(float(dt), eps)
    kappa_seg = dtheta / ds

    v = torch.cat([v_seg, v_seg[..., -1:]], dim=-1)
    kappa = torch.cat([kappa_seg, kappa_seg[..., -1:]], dim=-1)
    a = finite_difference(v, dt, eps=eps)
    kappa_dot = finite_difference(kappa, dt, eps=eps)
    omega = v * kappa

    return {"s": s, "v": v, "a": a, "kappa": kappa, "kappa_dot": kappa_dot, "omega": omega}


def _gather_time(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather values shaped [B, T, C] or [B, T] with index [B, M]."""
    if values.ndim == 2:
        return values.gather(dim=1, index=index)
    if values.ndim == 3:
        expanded = index.unsqueeze(-1).expand(*index.shape, values.shape[-1])
        return values.gather(dim=1, index=expanded)
    raise ValueError(f"unsupported gather values shape {tuple(values.shape)}")

# Same-timestep local Frenet projection around each planned pose.
def project_to_plan_frenet(plan_traj: torch.Tensor, exec_traj: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Approximate executed poses as Frenet residuals against the planned trajectory.

    Projection compares each executed pose to the planned pose at the same
    timestep, then resolves the XY error into local tangent/normal components.
    """
    if plan_traj.ndim != 3 or exec_traj.ndim != 3:
        raise ValueError("plan_traj and exec_traj must both be [B,T,3]")
    if plan_traj.shape != exec_traj.shape:
        raise ValueError(f"shape mismatch: plan={tuple(plan_traj.shape)} exec={tuple(exec_traj.shape)}")

    s_plan = cumulative_arc_length(plan_traj)
    theta_ref = plan_traj[..., 2]
    tangent = torch.stack([torch.cos(theta_ref), torch.sin(theta_ref)], dim=-1)
    normal = torch.stack([-torch.sin(theta_ref), torch.cos(theta_ref)], dim=-1)
    delta_xy = exec_traj[..., :2] - plan_traj[..., :2]
    delta_s = torch.sum(delta_xy * tangent, dim=-1)
    delta_d = torch.sum(delta_xy * normal, dim=-1)
    delta_theta = wrap_angle(exec_traj[..., 2] - plan_traj[..., 2])

    return {
        "delta_s": delta_s,
        "delta_d": delta_d,
        "delta_theta": delta_theta,
        "s_exec": s_plan + delta_s,
        "d_exec": delta_d,
        "theta_exec": exec_traj[..., 2],
    }


def frenet_to_cartesian(
    plan_traj: torch.Tensor,
    s_exec: torch.Tensor,
    d_exec: torch.Tensor,
    theta_exec: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert Frenet coordinates around plan_traj back to Cartesian poses."""
    if plan_traj.ndim != 3:
        raise ValueError(f"plan_traj must be [B,T,3], got {tuple(plan_traj.shape)}")
    if s_exec.shape != d_exec.shape or s_exec.shape != theta_exec.shape:
        raise ValueError("s_exec, d_exec, and theta_exec must have matching shapes")

    s_plan = cumulative_arc_length(plan_traj)
    horizon = plan_traj.shape[1]
    if horizon == 1:
        center = plan_traj[..., :2].expand(*s_exec.shape, 2)
        theta_plan = plan_traj[..., 2].expand_as(s_exec)
    else:
        idx_hi = torch.searchsorted(s_plan.contiguous(), s_exec.contiguous(), right=False)
        idx_hi = idx_hi.clamp(min=1, max=horizon - 1)
        idx_lo = idx_hi - 1

        s0 = _gather_time(s_plan, idx_lo)
        s1 = _gather_time(s_plan, idx_hi)
        weight = ((s_exec - s0) / (s1 - s0).clamp_min(eps)).clamp(0.0, 1.0)

        p0 = _gather_time(plan_traj[..., :2], idx_lo)
        p1 = _gather_time(plan_traj[..., :2], idx_hi)
        center = p0 + weight.unsqueeze(-1) * (p1 - p0)

        yaw = unwrap_yaw(plan_traj[..., 2])
        yaw0 = _gather_time(yaw, idx_lo)
        yaw1 = _gather_time(yaw, idx_hi)
        theta_plan = yaw0 + weight * (yaw1 - yaw0)

    normal = torch.stack([-torch.sin(theta_plan), torch.cos(theta_plan)], dim=-1)
    xy = center + d_exec.unsqueeze(-1) * normal
    yaw_out = wrap_angle(theta_exec)
    return torch.cat([xy, yaw_out.unsqueeze(-1)], dim=-1)
