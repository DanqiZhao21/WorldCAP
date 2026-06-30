import math

import torch

from navsim.agents.WoTE.HERM.geometry import (
    cumulative_arc_length,
    frenet_to_cartesian,
    project_to_plan_frenet,
    trajectory_intrinsics,
    wrap_angle,
)


def test_wrap_angle_bounds():
    angles = torch.tensor([-3 * math.pi, -math.pi, 0.0, math.pi, 3 * math.pi])
    wrapped = wrap_angle(angles)

    assert torch.all(wrapped >= -math.pi)
    assert torch.all(wrapped < math.pi)


def test_cumulative_arc_length_straight_line():
    traj = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ]
    )

    s = cumulative_arc_length(traj)

    assert torch.allclose(s, torch.tensor([[0.0, 1.0, 3.0]]), atol=1e-5)


def test_trajectory_intrinsics_shapes():
    traj = torch.zeros((2, 8, 3), dtype=torch.float32)
    traj[:, :, 0] = torch.arange(8, dtype=torch.float32)

    feat = trajectory_intrinsics(traj, dt=0.5)

    assert set(feat.keys()) == {"s", "v", "a", "kappa", "kappa_dot", "omega"}
    assert feat["s"].shape == (2, 8)
    assert feat["v"].shape == (2, 8)
    assert feat["a"].shape == (2, 8)
    assert feat["kappa"].shape == (2, 8)
    assert feat["kappa_dot"].shape == (2, 8)
    assert feat["omega"].shape == (2, 8)


def test_frenet_round_trip_for_identical_trajectory():
    plan = torch.zeros((1, 8, 3), dtype=torch.float32)
    plan[0, :, 0] = torch.arange(8, dtype=torch.float32)
    exec_traj = plan.clone()

    residual = project_to_plan_frenet(plan, exec_traj)
    recon = frenet_to_cartesian(
        plan,
        residual["s_exec"],
        residual["d_exec"],
        residual["theta_exec"],
    )

    assert torch.allclose(residual["delta_s"], torch.zeros((1, 8)), atol=1e-5)
    assert torch.allclose(residual["delta_d"], torch.zeros((1, 8)), atol=1e-5)
    assert torch.allclose(residual["delta_theta"], torch.zeros((1, 8)), atol=1e-5)
    assert torch.allclose(recon, exec_traj, atol=1e-5)


def test_frenet_projection_lateral_sign():
    plan = torch.zeros((1, 4, 3), dtype=torch.float32)
    plan[0, :, 0] = torch.arange(4, dtype=torch.float32)
    exec_traj = plan.clone()
    exec_traj[:, :, 1] = 0.5

    residual = project_to_plan_frenet(plan, exec_traj)

    assert torch.allclose(residual["delta_d"], torch.full((1, 4), 0.5), atol=1e-5)


def test_frenet_projection_uses_same_timestep_longitudinal_error():
    plan = torch.zeros((1, 4, 3), dtype=torch.float32)
    plan[0, :, 0] = torch.arange(4, dtype=torch.float32)
    exec_traj = plan.clone()
    exec_traj[0, 1, 0] = 1.3

    residual = project_to_plan_frenet(plan, exec_traj)

    assert torch.allclose(residual["delta_s"][0, 1], torch.tensor(0.3), atol=1e-5)
    assert torch.allclose(residual["delta_d"][0, 1], torch.tensor(0.0), atol=1e-5)
