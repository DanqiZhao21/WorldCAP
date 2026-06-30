# Frenet Error-Dynamics HERM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new independent HERM module that learns controller-style execution residuals from `(ref_traj, exec_traj)` trajectory pairs, then exposes a differentiable `planned_traj -> predicted_exec_traj` API for later injection before WoTE latent world-model rollout.

**Architecture:** HERM is a trajectory-space hybrid residual model, not a bicycle model. It extracts intrinsic trajectory features from planned/reference trajectories, predicts time-varying compound execution-dynamics parameters, rolls out Frenet residual states `(Delta s, Delta d, Delta theta)`, and converts the residual trajectory back to Cartesian `(x, y, yaw)`.

**Tech Stack:** Python, PyTorch, NumPy, PyTest. It should live under `navsim/agents/WoTE/HERM/` and remain importable without modifying WoTE training until the integration phase.

---

## Scope

This plan implements HERM as a standalone trainable module. It is a new subsystem; it does not depend on, wrap, migrate from, or remain compatible with the old controller response predictor.

Included:

- Differentiable trajectory geometry utilities.
- Dataset loader for `.npy` and `.npz` controller trajectory banks.
- Frenet residual target construction from `(ref_traj, exec_traj)`.
- Neural residual-dynamics model.
- Loss functions and metrics.
- Minimal training/evaluation CLI.
- Unit tests with synthetic trajectories.
- A future WoTE adapter API, but no behavioral modification to `WoTE_model.py` yet.

Not included in this phase:

- Direct modification of `navsim/agents/WoTE/WoTE_model.py`.
- Latent world-model reward changes.
- PDM simulator changes.
- Dynamic bicycle or kinematic bicycle assumptions.

## Current Repository Context

Relevant existing files:

- `navsim/agents/WoTE/WoTE_model.py`
  - Already loads controller ref/exec banks.
  - Already computes controller embeddings from active ref/exec banks.
  - Latent world-model transition happens in `_latent_world_model_processing`.
  - Candidate trajectories are built as `init_trajectory_anchor + trajectory_offset`.
  - Future integration should apply the new HERM module to those candidates before latent rollout/reward scoring.

- `navsim/agents/WoTE/configs/default.py`
  - Later integration should add HERM-specific config names only.
  - No old response-predictor config compatibility is required.

Controller data formats already used in the repo:

- Ref bank `.npy`: `[N, T, 3]`
- Exec bank `.npy`: `[N, T, 3]`
- Style bundle `.npz`:
  - `ref_traj`: `[N, T, 3]`
  - `exec_trajs`: `[S, N, T, 3]`
  - optional `style_names`, `train_style_indices`, `val_style_indices`

Trajectory convention:

- Last dimension is `(x, y, yaw)`.
- Current anchor horizon is usually `T=8`, interval `dt=0.5`.
- Coordinates are ego-frame local trajectories.

## Proposed HERM Formulation

For each planned/reference trajectory:

```text
tau_plan = {(x_t^p, y_t^p, theta_t^p)}_{t=0}^{T-1}
tau_exec = {(x_t^e, y_t^e, theta_t^e)}_{t=0}^{T-1}
```

HERM extracts intrinsic planned features:

```text
r_t = [v_t^p, a_t^p, kappa_t^p, dot_kappa_t^p, omega_t^p]
omega_t^p = v_t^p * kappa_t^p
```

It represents execution in the planned trajectory frame:

```text
e_t = [Delta s_t, Delta d_t, Delta theta_t]
```

The network predicts bounded time-varying rollout parameters:

```text
psi_t = [alpha_s, beta_s, b_s,
         alpha_d, beta_d, b_d,
         alpha_theta, beta_theta, b_theta]_t
```

The structured rollout is:

```text
Delta s_{t+1}     = alpha_s(t)     * Delta s_t     + beta_s(t)     * a_t^p                 + b_s(t)
Delta d_{t+1}     = alpha_d(t)     * Delta d_t     + beta_d(t)     * (v_t^p * kappa_t^p)   + b_d(t)
Delta theta_{t+1} = alpha_theta(t) * Delta theta_t + beta_theta(t) * (v_t^p * kappa_t^p)   + b_theta(t)
```

Initial residual:

```text
e_0 = [0, 0, 0]
```

Predicted executed trajectory:

```text
s_t^exec     = s_t^plan + Delta s_t
d_t^exec     = Delta d_t
theta_t^exec = theta_t^plan(s_t^exec) + Delta theta_t
tau_hat_exec = FrenetToCartesian(tau_plan, s_t^exec, d_t^exec, theta_t^exec)
```

This is intentionally not:

```text
delta = arctan(L * kappa)
```

and does not require wheelbase, steering angle, side-slip, or tire stiffness.

## File Structure

Create:

- `navsim/agents/WoTE/HERM/__init__.py`
  - Public exports for HERM modules.

- `navsim/agents/WoTE/HERM/geometry.py`
  - Differentiable PyTorch geometry utilities:
    - `wrap_angle`
    - `unwrap_yaw`
    - `pairwise_segment_lengths`
    - `cumulative_arc_length`
    - `finite_difference`
    - `trajectory_intrinsics`
    - `project_to_plan_frenet`
    - `frenet_to_cartesian`

- `navsim/agents/WoTE/HERM/model.py`
  - `HERMConfig`
  - `FrenetErrorDynamicsHERM`
  - `HERMOutput`
  - Model forward API:
    - `forward(plan_traj, controller_emb=None, return_debug=False)`
    - returns predicted executed trajectory and residual states.

- `navsim/agents/WoTE/HERM/data.py`
  - `ControllerTrajectoryDataset`
  - `load_controller_pairs`
  - Supports `.npy` pair and `.npz` style bundle.
  - Produces samples:
    - `plan_traj`: `[T, 3]`
    - `exec_traj`: `[T, 3]`
    - `style_idx`: scalar
    - optional `style_name`

- `navsim/agents/WoTE/HERM/losses.py`
  - `herm_loss`
  - Cartesian position loss.
  - Heading loss with angle wrapping.
  - Frenet residual loss.
  - Optional smoothness/stability regularization on rollout parameters.

- `navsim/agents/WoTE/HERM/train.py`
  - CLI entrypoint for offline training from controller trajectory banks.
  - Saves checkpoint containing model state, config, normalization stats, and train metadata.

- `navsim/agents/WoTE/HERM/evaluate.py`
  - CLI entrypoint for MAE/RMSE reporting on held-out styles or held-out trajectories.

- `navsim/agents/WoTE/HERM/inference.py`
  - Lightweight checkpoint loader.
  - `load_herm_checkpoint(path, device)`.
  - `execute_with_herm(model, planned_trajs, controller_emb=None)` supporting `[B,T,3]` and `[B,K,T,3]`.

Create tests:

- `tests/test_herm_geometry.py`
- `tests/test_herm_model.py`
- `tests/test_herm_data.py`

## API Contract

The final HERM module should support:

```python
from navsim.agents.WoTE.HERM import FrenetErrorDynamicsHERM, HERMConfig

config = HERMConfig(num_poses=8, dt=0.5)
model = FrenetErrorDynamicsHERM(config)

out = model(plan_traj, controller_emb=None)
exec_hat = out.exec_traj
```

Expected shapes:

- `plan_traj`: `[B, T, 3]` or `[B, K, T, 3]`
- `controller_emb`: optional `[B, D]` or `[B, K, D]`
- `exec_hat`: same shape as `plan_traj`
- `residual`: `[B, T, 3]` or `[B, K, T, 3]`
- `params`: `[B, T-1, 9]` or `[B, K, T-1, 9]`

Flattening `[B,K,T,3] -> [B*K,T,3]` should happen inside `inference.py` and model helpers so future `WoTE_model.py` integration stays simple.

## Task 1: Geometry Utilities

**Files:**

- Create: `navsim/agents/WoTE/HERM/__init__.py`
- Create: `navsim/agents/WoTE/HERM/geometry.py`
- Create: `tests/test_herm_geometry.py`

- [ ] Step 1: Add tests for angle wrapping and arc length.

Required behavior:

```python
import math
import torch

from navsim.agents.WoTE.HERM.geometry import wrap_angle, cumulative_arc_length


def test_wrap_angle_bounds():
    angles = torch.tensor([-3 * math.pi, -math.pi, 0.0, math.pi, 3 * math.pi])
    wrapped = wrap_angle(angles)
    assert torch.all(wrapped >= -math.pi)
    assert torch.all(wrapped < math.pi)


def test_cumulative_arc_length_straight_line():
    traj = torch.tensor([[[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [3.0, 0.0, 0.0]]])
    s = cumulative_arc_length(traj)
    assert torch.allclose(s, torch.tensor([[0.0, 1.0, 3.0]]), atol=1e-5)
```

Run:

```bash
pytest tests/test_herm_geometry.py -q
```

Expected:

```text
FAIL because geometry utilities do not exist yet.
```

- [ ] Step 2: Implement basic geometry utilities.

Implementation requirements:

- Pure PyTorch.
- Preserve input device and dtype.
- Avoid Python loops over batch dimension.
- Clamp divisions with `eps=1e-6`.
- `wrap_angle` returns values in `[-pi, pi)`.
- `cumulative_arc_length` uses XY segment length.

- [ ] Step 3: Add tests for trajectory intrinsics.

Required behavior:

```python
from navsim.agents.WoTE.HERM.geometry import trajectory_intrinsics


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
```

- [ ] Step 4: Implement `trajectory_intrinsics`.

Implementation notes:

- `v = ds / dt`, padded to length `T`.
- `a = dv / dt`, padded to length `T`.
- `kappa = dtheta / ds`, padded to length `T`.
- `kappa_dot = dkappa / dt`, padded to length `T`.
- `omega = v * kappa`.

- [ ] Step 5: Add projection and reconstruction round-trip tests.

Required behavior:

```python
from navsim.agents.WoTE.HERM.geometry import project_to_plan_frenet, frenet_to_cartesian


def test_frenet_round_trip_for_identical_trajectory():
    plan = torch.zeros((1, 8, 3), dtype=torch.float32)
    plan[0, :, 0] = torch.arange(8, dtype=torch.float32)
    exec_traj = plan.clone()

    residual = project_to_plan_frenet(plan, exec_traj)
    recon = frenet_to_cartesian(plan, residual["s_exec"], residual["d_exec"], residual["theta_exec"])

    assert torch.allclose(residual["delta_s"], torch.zeros((1, 8)), atol=1e-5)
    assert torch.allclose(residual["delta_d"], torch.zeros((1, 8)), atol=1e-5)
    assert torch.allclose(residual["delta_theta"], torch.zeros((1, 8)), atol=1e-5)
    assert torch.allclose(recon, exec_traj, atol=1e-5)
```

- [ ] Step 6: Implement approximate Frenet projection.

Implementation notes:

- For the first version, project each executed point to the nearest planned point index, not continuous segment projection.
- Compute `delta_s = s_exec_nearest - s_plan_t`.
- Compute signed lateral offset using planned normal vector `n = [-sin(theta), cos(theta)]`.
- Compute `delta_theta = wrap_angle(theta_exec - theta_plan_nearest)`.
- This approximation is stable enough for current `T=8` controller banks.
- Keep continuous segment projection as a future improvement.

Commit:

```bash
git add navsim/agents/WoTE/HERM/__init__.py navsim/agents/WoTE/HERM/geometry.py tests/test_herm_geometry.py
git commit -m "feat: add HERM trajectory geometry utilities"
```

## Task 2: HERM Model

**Files:**

- Create: `navsim/agents/WoTE/HERM/model.py`
- Modify: `navsim/agents/WoTE/HERM/__init__.py`
- Create: `tests/test_herm_model.py`

- [ ] Step 1: Add model shape tests.

Required behavior:

```python
import torch

from navsim.agents.WoTE.HERM import HERMConfig, FrenetErrorDynamicsHERM


def test_herm_forward_shape_without_controller_embedding():
    model = FrenetErrorDynamicsHERM(HERMConfig(num_poses=8, dt=0.5, hidden_dim=64))
    plan = torch.zeros((4, 8, 3))
    plan[:, :, 0] = torch.arange(8, dtype=torch.float32)

    out = model(plan, return_debug=True)

    assert out.exec_traj.shape == (4, 8, 3)
    assert out.residual.shape == (4, 8, 3)
    assert out.params.shape == (4, 7, 9)
```

- [ ] Step 2: Implement config/output dataclasses.

Required fields:

```python
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
```

- [ ] Step 3: Implement `FrenetErrorDynamicsHERM`.

Implementation requirements:

- Accept `[B,T,3]` first.
- Extract `[v, a, kappa, kappa_dot, omega]`.
- Concatenate controller embedding if configured.
- Use a small temporal encoder:
  - linear input projection
  - GRU or Transformer encoder
  - linear parameter head
- Bound parameters:
  - `alpha = max_alpha * tanh(raw_alpha)`
  - `beta = max_beta * tanh(raw_beta)`
  - `b_s = max_bias_s * tanh(raw_b_s)`
  - `b_d = max_bias_d * tanh(raw_b_d)`
  - `b_theta = max_bias_theta * tanh(raw_b_theta)`
- Roll out `T` residual states from `e0=0`.
- Convert to Cartesian with `frenet_to_cartesian`.

- [ ] Step 4: Add identity-initialization behavior test.

Required behavior:

```python
def test_herm_zero_head_is_identity_execution():
    model = FrenetErrorDynamicsHERM(HERMConfig(num_poses=8, dt=0.5, hidden_dim=64))
    torch.nn.init.zeros_(model.param_head.weight)
    torch.nn.init.zeros_(model.param_head.bias)

    plan = torch.zeros((2, 8, 3))
    plan[:, :, 0] = torch.arange(8, dtype=torch.float32)

    out = model(plan)
    assert torch.allclose(out.exec_traj, plan, atol=1e-5)
```

- [ ] Step 5: Add `[B,K,T,3]` helper in `inference.py` later, not directly in `model.py`.

Reason:

- Keeps core model simple.
- Future WoTE integration can flatten candidates once and reshape back.

Commit:

```bash
git add navsim/agents/WoTE/HERM/model.py navsim/agents/WoTE/HERM/__init__.py tests/test_herm_model.py
git commit -m "feat: add Frenet error-dynamics HERM model"
```

## Task 3: Dataset Loader

**Files:**

- Create: `navsim/agents/WoTE/HERM/data.py`
- Create: `tests/test_herm_data.py`

- [ ] Step 1: Add tests for `.npy` pair loading.

Required behavior:

```python
import numpy as np

from navsim.agents.WoTE.HERM.data import load_controller_pairs


def test_load_controller_pairs_from_npy(tmp_path):
    ref = np.zeros((3, 8, 3), dtype=np.float32)
    exe = np.ones((3, 8, 3), dtype=np.float32)
    ref_path = tmp_path / "ref.npy"
    exec_path = tmp_path / "exec.npy"
    np.save(ref_path, ref)
    np.save(exec_path, exe)

    pairs = load_controller_pairs(str(ref_path), str(exec_path))

    assert pairs["plan_traj"].shape == (3, 8, 3)
    assert pairs["exec_traj"].shape == (3, 8, 3)
    assert pairs["style_idx"].shape == (3,)
```

- [ ] Step 2: Add tests for `.npz` style bundle loading.

Required behavior:

```python
def test_load_controller_pairs_from_npz_bundle(tmp_path):
    ref = np.zeros((4, 8, 3), dtype=np.float32)
    exe = np.ones((2, 4, 8, 3), dtype=np.float32)
    bundle_path = tmp_path / "bundle.npz"
    np.savez(bundle_path, ref_traj=ref, exec_trajs=exe, style_names=np.array(["a", "b"]))

    pairs = load_controller_pairs(None, str(bundle_path))

    assert pairs["plan_traj"].shape == (8, 8, 3)
    assert pairs["exec_traj"].shape == (8, 8, 3)
    assert pairs["style_idx"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
```

- [ ] Step 3: Implement `load_controller_pairs`.

Implementation requirements:

- Validate shape equality after flattening styles.
- Cast to `float32`.
- Return a dictionary of NumPy arrays:
  - `plan_traj`
  - `exec_traj`
  - `style_idx`
  - optional `style_name`
- For `.npy` pairs, `style_idx` is all zeros.

- [ ] Step 4: Implement `ControllerTrajectoryDataset`.

Implementation requirements:

- Constructor accepts:
  - `ref_path`
  - `exec_path`
  - `normalize=False`
  - optional `style_indices`
- `__getitem__` returns PyTorch tensors:
  - `plan_traj`
  - `exec_traj`
  - `style_idx`
- No scene/BEV dependency.

Commit:

```bash
git add navsim/agents/WoTE/HERM/data.py tests/test_herm_data.py
git commit -m "feat: add HERM controller trajectory dataset"
```

## Task 4: Losses and Metrics

**Files:**

- Create: `navsim/agents/WoTE/HERM/losses.py`
- Modify: `tests/test_herm_model.py`

- [ ] Step 1: Add loss tests.

Required behavior:

```python
import torch

from navsim.agents.WoTE.HERM.losses import herm_loss


def test_herm_loss_zero_when_prediction_matches_target():
    pred = torch.zeros((2, 8, 3))
    target = torch.zeros((2, 8, 3))
    residual = torch.zeros((2, 8, 3))
    params = torch.zeros((2, 7, 9))

    loss, metrics = herm_loss(pred, target, residual, params)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    assert metrics["pos_l1"].item() == 0.0
```

- [ ] Step 2: Implement `herm_loss`.

Loss terms:

```text
L = w_xy * L1(xy_hat, xy_exec)
  + w_yaw * L1(wrap(theta_hat - theta_exec))
  + w_residual_smooth * L1(diff(residual))
  + w_param_smooth * L1(diff(params))
```

Default weights:

- `w_xy = 1.0`
- `w_yaw = 0.2`
- `w_residual_smooth = 0.01`
- `w_param_smooth = 0.001`

Return:

```python
loss: torch.Tensor
metrics: Dict[str, torch.Tensor]
```

Commit:

```bash
git add navsim/agents/WoTE/HERM/losses.py tests/test_herm_model.py
git commit -m "feat: add HERM training loss"
```

## Task 5: Training CLI

**Files:**

- Create: `navsim/agents/WoTE/HERM/train.py`
- Create: `navsim/agents/WoTE/HERM/evaluate.py`
- Create: `navsim/agents/WoTE/HERM/inference.py`

- [ ] Step 1: Implement checkpoint schema in `inference.py`.

Checkpoint fields:

```python
{
    "model_state": model.state_dict(),
    "config": asdict(config),
    "epoch": epoch,
    "metrics": metrics,
    "ref_path": ref_path,
    "exec_path": exec_path,
}
```

Public functions:

```python
def load_herm_checkpoint(path: str, device: torch.device | str = "cpu") -> FrenetErrorDynamicsHERM:
    ...

def execute_with_herm(model, planned_trajs, controller_emb=None):
    ...
```

`execute_with_herm` requirements:

- Supports `[B,T,3]`.
- Supports `[B,K,T,3]` by flattening to `[B*K,T,3]`.
- Preserves dtype and device.
- Returns same shape as input.

- [ ] Step 2: Implement `train.py`.

CLI arguments:

```bash
python -m navsim.agents.WoTE.HERM.train \
  --ref-path /path/to/ref.npy \
  --exec-path /path/to/exec.npy \
  --output /path/to/herm.pt \
  --epochs 200 \
  --batch-size 256 \
  --lr 1e-3 \
  --dt 0.5 \
  --hidden-dim 256 \
  --val-ratio 0.1
```

Training behavior:

- Load controller pairs.
- Random split by sample unless bundle style split is requested later.
- Train HERM with AdamW.
- Save best checkpoint by validation position RMSE.
- Print per-epoch train/val metrics.

- [ ] Step 3: Implement `evaluate.py`.

CLI arguments:

```bash
python -m navsim.agents.WoTE.HERM.evaluate \
  --checkpoint /path/to/herm.pt \
  --ref-path /path/to/ref.npy \
  --exec-path /path/to/exec.npy
```

Metrics:

- mean XY L1
- mean XY RMSE
- final displacement error
- mean yaw error
- final yaw error

Commit:

```bash
git add navsim/agents/WoTE/HERM/train.py navsim/agents/WoTE/HERM/evaluate.py navsim/agents/WoTE/HERM/inference.py
git commit -m "feat: add HERM training and inference CLIs"
```

## Task 6: Verification on Synthetic Data

**Files:**

- Modify: `tests/test_herm_model.py`

- [ ] Step 1: Add synthetic lateral-bias learning test.

Test idea:

- Generate straight reference trajectories.
- Executed trajectories are shifted by fixed `y=0.5`.
- Train for a small number of iterations inside the test.
- Assert final loss is lower than initial loss.

Required behavior:

```python
def test_herm_can_learn_simple_lateral_bias():
    torch.manual_seed(0)
    model = FrenetErrorDynamicsHERM(HERMConfig(num_poses=8, dt=0.5, hidden_dim=64))
    plan = torch.zeros((32, 8, 3))
    plan[:, :, 0] = torch.arange(8, dtype=torch.float32)
    target = plan.clone()
    target[:, :, 1] = 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    with torch.no_grad():
        initial = torch.mean(torch.abs(model(plan).exec_traj[..., :2] - target[..., :2])).item()

    for _ in range(80):
        out = model(plan)
        loss = torch.mean(torch.abs(out.exec_traj[..., :2] - target[..., :2]))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = torch.mean(torch.abs(model(plan).exec_traj[..., :2] - target[..., :2])).item()

    assert final < initial * 0.5
```

- [ ] Step 2: Run focused tests.

Run:

```bash
pytest tests/test_herm_geometry.py tests/test_herm_model.py tests/test_herm_data.py -q
```

Expected:

```text
all HERM tests pass
```

Commit:

```bash
git add tests/test_herm_model.py
git commit -m "test: verify HERM learns simple execution residuals"
```

## Task 7: Future WoTE Integration Plan

This task should be done only after standalone HERM is approved and trained.

**Files later likely touched:**

- Modify: `navsim/agents/WoTE/WoTE_model.py`
- Modify: `navsim/agents/WoTE/configs/default.py`

Planned config additions:

```python
herm_enabled: bool = False
herm_checkpoint_path: str = ""
herm_apply_in_train: bool = False
herm_apply_in_eval: bool = True
herm_trainable: bool = False
herm_residual_weight: float = 1.0
herm_use_controller_emb: bool = False
```

No legacy config aliases should be added. HERM should be controlled only by `herm_*` names.

Planned model integration point:

1. Build candidate planned trajectories:

```python
planned = init_trajectory_anchor + trajectory_offset
```

2. Execute them with HERM:

```python
exec_hat = execute_with_herm(self.herm_model, planned, controller_emb=optional_style_emb)
```

3. Use `exec_hat` as the trajectory used for:

- reward feature encoding,
- future ego injection into BEV,
- optional output key `trajectory_offset_exec_trajs`.

Important constraint:

- Do not replace the planner's selected output silently until evaluation behavior is checked.
- First expose both:
  - `all_trajectory`: planned candidates
  - `all_trajectory_exec_hat`: HERM-predicted executed candidates

## Risks and Design Decisions

- Nearest-point Frenet projection is an approximation.
  - It is acceptable for short `T=8` anchor banks.
  - If training residuals look noisy on curved anchors, upgrade to segment projection.

- `Delta s` target can be unstable when the executed trajectory moves far from the planned trajectory.
  - Mitigation: train primarily with Cartesian loss and use Frenet residual loss only as an auxiliary term if needed.

- Style conditioning is optional in the first version.
  - Since the user plans to train with `ref_exec` and `exec_traj` of the same style, a per-style HERM checkpoint can work without controller embedding.
  - Multi-style bundle support remains available through `style_idx`; style embedding can be added after baseline works.

- HERM should not depend on WoTE scene features.
  - It must remain trainable from controller trajectory pairs alone.

## Acceptance Criteria

- `pytest tests/test_herm_geometry.py tests/test_herm_model.py tests/test_herm_data.py -q` passes.
- `python -m navsim.agents.WoTE.HERM.train ...` can train on a `.npy` pair.
- `python -m navsim.agents.WoTE.HERM.train ...` can train on a `.npz` style bundle.
- `execute_with_herm` accepts `[B,K,T,3]` and returns `[B,K,T,3]`.
- With zero model head, HERM returns identity execution.
- On a simple synthetic lateral-bias dataset, HERM reduces prediction loss by at least 50%.

## Confirmation Needed Before Code

Please confirm these two design choices before implementation:

1. First HERM version trains one execution style per checkpoint by default; multi-style `.npz` is supported as data input, but no learned `style_idx` embedding is required unless you ask for it.
2. First Frenet projection uses nearest planned waypoint rather than continuous segment projection; if results are noisy, we upgrade projection in the next iteration.
