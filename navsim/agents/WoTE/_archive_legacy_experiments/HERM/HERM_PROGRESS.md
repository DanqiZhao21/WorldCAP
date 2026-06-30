# HERM Progress

Date: 2026-05-14

This document records the current HERM work under `navsim/agents/WoTE/HERM/` and the current conclusion before integrating it into WoTE.

## Goal

HERM is intended to predict how a controller will execute a planned trajectory.

The current intended inference form is support-conditioned:

```text
support ref trajectories + support exec trajectories + new plan trajectory
    -> predicted exec trajectory for the new plan
```

This matches the actual controller-style setting: the model should infer the current controller style from known `(ref, exec)` pairs, then apply that style to newly planned candidate trajectories.

## Files

Current HERM files:

- `geometry.py`: Frenet/cartesian geometry utilities.
- `data.py`: plain and support-conditioned datasets.
- `model.py`: HERM dynamics model and support-conditioned model.
- `losses.py`: HERM training losses.
- `train.py`: plain `plan -> exec` training entry.
- `train_support.py`: support-conditioned training entry.
- `evaluate.py`: checkpoint evaluation helper.
- `inference.py`: plain inference helper.
- `inference_support.py`: support-conditioned inference helper.
- `__init__.py`: package exports.

Current tests:

- `tests/test_herm_geometry.py`
- `tests/test_herm_data.py`
- `tests/test_herm_model.py`
- `tests/test_herm_train.py`
- `tests/test_herm_train_support.py`
- `tests/test_herm_inference_support.py`

Latest verification:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE pytest \
  tests/test_herm_geometry.py \
  tests/test_herm_data.py \
  tests/test_herm_model.py \
  tests/test_herm_train.py \
  tests/test_herm_train_support.py \
  tests/test_herm_inference_support.py \
  -q
```

Result:

```text
26 passed in 3.67s
```

## Geometry

`project_to_plan_frenet` now uses same-timestep local Frenet error instead of nearest-waypoint matching.

For each timestep:

```text
delta_xy = exec_xy - plan_xy
delta_s = dot(delta_xy, plan_tangent)
delta_d = dot(delta_xy, plan_normal)
delta_theta = wrap(exec_yaw - plan_yaw)
s_exec = s_plan + delta_s
```

This makes the meaning clearer for the current problem: at the same time index, measure how the executed trajectory differs from the planned trajectory. It avoids the previous nearest-waypoint ambiguity where a point between waypoints could be assigned to the wrong discrete anchor point.

`frenet_to_cartesian` converts predicted Frenet state back into world-frame `(x, y, yaw)`. This is where HERM's residual prediction becomes an executable trajectory.

## Model

The main support-conditioned model is `SupportConditionalHERM`.

Inputs:

```text
support_plan: [B, S, T, 3]
support_exec: [B, S, T, 3]
query_plan:   [B, Q, T, 3]
```

Outputs:

```text
pred_exec: [B, Q, T, 3]
residual:  [B, Q, T, 3]
params:    [B, Q, T, 9]
```

The model has two parts:

1. `SupportStyleEncoder`
   - Computes same-timestep Frenet residuals for every support pair.
   - Builds per-timestep features:

```text
[v, a, kappa, kappa_dot, omega, delta_s, delta_d, delta_theta]
```

   - Encodes every support pair with 1D Conv over time.
   - Aggregates all support pair embeddings with learned attention instead of simple mean pooling.

2. `FrenetErrorDynamicsHERM`
   - Computes query trajectory intrinsic features:

```text
[v, a, kappa, kappa_dot, omega]
```

   - Concatenates intrinsic features with the support style embedding.
   - Predicts bounded residual dynamics parameters.
   - Rolls out Frenet residual states and converts them to cartesian predicted execution.

The `controller_emb_dim` field in `HERMConfig` is still meaningful. It is the dimensionality of the style/controller embedding passed into the dynamics model. It is not just leftover naming.

## Dataset

The support-conditioned dataset is `SupportQueryTrajectoryDataset`.

It treats each controller style as one task:

```text
same style:
  support trajectories -> infer style
  query trajectories   -> predict execution under that style
```

The dataset supports reproducible epoch-wise resampling:

```text
epoch seed = base_seed + epoch
```

So support/query split can change every epoch while still being reproducible from the same base seed.

## Loss

Main loss terms:

```text
pos_l1 = L1(pred_exec[..., :2], target_exec[..., :2])
yaw_l1 = abs(wrap(pred_yaw - target_yaw))
residual_smooth = L1(residual[:, :, 1:] - residual[:, :, :-1])
param_smooth = L1(params[:, :, 1:] - params[:, :, :-1])
```

Base loss:

```text
loss = 1.0 * pos_l1
     + 0.2 * yaw_l1
     + 0.01 * residual_smooth
     + 0.001 * param_smooth
```

An optional residual supervision term was tested:

```text
residual_l1 = L1(predicted_frenet_residual, target_frenet_residual)
loss += w_residual * residual_l1
```

In the current experiments, residual supervision did not improve cartesian position accuracy.

## Training Results

Wandb project:

```text
https://wandb.ai/2564380679-/WoTE-HERM
```

| Run | Bank | Support/Query | Main change | Best val pos L1 | Notes |
| --- | ---: | ---: | --- | ---: | --- |
| `herm-support-256-bank-style-split-20260514` | 256 | 192/64 | support-conditioned baseline | about 0.878 m | Too little support/query coverage. |
| `herm_support_1024_768_256_wandb_20260514.pt` | 1024 | 768/256 | larger bank, mean support pooling | 0.394619 m | Big improvement from more data. |
| `herm_support_1024_768_256_residual_wandb_20260514.pt` | 1024 | 768/256 | residual supervision, `w_residual=0.2` | 0.395530 m | Position did not improve; yaw slightly improved. |
| `herm_support_1024_768_256_convattn_wandb_20260514.pt` | 1024 | 768/256 | 1D Conv support encoder + attention support pooling | 0.373605 m | Current best. |

Current best checkpoint:

```text
trainingResult/HERM/herm_support_1024_768_256_convattn_wandb_20260514.pt
```

Current best wandb run:

```text
https://wandb.ai/2564380679-/WoTE-HERM/runs/ztajh8jg
```

Best epoch:

```text
epoch 102
val_loss = 0.381070
val_pos_l1 = 0.373605
val_yaw_l1 = 0.019017
residual_smooth = 0.355679
param_smooth = 0.104060
```

Compared with the 1024 mean-pooling baseline:

```text
0.394619 m -> 0.373605 m
absolute gain: about 0.021 m
relative gain: about 5.3%
```

## Reproduction Command

Suggested 200-epoch command:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE \
python -m navsim.agents.WoTE.HERM.train_support \
  --exec-path /home/zhaodanqi/clone/WoTE/ControllerExp/generated/1024/controller_styles_1024.npz \
  --output /home/zhaodanqi/clone/WoTE/trainingResult/HERM/herm_support_1024_768_256_convattn_200ep.pt \
  --epochs 200 \
  --batch-size 4 \
  --support-size 768 \
  --query-size 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dt 0.5 \
  --num-poses 8 \
  --hidden-dim 256 \
  --num-layers 2 \
  --dropout 0.1 \
  --style-emb-dim 64 \
  --style-hidden-dim 128 \
  --seed 0 \
  --device cuda:0 \
  --wandb \
  --wandb-project WoTE-HERM \
  --wandb-run-name herm-support-1024-s768-q256-convattn-200ep
```

## Current Interpretation

The 1024 bank clearly helps. The main reason is that support contains more examples of the same controller style, and query covers more trajectory shapes. This lets the style encoder estimate controller behavior more reliably.

The remaining position error is still not tiny. Likely reasons:

- HERM only sees trajectory geometry and support pairs; it does not see scene context.
- The current residual dynamics form is intentionally simple.
- Some controller errors may be high-frequency or discontinuous.
- Support/query sampling is still a proxy for the final WoTE usage.
- WoTE integration will create a different input distribution: candidate trajectories are model-generated, not exactly the same as controller-bank anchors.

## Next Engineering Step

The next step is to connect the best support-conditioned HERM checkpoint into WoTE:

```text
WoTE candidate plan trajectories
  + active controller support bank ref/exec pairs
  -> HERM predicted exec candidate trajectories
  -> reward/world-model ranking
```

The important design choice is that HERM should not replace the planner's candidate generation. It should sit between generated plan candidates and the scoring/world-model path, so the reward model ranks trajectories closer to what the controller will actually execute.
