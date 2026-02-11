#!/bin/bash
set -euo pipefail

# Minimal WoTE training script (Test-time Controller Adaptation, simplified):
# - Trains ONLY a controller-conditioned BEV/world-model transition + reward heads.
# - Controller affects *latent transition* (future BEV evolution), not trajectory generation/refine.
# - Supervision for controller styles comes from controller_styles.npz (sample one style per forward).
# - No response predictor / risk penalty / pre-compensation.
#
# NOTE (2026-02): Freezing policy is controlled by env var WOTE_TRAIN_PROFILE in
#   navsim/planning/training/agent_lightning_module.py
# Training profile is controlled by env var WOTE_TRAIN_PROFILE in
#   navsim/planning/training/agent_lightning_module.py
# Recommended for this setting:
export WOTE_TRAIN_PROFILE=wm_reward_only

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}

# Prefer current env python. Override with: export PYTHON=/abs/path/to/python
if [[ -n "${PYTHON:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=$(command -v python3)
else
  echo "[ERR] Cannot find python. Activate your conda/venv first or export PYTHON=/abs/path/to/python" >&2
  exit 1
fi

echo "[INFO] Using PYTHON=${PYTHON}"
"${PYTHON}" -c "import sys; print('[INFO] python:', sys.executable); print('[INFO] version:', sys.version.replace('\\n',' '))" || true

# GPUs (force to 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Python import paths
export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"

# -----------------------------
# Inputs / paths
# -----------------------------
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}

# Controller bank (ref/exec). Use .npz exec bundle to train for multi-style generalization.
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC=${CTRL_EXEC:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}

# NAVSIM_EXP_ROOT is used by the default hydra training config.
NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}"}
export NAVSIM_EXP_ROOT

CACHE_PATH=${CACHE_PATH:-"${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114"}
EXP_BASE=${EXP_BASE:-"WoTE/cap_controller_min"}
EXP_TAG=${EXP_TAG:-"ctrl_cap_min"}

# -----------------------------
# Training hyperparams
# -----------------------------
BATCH_SIZE=${WOTE_BATCH_SIZE:-16}
MAX_EPOCHS=${WOTE_MAX_EPOCHS:-40}
LR=${WOTE_LR:-"1e-4"}
MIN_LR=${WOTE_MIN_LR:-"1e-6"}
WARMUP_EPOCHS=${WOTE_WARMUP_EPOCHS:-3}

# Ensure DDP uses all visible GPUs (0,1,2,3)
DEVICES=${DEVICES:-4}

# -----------------------------
# Controller -> World-Model conditioning knobs
# -----------------------------
CTRL_FEATURE_MODE=${CTRL_FEATURE_MODE:-full}            # full | lateral_only
CTRL_POOLING=${CTRL_POOLING:-attn}                     # attn | mean
COND_WM=${COND_WM:-true}
WM_STRENGTH=${WM_STRENGTH:-0.30}
WM_INJECT_TARGET=${WM_INJECT_TARGET:-all}              # all | ego

# Loss weights (now effective in WoTE_loss.py)
IM_LOSS_W=${IM_LOSS_W:-1.0}
METRIC_LOSS_W=${METRIC_LOSS_W:-1.0}

echo "============================================================"
echo "[TRAIN] WoTE controller-conditioned WM + reward (TTCA)"
echo "  anchors      = ${PLANNER_ANCHORS}"
echo "  ctrl_ref      = ${CTRL_REF}"
echo "  ctrl_exec     = ${CTRL_EXEC}"
echo "  cache_path    = ${CACHE_PATH}"
echo "  exp           = ${EXP_BASE}/${EXP_TAG}"
echo "  devices       = ${DEVICES} (visible: ${CUDA_VISIBLE_DEVICES})"
echo "  ctrl->wm      = enable=${COND_WM} strength=${WM_STRENGTH} target=${WM_INJECT_TARGET}"
echo "  ctrl_pooling  = ${CTRL_POOLING}"
echo "  im_w/metric_w = ${IM_LOSS_W}/${METRIC_LOSS_W}"
echo "  WOTE_TRAIN_PROFILE=${WOTE_TRAIN_PROFILE:-wm_reward_only}"
echo "============================================================"

# -----------------------------
# Preflight: fail fast on obvious wiring errors
# -----------------------------
"${PYTHON}" - <<PY
from navsim.agents.WoTE.configs.default import WoTEConfig
from navsim.agents.WoTE.WoTE_model import WoTEModel

import os

cfg = WoTEConfig()
cfg.cluster_file_path = r"""${PLANNER_ANCHORS}"""
cfg.controller_ref_traj_path = r"""${CTRL_REF}"""
cfg.controller_exec_traj_path = r"""${CTRL_EXEC}"""
cfg.controller_feature_mode = r"""${CTRL_FEATURE_MODE}"""
cfg.controller_style_pooling = r"""${CTRL_POOLING}"""

m = WoTEModel(cfg)
pooling = str(r"""${CTRL_POOLING}""" or 'attn').lower()

need = ['controller_style_pooling', 'ctrl_proj', 'ctrl_token_ln', 'controller_encoder']
if pooling == 'attn':
  need += ['ctrl_style_attn']

missing = [k for k in need if not hasattr(m, k)]
if missing:
    raise RuntimeError('Preflight failed, missing attrs: ' + ','.join(missing))

# Optional (can be slow): run a tiny forward_train contract check.
if os.environ.get('WOTE_PREFLIGHT_FORWARD', '0') == '1':
  import torch
  pred = m.forward_train(
    {
      'camera_feature': torch.zeros((1, 3, 256, 1024)),
      'lidar_feature': torch.zeros((1, 1, 256, 256)),
      'status_feature': torch.zeros((1, 8)),
    },
    targets=None,
  )
  must_keys = ['trajectory_offset', 'trajectory_offset_rewards']
  missing_pred = [k for k in must_keys if k not in pred]
  if missing_pred:
    raise RuntimeError('Preflight failed, forward_train missing keys: ' + ','.join(missing_pred))

# Check controller bundle path existence (non-fatal but helpful).
ctrl_exec = r"""${CTRL_EXEC}"""
if ctrl_exec and (not os.path.isfile(ctrl_exec)):
  print('[WARN] CTRL_EXEC does not exist on disk:', ctrl_exec)
print('[INFO] Preflight OK')
PY

"${PYTHON}" ./navsim/planning/script/run_training.py \
  agent=WoTE_agent \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  ++agent.config.cluster_file_path="${PLANNER_ANCHORS}" \
  ++agent.config.controller_ref_traj_path="${CTRL_REF}" \
  ++agent.config.controller_exec_traj_path="${CTRL_EXEC}" \
  use_cache_without_dataset=true \
  cache_path="${CACHE_PATH}" \
  dataloader.params.batch_size=${BATCH_SIZE} \
  trainer.params.max_epochs=${MAX_EPOCHS} \
  +trainer.params.devices=${DEVICES} \
  split=trainval \
  experiment_name="${EXP_BASE}/${EXP_TAG}" \
  scene_filter=navtrain \
  agent.lr=${LR} \
  agent.config.min_lr=${MIN_LR} \
  agent.config.warmup_epochs=${WARMUP_EPOCHS} \
  ++agent.config.controller_feature_mode=${CTRL_FEATURE_MODE} \
  ++agent.config.controller_style_pooling=${CTRL_POOLING} \
  ++agent.config.use_agent_loss=false \
  ++agent.config.use_map_loss=true \
  ++agent.config.bev_semantic_weight=0.0 \
  ++agent.config.fut_bev_semantic_weight=1.0 \
  ++agent.config.traj_offset_loss_weight=0.0 \
  ++agent.config.offset_im_reward_weight=0.0 \
  ++agent.config.im_loss_weight=${IM_LOSS_W} \
  ++agent.config.metric_loss_weight=${METRIC_LOSS_W}

echo "[DONE] Training finished: ${EXP_BASE}/${EXP_TAG}"

# Tips:
# - Force a fixed controller style (bundle only):
#     export WOTE_CTRL_STYLE_IDX=0
# - Debug sampled styles (bundle only):
#     export WOTE_CTRL_STYLE_DEBUG=1
