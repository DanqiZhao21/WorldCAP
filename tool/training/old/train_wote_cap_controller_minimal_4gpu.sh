#!/bin/bash
set -euo pipefail

# WoTE training (minimal controller-aware CAP) - 4 GPUs (0,1,2,3)
# Keeps ONLY controller embedding + injection into:
#   (A) traj feature (CAP)
#   (B) offset branch
#   (C) reward_feature (scoring)
# Notes:
# - If CTRL_EXEC is a .npz bundle with multiple styles, training samples a random style per forward.
# - im_loss_weight / metric_loss_weight are effective (scaled in WoTE_loss.py).

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

CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC=${CTRL_EXEC:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}

NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}"}
export NAVSIM_EXP_ROOT

CACHE_PATH=${CACHE_PATH:-"${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114"}
EXP_BASE=${EXP_BASE:-"WoTE/cap_controller_min"}
EXP_TAG=${EXP_TAG:-"ctrl_cap_min_4gpu"}

# -----------------------------
# Training hyperparams
# -----------------------------
BATCH_SIZE=${WOTE_BATCH_SIZE:-16}
MAX_EPOCHS=${WOTE_MAX_EPOCHS:-40}
LR=${WOTE_LR:-"1e-4"}
MIN_LR=${WOTE_MIN_LR:-"1e-6"}
WARMUP_EPOCHS=${WOTE_WARMUP_EPOCHS:-3}

# DDP uses all visible GPUs (0,1,2,3)
DEVICES=${DEVICES:-4}

# -----------------------------
# Controller injection knobs
# -----------------------------
CTRL_FEATURE_MODE=${CTRL_FEATURE_MODE:-full}            # full | lateral_only
INJECT_MODE=${INJECT_MODE:-film}                       # film | add | none
INJECT_STRENGTH=${INJECT_STRENGTH:-0.25}               # global gate (0~1)

COND_TRAJ=${COND_TRAJ:-true}
TRAJ_STRENGTH=${TRAJ_STRENGTH:-0.25}

COND_OFFSET=${COND_OFFSET:-true}
OFFSET_STRENGTH=${OFFSET_STRENGTH:-0.25}

COND_REWARD=${COND_REWARD:-true}
REWARD_STRENGTH=${REWARD_STRENGTH:-0.35}

# Loss weights
IM_LOSS_W=${IM_LOSS_W:-1.0}
METRIC_LOSS_W=${METRIC_LOSS_W:-1.0}

echo "============================================================"
echo "[TRAIN] WoTE minimal CAP (controller-aware traj/offset/reward)"
echo "  anchors      = ${PLANNER_ANCHORS}"
echo "  ctrl_ref      = ${CTRL_REF}"
echo "  ctrl_exec     = ${CTRL_EXEC}"
echo "  cache_path    = ${CACHE_PATH}"
echo "  exp           = ${EXP_BASE}/${EXP_TAG}"
echo "  devices       = ${DEVICES} (visible: ${CUDA_VISIBLE_DEVICES})"
echo "  inject        = mode=${INJECT_MODE} strength=${INJECT_STRENGTH}"
echo "  traj/offset/reward = ${COND_TRAJ}/${COND_OFFSET}/${COND_REWARD}"
echo "  im_w/metric_w = ${IM_LOSS_W}/${METRIC_LOSS_W}"
echo "============================================================"

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
  ++agent.config.controller_injection_mode=${INJECT_MODE} \
  ++agent.config.controller_injection_strength=${INJECT_STRENGTH} \
  ++agent.config.controller_condition_on_traj_feature=${COND_TRAJ} \
  ++agent.config.controller_traj_condition_strength=${TRAJ_STRENGTH} \
  ++agent.config.controller_condition_on_offset=${COND_OFFSET} \
  ++agent.config.controller_offset_condition_strength=${OFFSET_STRENGTH} \
  ++agent.config.controller_condition_on_reward_feature=${COND_REWARD} \
  ++agent.config.controller_reward_condition_strength=${REWARD_STRENGTH} \
  ++agent.config.im_loss_weight=${IM_LOSS_W} \
  ++agent.config.metric_loss_weight=${METRIC_LOSS_W}

echo "[DONE] Training finished: ${EXP_BASE}/${EXP_TAG}"

# Optional:
#   export WOTE_CTRL_STYLE_IDX=0
#   export WOTE_CTRL_STYLE_DEBUG=1
