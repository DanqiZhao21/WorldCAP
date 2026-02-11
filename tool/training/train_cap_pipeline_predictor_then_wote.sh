#!/bin/bash
set -euo pipefail

# One-click pipeline:
#   1) Train Controller Response Predictor from generated controller style data
#   2) Train WoTE (planner) using your existing controller global-style/CAP setup
#
# Why separate?
# - The response predictor is a plug-in learned only from (ref, exec) controller-bank data.
# - WoTE training uses scene/BEV targets; predictor does not need them and is typically kept frozen.

ROOT=/home/zhaodanqi/clone/WoTE

# Prefer the currently active environment's python.
# You can override by exporting PYTHON=/abs/path/to/python
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
"${PYTHON}" -c "import sys; print('[INFO] python:', sys.executable); print('[INFO] version:', sys.version.replace('\n',' '))" || true

# GPUs (both stages inherit this)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# -----------------------------
# (1) Train response predictor
# -----------------------------
BUNDLE_IN=${BUNDLE_IN:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}
DEBUG_DEFAULT_EXEC=${DEBUG_DEFAULT_EXEC:-"${ROOT}/ControllerExp/generated/debug_default_none.npy"}
CTRL_RP_OUT=${CTRL_RP_OUT:-"${ROOT}/ControllerExp/generated/controller_response_predictor.pt"}

PRED_DEVICE=${PRED_DEVICE:-"cuda"}
PRED_EPOCHS=${PRED_EPOCHS:-30}
PRED_STEPS_PER_EPOCH=${PRED_STEPS_PER_EPOCH:-2000}
PRED_BATCH_SIZE=${PRED_BATCH_SIZE:-16}
PRED_PHI_BANK=${PRED_PHI_BANK:-32}
PRED_LR=${PRED_LR:-1e-3}
PRED_SEED=${PRED_SEED:-0}
PRED_FEATURE_MODE=${PRED_FEATURE_MODE:-full}   # full | lateral_only

# Quick runtime estimate (very rough):
#   total_steps = PRED_EPOCHS * PRED_STEPS_PER_EPOCH (default 60k)
# On a typical GPU this is often ~20-90 minutes depending on phi_bank, GPU, and ControllerEmbedding cost.
# For a smoke test, set: PRED_EPOCHS=2 PRED_STEPS_PER_EPOCH=200

echo "============================================================"
echo "[1/2] Train Controller Response Predictor"
echo "      bundle=${BUNDLE_IN}"
echo "      out=${CTRL_RP_OUT}"
echo "============================================================"

BUNDLE_IN="${BUNDLE_IN}" \
DEBUG_DEFAULT_EXEC="${DEBUG_DEFAULT_EXEC}" \
OUT_CKPT="${CTRL_RP_OUT}" \
DEVICE="${PRED_DEVICE}" \
EPOCHS="${PRED_EPOCHS}" \
STEPS_PER_EPOCH="${PRED_STEPS_PER_EPOCH}" \
BATCH_SIZE="${PRED_BATCH_SIZE}" \
PHI_BANK="${PRED_PHI_BANK}" \
LR="${PRED_LR}" \
SEED="${PRED_SEED}" \
FEATURE_MODE="${PRED_FEATURE_MODE}" \
PYTHON="${PYTHON}" \
  bash "${ROOT}/tool/controller_response/train_controller_response_predictor_generated.sh"

# -----------------------------
# (2) Train WoTE
# -----------------------------
export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"

# Planner candidate anchors (THIS defines the planning trajectory set)
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}

# Controller bank for style embedding / CAP conditioning during WoTE training
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC=${CTRL_EXEC:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}

# Cache + exp
CACHE_PATH=${CACHE_PATH:-"${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114"}
EXP_BASE=${EXP_BASE:-"WoTE/cap_pipeline"}

# WoTE hyperparams
BATCH_SIZE=${WOTE_BATCH_SIZE:-16}
MAX_EPOCHS=${WOTE_MAX_EPOCHS:-20}
LR=${WOTE_LR:-"1e-4"}
MIN_LR=${WOTE_MIN_LR:-"1e-5"}
WARMUP_EPOCHS=${WOTE_WARMUP_EPOCHS:-5}

# Choose one training mode:
#   MODE=cap_trajfeat  (recommended for CAP)
#   MODE=global_style  (your previous BEV-token conditioning)
MODE=${MODE:-cap_trajfeat}

echo "============================================================"
echo "[2/2] Train WoTE (${MODE})"
echo "      anchors=${PLANNER_ANCHORS}"
echo "      ctrl_exec=${CTRL_EXEC}"
echo "============================================================"

EXTRA_OVERRIDES=()

if [[ "${MODE}" == "cap_trajfeat" ]]; then
  # CAP: controller affects per-trajectory feature branch; disable BEV-token injection.
  EXTRA_OVERRIDES+=(
    ++agent.config.controller_condition_scope=global
    ++agent.config.controller_style_pooling=mean
    ++agent.config.controller_condition_on_traj_feature=true
    ++agent.config.controller_traj_condition_strength=0.1
    ++agent.config.controller_condition_on_bev_tokens=false
    agent.config.controller_injection_mode=film
    agent.config.controller_injection_strength=0.1
    agent.config.controller_inject_every_step=false
  )
elif [[ "${MODE}" == "global_style" ]]; then
  EXTRA_OVERRIDES+=(
    ++agent.config.controller_condition_scope=global
    ++agent.config.controller_style_pooling=mean
    ++agent.config.controller_condition_on_traj_feature=false
    ++agent.config.controller_condition_on_bev_tokens=true
    agent.config.controller_injection_mode=film
    agent.config.controller_injection_strength=0.1
    agent.config.controller_inject_every_step=false
  )
else
  echo "[ERR] Unknown MODE=${MODE} (cap_trajfeat|global_style)" >&2
  exit 1
fi

# Note: The response predictor ckpt is typically used at *evaluation* time to penalize rewards.
# WoTE training does not need it unless you later decide to add risk loss into forward_train.

"${PYTHON}" ./navsim/planning/script/run_training.py \
  agent=WoTE_agent \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  ++agent.config.cluster_file_path="${PLANNER_ANCHORS}" \
  agent.config.controller_ref_traj_path="${CTRL_REF}" \
  agent.config.controller_exec_traj_path="${CTRL_EXEC}" \
  use_cache_without_dataset=true \
  cache_path="${CACHE_PATH}" \
  dataloader.params.batch_size=${BATCH_SIZE} \
  trainer.params.max_epochs=${MAX_EPOCHS} \
  split=trainval \
  experiment_name="${EXP_BASE}/${MODE}" \
  scene_filter=navtrain \
  agent.lr=${LR} \
  agent.config.min_lr=${MIN_LR} \
  agent.config.warmup_epochs=${WARMUP_EPOCHS} \
  "${EXTRA_OVERRIDES[@]}"

echo "[DONE] Pipeline finished."
echo "- Predictor ckpt: ${CTRL_RP_OUT}"
echo "- WoTE exp: ${EXP_BASE}/${MODE}"

echo "\nNext (evaluation with CAP risk penalty):"
echo "  export WOTE_USE_CTRL_RISK=1"
echo "  export WOTE_CTRL_RP_CKPT=\"${CTRL_RP_OUT}\""
echo "  export WOTE_CTRL_RISK_W=0.2"
echo "  bash ${ROOT}/tool/evaluate/evaluate_all_0202.sh"
