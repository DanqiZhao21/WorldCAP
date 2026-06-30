#!/bin/bash
set -euo pipefail

# One-click CAP training pipeline (2026-02):
#   1) Train Controller Response Predictor from controller-bank (ref, exec)
#   2) Train WoTE with CAP (traj-feature conditioning)
#      and enable response-based trajectory pre-compensation in eval/validation.
#
# Notes:
# - Pre-compensation is applied in WoTE_model.py when model is in eval mode (self.is_eval).
# - Training forward typically does not apply compensation unless the model uses eval path.
# - This script wires the predictor ckpt into WoTE so that validation/evaluation can use it.

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

# GPUs (both stages inherit this)
# Force to 0,1,2,3 as requested.
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ============================================================
# (1) Train response predictor (optional)
# ============================================================
SKIP_PRED_TRAIN=${SKIP_PRED_TRAIN:-0}

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

if [[ "${SKIP_PRED_TRAIN}" == "1" ]]; then
  echo "============================================================"
  echo "[1/2] Skip predictor training (SKIP_PRED_TRAIN=1)"
  echo "      expect ckpt at: ${CTRL_RP_OUT}"
  echo "============================================================"
else
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
fi

# ============================================================
# (2) Train WoTE (CAP + pre-compensation wiring)
# ============================================================
export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"

# Planner candidate anchors (defines planning trajectory set)
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}

# Controller bank for controller embedding during WoTE training/validation
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC=${CTRL_EXEC:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}

# Cache + exp
CACHE_PATH=${CACHE_PATH:-"${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114"}
EXP_BASE=${EXP_BASE:-"WoTE/cap_pipeline_comp"}

# WoTE hyperparams
BATCH_SIZE=${WOTE_BATCH_SIZE:-16}
MAX_EPOCHS=${WOTE_MAX_EPOCHS:-20}
LR=${WOTE_LR:-"1e-4"}
MIN_LR=${WOTE_MIN_LR:-"1e-5"}
WARMUP_EPOCHS=${WOTE_WARMUP_EPOCHS:-5}

# CAP knobs
CAP_STRENGTH=${CAP_STRENGTH:-0.1}
INJECT_MODE=${INJECT_MODE:-film} # film|attn|concat|add

# Response-based trajectory pre-compensation knobs
USE_COMP=${USE_COMP:-1}                       # 1 to enable, 0 to disable
COMP_W=${COMP_W:-1.0}                         # beta in traj := traj - beta * residual_hat
COMP_APPLY_IN_TRAIN=${COMP_APPLY_IN_TRAIN:-0} # kept for future; currently compensation runs in eval path

# Optional risk penalty knobs (typically for evaluation; set non-zero if you want validation to reflect it)
RISK_W=${RISK_W:-0.0}
RISK_XY_W=${RISK_XY_W:-1.0}
RISK_YAW_W=${RISK_YAW_W:-0.2}
RISK_APPLY_IN_TRAIN=${RISK_APPLY_IN_TRAIN:-0}

echo "============================================================"
echo "[2/2] Train WoTE (CAP + response compensation wiring)"
echo "      anchors=${PLANNER_ANCHORS}"
echo "      ctrl_exec=${CTRL_EXEC}"
echo "      predictor_ckpt=${CTRL_RP_OUT}"
echo "      USE_COMP=${USE_COMP}  COMP_W=${COMP_W}"
echo "      RISK_W=${RISK_W}"
echo "============================================================"

# Build hydra overrides
EXTRA_OVERRIDES=(
  # CAP recommended: condition trajectory feature branch; disable BEV-token injection
  ++agent.config.controller_condition_scope=global
  ++agent.config.controller_style_pooling=mean
  ++agent.config.controller_condition_on_traj_feature=true
  ++agent.config.controller_traj_condition_strength=${CAP_STRENGTH}
  ++agent.config.controller_condition_on_bev_tokens=false
  ++agent.config.controller_injection_mode=${INJECT_MODE}
  ++agent.config.controller_injection_strength=${CAP_STRENGTH}
  ++agent.config.controller_inject_every_step=false

  # Wire predictor (even if you only use compensation)
  ++agent.config.controller_use_response_predictor=true
  ++agent.config.controller_response_predictor_path="${CTRL_RP_OUT}"
  ++agent.config.controller_response_predictor_trainable=false

  # Trajectory pre-compensation
  ++agent.config.controller_use_response_compensation=$([[ "${USE_COMP}" == "1" ]] && echo true || echo false)
  ++agent.config.controller_compensation_weight=${COMP_W}
  ++agent.config.controller_compensation_apply_in_train=$([[ "${COMP_APPLY_IN_TRAIN}" == "1" ]] && echo true || echo false)

  # Optional risk penalty (mostly eval/validation)
  ++agent.config.controller_risk_weight=${RISK_W}
  ++agent.config.controller_risk_xy_weight=${RISK_XY_W}
  ++agent.config.controller_risk_yaw_weight=${RISK_YAW_W}
  ++agent.config.controller_risk_apply_in_train=$([[ "${RISK_APPLY_IN_TRAIN}" == "1" ]] && echo true || echo false)
)

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
  experiment_name="${EXP_BASE}/cap_trajfeat_comp" \
  scene_filter=navtrain \
  agent.lr=${LR} \
  agent.config.min_lr=${MIN_LR} \
  agent.config.warmup_epochs=${WARMUP_EPOCHS} \
  "${EXTRA_OVERRIDES[@]}"

echo "[DONE] Training finished."
echo "- Predictor ckpt: ${CTRL_RP_OUT}"
echo "- WoTE exp: ${EXP_BASE}/cap_trajfeat_comp"
