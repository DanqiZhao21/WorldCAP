#!/bin/bash
set -euo pipefail

# Evaluate WoTE ckpt with latest CAP + response-based trajectory pre-compensation (2026-02).
# - CAP: controller conditioning on trajectory-feature branch (recommended)
# - Response predictor: optional, but required when using compensation or risk penalty
# - Pre-compensation is applied inside WoTE_model.py when agent.is_eval = True
#
# Default CKPT:
#   /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260205_020624/epoch=19-step=26600.ckpt

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
CKPT=${CKPT:-"${ROOT}/trainingResult/ckpts_20260205_020624/epoch=19-step=26600.ckpt"}

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

# GPUs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Runtime env
export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}

# Planner anchors (keep consistent with training if you overrode it)
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}

# Controller bank paths (used to compute controller embedding phi at eval time)
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC_DEFAULT=${CTRL_EXEC_DEFAULT:-"${ROOT}/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy"}
CTRL_EXEC_POST1515=${CTRL_EXEC_POST1515:-"${ROOT}/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy"}
CTRL_EXEC_AGGRESSIVE=${CTRL_EXEC_AGGRESSIVE:-"${ROOT}/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy"}

# Response predictor ckpt (trained by tool/controller_response/*)
CTRL_RP_CKPT=${CTRL_RP_CKPT:-"${ROOT}/ControllerExp/generated/controller_response_predictor.pt"}

# Output root (under ${NAVSIM_EXP_ROOT})
EXP_ROOT=${EXP_ROOT:-"eval/WoTE/ckpt_20260205_020624/cap_trajfeat_comp"}

# =============================
# CAP conditioning switches
# =============================
# MODE:
#   cap_trajfeat : controller affects per-trajectory feature branch; no BEV token injection (recommended)
#   global_style : legacy global style injection into BEV tokens
MODE=${MODE:-cap_trajfeat}
INJ_MODE=${INJ_MODE:-film}     # film|attn|concat|add
POOLING=${POOLING:-mean}       # mean|attn
INJ_STRENGTH=${INJ_STRENGTH:-0.1}
TRAJ_COND_STRENGTH=${TRAJ_COND_STRENGTH:-0.1}

COND_ARGS=()
if [[ "${MODE}" == "cap_trajfeat" ]]; then
  COND_ARGS=(
    ++agent.config.controller_condition_scope=global
    ++agent.config.controller_style_pooling="${POOLING}"
    ++agent.config.controller_condition_on_traj_feature=true
    ++agent.config.controller_traj_condition_strength=${TRAJ_COND_STRENGTH}
    ++agent.config.controller_condition_on_bev_tokens=false
    ++agent.config.controller_injection_mode="${INJ_MODE}"
    ++agent.config.controller_injection_strength=${INJ_STRENGTH}
    ++agent.config.controller_inject_every_step=false
  )
elif [[ "${MODE}" == "global_style" ]]; then
  COND_ARGS=(
    ++agent.config.controller_condition_scope=global
    ++agent.config.controller_style_pooling="${POOLING}"
    ++agent.config.controller_condition_on_traj_feature=false
    ++agent.config.controller_condition_on_bev_tokens=true
    ++agent.config.controller_injection_mode="${INJ_MODE}"
    ++agent.config.controller_injection_strength=${INJ_STRENGTH}
    ++agent.config.controller_inject_every_step=false
  )
else
  echo "[ERR] Unknown MODE=${MODE} (cap_trajfeat|global_style)" >&2
  exit 1
fi

# =============================
# Response-based trajectory pre-compensation
# =============================
# Enable with USE_COMP=1 (default). Requires CTRL_RP_CKPT.
USE_COMP=${USE_COMP:-1}
COMP_W=${COMP_W:-1.0}  # beta in: traj := traj - beta * residual_hat

# =============================
# Optional: response-based risk penalty
# =============================
# Set RISK_W>0 to enable risk penalty (requires predictor).
RISK_W=${RISK_W:-0.0}
RISK_XY_W=${RISK_XY_W:-1.0}
RISK_YAW_W=${RISK_YAW_W:-0.2}

PRED_ARGS=()
COMP_ARGS=()
RISK_ARGS=()

need_predictor=0
if [[ "${USE_COMP}" == "1" ]]; then
  need_predictor=1
  COMP_ARGS=(
    ++agent.config.controller_use_response_compensation=true
    ++agent.config.controller_compensation_weight=${COMP_W}
    ++agent.config.controller_compensation_apply_in_train=false
  )
else
  COMP_ARGS=(
    ++agent.config.controller_use_response_compensation=false
    ++agent.config.controller_compensation_weight=${COMP_W}
    ++agent.config.controller_compensation_apply_in_train=false
  )
fi

# Risk only when RISK_W > 0
if "${PYTHON}" - <<'PY'
import os, sys
try:
    w = float(os.environ.get('RISK_W', '0') or '0')
except Exception:
    w = 0.0
sys.exit(0 if w > 0.0 else 1)
PY
then
  need_predictor=1
  RISK_ARGS=(
    ++agent.config.controller_risk_weight=${RISK_W}
    ++agent.config.controller_risk_xy_weight=${RISK_XY_W}
    ++agent.config.controller_risk_yaw_weight=${RISK_YAW_W}
    ++agent.config.controller_risk_apply_in_train=false
  )
else
  RISK_ARGS=(
    ++agent.config.controller_risk_weight=${RISK_W}
    ++agent.config.controller_risk_xy_weight=${RISK_XY_W}
    ++agent.config.controller_risk_yaw_weight=${RISK_YAW_W}
    ++agent.config.controller_risk_apply_in_train=false
  )
fi

if [[ "${need_predictor}" == "1" ]]; then
  if [[ -z "${CTRL_RP_CKPT}" || ! -f "${CTRL_RP_CKPT}" ]]; then
    echo "[ERR] Need response predictor, but CTRL_RP_CKPT not found: ${CTRL_RP_CKPT}" >&2
    echo "      Set CTRL_RP_CKPT=/abs/path/to/controller_response_predictor.pt or disable USE_COMP and set RISK_W=0" >&2
    exit 1
  fi
  PRED_ARGS=(
    ++agent.config.controller_use_response_predictor=true
    ++agent.config.controller_response_predictor_path="${CTRL_RP_CKPT}"
    ++agent.config.controller_response_predictor_trainable=false
  )
else
  PRED_ARGS=(
    ++agent.config.controller_use_response_predictor=false
    ++agent.config.controller_response_predictor_trainable=false
  )
fi

run_one() {
  local TAG="$1"; shift
  local TRACKER_STYLE="$1"; shift
  local POST_STYLE="$1"; shift
  local H_SCALE="$1"; shift
  local S_SCALE="$1"; shift
  local CTRL_EXEC="$1"; shift

  "${PYTHON}" "${SCRIPT}" \
    agent=WoTE_agent \
    "agent.checkpoint_path='${CKPT}'" \
    ++agent.config.cluster_file_path="${PLANNER_ANCHORS}" \
    experiment_name="${EXP_ROOT}/${TAG}" \
    split=test \
    scene_filter=navtest \
    simulator.tracker_style="${TRACKER_STYLE}" \
    simulator.post_style="${POST_STYLE}" \
    simulator.post_params.heading_scale=${H_SCALE} \
    simulator.post_params.heading_bias=0 \
    simulator.post_params.speed_scale=${S_SCALE} \
    simulator.post_params.speed_bias=0 \
    simulator.post_params.noise_std=0 \
    agent.config.controller_ref_traj_path="${CTRL_REF}" \
    agent.config.controller_exec_traj_path="${CTRL_EXEC}" \
    "${COND_ARGS[@]}" \
    "${PRED_ARGS[@]}" \
    "${COMP_ARGS[@]}" \
    "${RISK_ARGS[@]}" \
    "$@"
}

echo "============================================================"
echo "[EVAL] ckpt=${CKPT}"
echo "      anchors=${PLANNER_ANCHORS}"
echo "      mode=${MODE} inj=${INJ_MODE} pooling=${POOLING} inj_strength=${INJ_STRENGTH} traj_strength=${TRAJ_COND_STRENGTH}"
echo "      use_comp=${USE_COMP} comp_w=${COMP_W}"
echo "      risk_w=${RISK_W}"
echo "      predictor=${CTRL_RP_CKPT}"
echo "      out=${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"
echo "============================================================"

# 1) default tracker + no post
run_one "sim_default_none" \
  default none 1.0 1.0 "${CTRL_EXEC_DEFAULT}"

# 2) default tracker + poststyle yaw/speed extreme 1.5
run_one "sim_default_post1515" \
  default yaw_speed_extreme 1.5 1.5 "${CTRL_EXEC_POST1515}"

# 3) aggressive tracker + no post
run_one "sim_aggressive_none" \
  aggressive none 1.0 1.0 "${CTRL_EXEC_AGGRESSIVE}"

echo "[DONE] Outputs under: ${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"
