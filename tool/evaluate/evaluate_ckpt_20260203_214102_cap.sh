#!/bin/bash
set -euo pipefail

# Evaluate a single ckpt with the latest CAP setup (controller-aware planning).
# - Uses controller conditioning in trajectory-feature branch (recommended)
# - Optionally enables Controller Response Predictor risk penalty
#
# CKPT (given by you):
#   /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260203_214102/epoch=19-step=26600.ckpt

ROOT=/home/zhaodanqi/clone/WoTE
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
CKPT=${CKPT:-"${ROOT}/trainingResult/ckpts_20260203_214102/epoch=19-step=26600.ckpt"}

# Prefer the currently active environment's python.
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

export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}

# Controller bank paths used only to compute controller embedding phi at eval time
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC_DEFAULT=${CTRL_EXEC_DEFAULT:-"${ROOT}/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy"}
CTRL_EXEC_POST1515=${CTRL_EXEC_POST1515:-"${ROOT}/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy"}
CTRL_EXEC_AGGRESSIVE=${CTRL_EXEC_AGGRESSIVE:-"${ROOT}/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy"}

# Output root
EXP_ROOT=${EXP_ROOT:-"eval/WoTE/ckpt_20260203_214102"}

# =============================
# CAP conditioning switches
# =============================
# MODE:
#   cap_trajfeat : controller affects per-trajectory feature branch; no BEV token injection (recommended)
#   global_style : legacy global style injection into BEV tokens
MODE=${MODE:-cap_trajfeat}
INJ_MODE=${INJ_MODE:-film}
POOLING=${POOLING:-mean}
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
    agent.config.controller_injection_mode="${INJ_MODE}"
    agent.config.controller_injection_strength=${INJ_STRENGTH}
    agent.config.controller_inject_every_step=false
  )
elif [[ "${MODE}" == "global_style" ]]; then
  COND_ARGS=(
    ++agent.config.controller_condition_scope=global
    ++agent.config.controller_style_pooling="${POOLING}"
    ++agent.config.controller_condition_on_traj_feature=false
    ++agent.config.controller_condition_on_bev_tokens=true
    agent.config.controller_injection_mode="${INJ_MODE}"
    agent.config.controller_injection_strength=${INJ_STRENGTH}
    agent.config.controller_inject_every_step=false
  )
else
  echo "[ERR] Unknown MODE=${MODE} (cap_trajfeat|global_style)" >&2
  exit 1
fi

# =============================
# Optional: CAP risk penalty args
# =============================
# Enable with:
#   export WOTE_USE_CTRL_RISK=1
#   export WOTE_CTRL_RP_CKPT=/abs/path/to/controller_response_predictor.pt
#   export WOTE_CTRL_RISK_W=0.2
WOTE_USE_CTRL_RISK=${WOTE_USE_CTRL_RISK:-0}
WOTE_CTRL_RP_CKPT=${WOTE_CTRL_RP_CKPT:-""}
WOTE_CTRL_RISK_W=${WOTE_CTRL_RISK_W:-0.0}
WOTE_CTRL_RISK_XY_W=${WOTE_CTRL_RISK_XY_W:-1.0}
WOTE_CTRL_RISK_YAW_W=${WOTE_CTRL_RISK_YAW_W:-0.2}

RISK_ARGS=()
if [[ "${WOTE_USE_CTRL_RISK}" == "1" ]]; then
  if [[ -z "${WOTE_CTRL_RP_CKPT}" ]]; then
    echo "[WARN] WOTE_USE_CTRL_RISK=1 but WOTE_CTRL_RP_CKPT is empty; skipping risk penalty args"
  else
    RISK_ARGS=(
      ++agent.config.controller_use_response_predictor=true
      ++agent.config.controller_response_predictor_path=\"${WOTE_CTRL_RP_CKPT}\"
      ++agent.config.controller_risk_weight=${WOTE_CTRL_RISK_W}
      ++agent.config.controller_risk_xy_weight=${WOTE_CTRL_RISK_XY_W}
      ++agent.config.controller_risk_yaw_weight=${WOTE_CTRL_RISK_YAW_W}
    )
  fi
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
    agent.checkpoint_path=\"${CKPT}\" \
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
    "${RISK_ARGS[@]}" \
    "$@"
}

echo "============================================================"
echo "[EVAL] ckpt=${CKPT}"
echo "      mode=${MODE} inj=${INJ_MODE} pooling=${POOLING} strength=${INJ_STRENGTH}"
echo "      risk=${WOTE_USE_CTRL_RISK} risk_w=${WOTE_CTRL_RISK_W} rp_ckpt=${WOTE_CTRL_RP_CKPT}"
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
