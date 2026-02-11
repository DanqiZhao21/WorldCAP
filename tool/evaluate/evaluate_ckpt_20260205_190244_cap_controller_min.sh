#!/bin/bash
set -euo pipefail

# Evaluate WoTE ckpt with the latest minimal CAP controller conditioning (2026-02).
# Matches: tool/training/train_wote_cap_controller_minimal.sh
# - controller embedding injected into:
#   (A) traj feature branch (CAP)
#   (B) offset branch
#   (C) reward_feature (scoring)
# - No response predictor / risk penalty / pre-compensation.
#
# Default CKPT:
#   /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260205_190244/epoch=39-step=53200.ckpt

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
CKPT=${CKPT:-"${ROOT}/trainingResult/ckpts_20260205_190244/epoch=39-step=53200.ckpt"}

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

# NAVSIM_EXP_ROOT controls where hydra writes outputs (used by default configs).
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}"}

# Planner anchors (keep consistent with training if you overrode it)
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}

# Controller bank paths (used to compute controller embedding phi at eval time)
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC_DEFAULT=${CTRL_EXEC_DEFAULT:-"${ROOT}/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy"}
CTRL_EXEC_POST1515=${CTRL_EXEC_POST1515:-"${ROOT}/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy"}
CTRL_EXEC_AGGRESSIVE=${CTRL_EXEC_AGGRESSIVE:-"${ROOT}/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy"}

# Output root (under ${NAVSIM_EXP_ROOT})
EXP_ROOT=${EXP_ROOT:-"eval/WoTE/ckpt_20260205_190244/cap_controller_min"}

# =============================
# Controller injection knobs (match training script)
# =============================
CTRL_FEATURE_MODE=${CTRL_FEATURE_MODE:-full}          # full | lateral_only
INJECT_MODE=${INJECT_MODE:-film}                     # film | add | none
INJECT_STRENGTH=${INJECT_STRENGTH:-0.25}             # 0~1
CTRL_POOLING=${CTRL_POOLING:-attn}                   # attn | mean

COND_TRAJ=${COND_TRAJ:-true}
TRAJ_STRENGTH=${TRAJ_STRENGTH:-0.25}

COND_OFFSET=${COND_OFFSET:-true}
OFFSET_STRENGTH=${OFFSET_STRENGTH:-0.25}

COND_REWARD=${COND_REWARD:-true}
REWARD_STRENGTH=${REWARD_STRENGTH:-0.35}

COND_ARGS=(
	++agent.config.cluster_file_path="${PLANNER_ANCHORS}"
	++agent.config.controller_feature_mode=${CTRL_FEATURE_MODE}
	++agent.config.controller_style_pooling=${CTRL_POOLING}
	++agent.config.controller_injection_mode=${INJECT_MODE}
	++agent.config.controller_injection_strength=${INJECT_STRENGTH}
	++agent.config.controller_condition_on_traj_feature=${COND_TRAJ}
	++agent.config.controller_traj_condition_strength=${TRAJ_STRENGTH}
	++agent.config.controller_condition_on_offset=${COND_OFFSET}
	++agent.config.controller_offset_condition_strength=${OFFSET_STRENGTH}
	++agent.config.controller_condition_on_reward_feature=${COND_REWARD}
	++agent.config.controller_reward_condition_strength=${REWARD_STRENGTH}
)

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
		"$@"
}

echo "============================================================"
echo "[EVAL] ckpt=${CKPT}"
echo "      anchors=${PLANNER_ANCHORS}"
echo "      ctrl_ref=${CTRL_REF}"
echo "      inject: mode=${INJECT_MODE} strength=${INJECT_STRENGTH} pooling=${CTRL_POOLING} feature_mode=${CTRL_FEATURE_MODE}"
echo "      traj/offset/reward = ${COND_TRAJ}/${COND_OFFSET}/${COND_REWARD}"
echo "      strengths          = ${TRAJ_STRENGTH}/${OFFSET_STRENGTH}/${REWARD_STRENGTH}"
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