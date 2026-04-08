#!/usr/bin/env bash
set -euo pipefail

# Sweep controller injection strength for OFFSET branch.
# Runs 3 trainings with controller_offset_inject_strength = 0.3 / 0.6 / 0.9
# Also enables differentiable "execution" via ControllerResponsePredictor during training.

export WOTE_TRAIN_PROFILE=${WOTE_TRAIN_PROFILE:-controller_inject_offset_reward}

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}

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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"

PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC=${CTRL_EXEC:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}
CTRL_RP_CKPT=${CTRL_RP_CKPT:-"${ROOT}/ControllerExp/generated/controller_response_predictor.pt"}

NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}"}
export NAVSIM_EXP_ROOT
CACHE_PATH=${CACHE_PATH:-"${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114"}

BATCH_SIZE=${WOTE_BATCH_SIZE:-16}
MAX_EPOCHS=${WOTE_MAX_EPOCHS:-40}
LR=${WOTE_LR:-"1e-4"}
MIN_LR=${WOTE_MIN_LR:-"1e-6"}
WARMUP_EPOCHS=${WOTE_WARMUP_EPOCHS:-3}

# Defaults you asked to keep fixed for now
TRAJ_OFFSET_W=${TRAJ_OFFSET_W:-1.0}
OFFSET_IM_W=${OFFSET_IM_W:-0.5}
WM_STRENGTH=${WM_STRENGTH:-0.3}

IM_LOSS_W=${IM_LOSS_W:-1.0}
METRIC_LOSS_W=${METRIC_LOSS_W:-1.0}

CTRL_FEATURE_MODE=${CTRL_FEATURE_MODE:-full}
CTRL_POOLING=${CTRL_POOLING:-mean}

EXEC_ENABLE=${EXEC_ENABLE:-true}
EXEC_TRAIN=${EXEC_TRAIN:-true}
EXEC_EVAL=${EXEC_EVAL:-false}

EXP_BASE=${EXP_BASE:-"WoTE/cap_controller_offset"}
EXP_TAG_BASE=${EXP_TAG_BASE:-"ctrl_offset_inject_strength_sweep"}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}
EXP_RUN_ID=${EXP_RUN_ID:-"${RUN_TS}"}

run_one() {
	local inj_strength="$1"; shift
	local exp_suffix="$1"; shift

	echo "============================================================"
	echo "[TRAIN] ${EXP_BASE}/${exp_suffix}"
	echo "  offset_inject_strength = ${inj_strength}"
	echo "  ctrl_exec              = ${CTRL_EXEC}"
	echo "  ctrl_rp_ckpt           = ${CTRL_RP_CKPT}"
	echo "  cache_path             = ${CACHE_PATH}"
	echo "  loss_w                 = traj_offset=${TRAJ_OFFSET_W} offset_im=${OFFSET_IM_W}"
	echo "  wm_strength            = ${WM_STRENGTH}"
	echo "  WOTE_TRAIN_PROFILE     = ${WOTE_TRAIN_PROFILE}"
	echo "============================================================"

	"${PYTHON}" -u ./navsim/planning/script/run_training.py \
		agent=WoTE_agent \
		agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
		++agent.config.cluster_file_path="${PLANNER_ANCHORS}" \
		++agent.config.controller_ref_traj_path="${CTRL_REF}" \
		++agent.config.controller_exec_traj_path="${CTRL_EXEC}" \
		++agent.config.controller_response_predictor_path="${CTRL_RP_CKPT}" \
		++agent.config.controller_feature_mode=${CTRL_FEATURE_MODE} \
		++agent.config.controller_style_pooling=${CTRL_POOLING} \
		++agent.config.controller_condition_on_world_model=true \
		++agent.config.controller_world_model_strength=${WM_STRENGTH} \
		++agent.config.controller_condition_on_offset=true \
		++agent.config.controller_offset_inject_strength=${inj_strength} \
		++agent.config.controller_execute_predicted_traj=${EXEC_ENABLE} \
		++agent.config.controller_execute_apply_in_train=${EXEC_TRAIN} \
		++agent.config.controller_execute_apply_in_eval=${EXEC_EVAL} \
		use_cache_without_dataset=true \
		cache_path="${CACHE_PATH}" \
		dataloader.params.batch_size=${BATCH_SIZE} \
		trainer.params.max_epochs=${MAX_EPOCHS} \
		split=trainval \
		experiment_name="${EXP_BASE}/${exp_suffix}" \
		scene_filter=navtrain \
		agent.lr=${LR} \
		agent.config.min_lr=${MIN_LR} \
		agent.config.warmup_epochs=${WARMUP_EPOCHS} \
		++agent.config.traj_offset_loss_weight=${TRAJ_OFFSET_W} \
		++agent.config.offset_im_reward_weight=${OFFSET_IM_W} \
		++agent.config.im_loss_weight=${IM_LOSS_W} \
		++agent.config.metric_loss_weight=${METRIC_LOSS_W} \
		"$@"
}

run_one 0.3 "${EXP_TAG_BASE}_${EXP_RUN_ID}_off0p3"
run_one 0.6 "${EXP_TAG_BASE}_${EXP_RUN_ID}_off0p6"
run_one 0.9 "${EXP_TAG_BASE}_${EXP_RUN_ID}_off0p9"

echo "[DONE] Finished 3 runs."
