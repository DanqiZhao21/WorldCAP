#!/usr/bin/env bash
set -euo pipefail

# Launch two WoTE finetuning runs with frozen support-conditioned HERM:
# 1) HERM-executed candidates only, no controller-token latent transition.
# 2) HERM-executed candidates + controller-token latent transition.

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}
source "${ROOT}/tool/common/worldcap_newctrl_paths.sh"

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

export PYTHONPATH="${ROOT}:${ROOT}/navsim:${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}/trainingResult"}
export WOTE_CTRL_STYLE_SPLIT=${WOTE_CTRL_STYLE_SPLIT:-train}

PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}
CTRL_REF=${CTRL_REF:-"${WORLDCAP_CTRL_REF_1024}"}
CTRL_EXEC=${CTRL_EXEC:-"${WORLDCAP_CTRL_EXEC_1024}"}
HERM_CKPT=${HERM_CKPT:-"${ROOT}/trainingResult/HERM/herm_support_1024_768_256_convattn_200ep.pt"}
WOTE_INIT_CKPT=${WOTE_INIT_CKPT:-"${ROOT}/trainingResult/epoch=29-step=19950.ckpt"}
CACHE_PATH=${CACHE_PATH:-"${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114"}

BATCH_SIZE=${WOTE_BATCH_SIZE:-16}
MAX_EPOCHS=${WOTE_MAX_EPOCHS:-40}
LR=${WOTE_LR:-"1e-4"}
MIN_LR=${WOTE_MIN_LR:-"1e-6"}
WARMUP_EPOCHS=${WOTE_WARMUP_EPOCHS:-3}
DEVICES_PER_RUN=${DEVICES_PER_RUN:-1}

GPU_IDS_HERM_ONLY=${GPU_IDS_HERM_ONLY:-0}
GPU_IDS_HERM_CTRL=${GPU_IDS_HERM_CTRL:-1}

RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}
LOG_DIR=${LOG_DIR:-"${ROOT}/trainingResult/logs/herm_wote_${RUN_TS}"}
mkdir -p "${LOG_DIR}"

export WOTE_WANDB_PROJECT=${WOTE_WANDB_PROJECT:-"WOTE-training-2"}
export WOTE_WANDB_GROUP=${WOTE_WANDB_GROUP:-"wote-herm-exec-${RUN_TS}"}
export WOTE_WANDB_TAGS=${WOTE_WANDB_TAGS:-"wote,herm,exec,controller-ablation"}

"${PYTHON}" - <<PY
import os
paths = {
    "PLANNER_ANCHORS": r"""${PLANNER_ANCHORS}""",
    "CTRL_REF": r"""${CTRL_REF}""",
    "CTRL_EXEC": r"""${CTRL_EXEC}""",
    "HERM_CKPT": r"""${HERM_CKPT}""",
    "WOTE_INIT_CKPT": r"""${WOTE_INIT_CKPT}""",
    "CACHE_PATH": r"""${CACHE_PATH}""",
}
missing = [f"{name}={path}" for name, path in paths.items() if not os.path.exists(path)]
if missing:
    raise SystemExit("[ERR] Missing required path(s):\n" + "\n".join(missing))
print("[INFO] HERM/WoTE training preflight paths OK")
PY

launch_one() {
  local run_key="$1"; shift
  local gpu_ids="$1"; shift
  local train_profile="$1"; shift
  local ctrl_wm="$1"; shift

  local run_name="wote_${run_key}_${RUN_TS}"
  local log_path="${LOG_DIR}/${run_key}.log"
  local pid_path="${LOG_DIR}/${run_key}.pid"
  local cmd_path="${LOG_DIR}/${run_key}.cmd.sh"

  echo "[LAUNCH] ${run_name}"
  echo "  gpu_ids=${gpu_ids}"
  echo "  train_profile=${train_profile}"
  echo "  controller_condition_on_world_model=${ctrl_wm}"
  echo "  log=${log_path}"

  cat > "${cmd_path}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
    cd "${ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    export WOTE_TRAIN_PROFILE="${train_profile}"
    export WOTE_INIT_CKPT="${WOTE_INIT_CKPT}"
    export WOTE_WANDB_RUN_NAME="${run_name}"
    export WOTE_WANDB_PROJECT="${WOTE_WANDB_PROJECT}"
    export WOTE_WANDB_GROUP="${WOTE_WANDB_GROUP}"
    export WOTE_WANDB_TAGS="${WOTE_WANDB_TAGS}"
    export PYTHONPATH="${PYTHONPATH}"
    export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT}"
    export WOTE_CTRL_STYLE_SPLIT="${WOTE_CTRL_STYLE_SPLIT}"
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
      +trainer.params.devices=${DEVICES_PER_RUN} \
      split=trainval \
      experiment_name="WoTE/herm_exec/${run_key}_${RUN_TS}" \
      scene_filter=navtrain \
      agent.lr=${LR} \
      agent.config.min_lr=${MIN_LR} \
      agent.config.warmup_epochs=${WARMUP_EPOCHS} \
      ++agent.config.herm_enable=true \
      ++agent.config.herm_checkpoint_path="${HERM_CKPT}" \
      ++agent.config.herm_support_size=768 \
      ++agent.config.herm_support_seed=0 \
      ++agent.config.herm_query_chunk_size=256 \
      ++agent.config.herm_apply_in_train=true \
      ++agent.config.herm_apply_in_eval=true \
      ++agent.config.controller_condition_on_world_model=${ctrl_wm} \
      ++agent.config.controller_world_model_fusion=attn_film \
      ++agent.config.controller_world_model_inject_target=all \
      ++agent.config.controller_feature_mode=full \
      ++agent.config.controller_style_pooling=attn \
      ++agent.config.use_agent_loss=false \
      ++agent.config.use_map_loss=true \
      ++agent.config.bev_semantic_weight=0.5 \
      ++agent.config.fut_bev_semantic_weight=1.0 \
      ++agent.config.traj_offset_loss_weight=0.2 \
      ++agent.config.offset_im_reward_weight=0.1
EOF
  chmod +x "${cmd_path}"

  nohup bash "${cmd_path}" >"${log_path}" 2>&1 < /dev/null &

  echo "$!" > "${pid_path}"
  echo "[PID] ${run_key} $(cat "${pid_path}")"
}

launch_one "herm_exec_only" "${GPU_IDS_HERM_ONLY}" "herm_wm_reward_only" "false"
launch_one "herm_exec_controller_token" "${GPU_IDS_HERM_CTRL}" "wm_reward_only" "true"

echo "[INFO] Both runs launched. Logs and pid files are under: ${LOG_DIR}"
