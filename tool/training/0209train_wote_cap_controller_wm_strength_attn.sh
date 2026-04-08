#!/usr/bin/env bash
set -euo pipefail

# WoTE training script (controller-conditioned world model).
# NOTE: WoTE_model.py now supports controller_world_model_fusion=attn (default).
# In attn fusion mode, controller_world_model_strength is ignored (kept only for legacy 'add' mode).
# This script keeps the old strength sweep for backward-compatibility / bookkeeping.
#
# Based on: tool/training/train_wote_cap_controller_minimal.sh
#
# Recommended for this setting (freeze policy; train WM + reward heads only)
export WOTE_TRAIN_PROFILE=${WOTE_TRAIN_PROFILE:-wm_reward_only}

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

# GPUs
# IMPORTANT:
# - Lightning can only use GPUs that are *visible* to this process.
# - If your environment (tmux/ssh/slurm/docker) has CUDA_VISIBLE_DEVICES preset (e.g. "0"),
#   then only that GPU will be visible and requesting DEVICES>1 will crash.
#
# Default to 8 GPUs (0-7) on your machine; override if needed:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICES=4 bash tool/training/0209train_wote_cap_controller_wm_strength_06_09.sh
export CUDA_VISIBLE_DEVICES=0,1
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

# NAVSIM_EXP_ROOT is used by hydra config to construct output_dir.
NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}"}
export NAVSIM_EXP_ROOT

CACHE_PATH=${CACHE_PATH:-"${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114"}

# -----------------------------
# Training hyperparams
# -----------------------------
BATCH_SIZE=${WOTE_BATCH_SIZE:-16}
MAX_EPOCHS=${WOTE_MAX_EPOCHS:-40}
LR=${WOTE_LR:-"1e-4"}
MIN_LR=${WOTE_MIN_LR:-"1e-6"}
WARMUP_EPOCHS=${WOTE_WARMUP_EPOCHS:-3}

# Devices for Lightning Trainer.
# If not specified, derive it from CUDA_VISIBLE_DEVICES.
if [[ -z "${DEVICES:-}" ]]; then
  _cvis="${CUDA_VISIBLE_DEVICES// /}"
  if [[ "${_cvis}" =~ ^[0-9]+-[0-9]+$ ]]; then
    _start="${_cvis%-*}"
    _end="${_cvis#*-}"
    DEVICES=$(( _end - _start + 1 ))
  else
    IFS=',' read -ra _gpu_ids <<<"${_cvis}"
    DEVICES=${#_gpu_ids[@]}
  fi
fi
echo "[INFO] trainer.devices=${DEVICES}"

# Sanity check: what torch actually sees.
"${PYTHON}" - <<'PY'
import torch
print('[INFO] torch.cuda.is_available:', torch.cuda.is_available())
print('[INFO] torch.cuda.device_count :', torch.cuda.device_count())
if torch.cuda.is_available():
    names = []
    for i in range(torch.cuda.device_count()):
        try:
            names.append(torch.cuda.get_device_name(i))
        except Exception:
            names.append('unknown')
    print('[INFO] torch visible gpus     :', names)
PY

# -----------------------------
# Controller → WM knobs
# -----------------------------
CTRL_FEATURE_MODE=${CTRL_FEATURE_MODE:-full}        # full | lateral_only
CTRL_POOLING=${CTRL_POOLING:-attn}                 # attn | mean
WM_INJECT_TARGET=${WM_INJECT_TARGET:-all}          # all | ego
WM_FUSION=${WM_FUSION:-attn}                      # attn | add

# Loss weights (effective in WoTE_loss.py)
IM_LOSS_W=${IM_LOSS_W:-1.0}
METRIC_LOSS_W=${METRIC_LOSS_W:-1.0}

# Experiment
EXP_BASE=${EXP_BASE:-"WoTE/cap_controller_min"}
EXP_TAG_BASE=${EXP_TAG_BASE:-"ctrl_cap_wm_strength_sweep"}

# Unique run id to avoid overwriting outputs across repeated executions.
# You can override explicitly, e.g.:
#   EXP_RUN_ID=ablation01 bash tool/training/0209train_wote_cap_controller_wm_strength_06_09.sh
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}
EXP_RUN_ID=${EXP_RUN_ID:-"${RUN_TS}"}

# -----------------------------
# Run control
# -----------------------------
# In attn fusion mode, wm_strength is ignored by the model (kept only for legacy add-mode).
# Default to a single run unless you explicitly enable sweep.
SWEEP_WM_STRENGTH=${SWEEP_WM_STRENGTH:-0}   # 0|1
WM_STRENGTH_SINGLE=${WM_STRENGTH_SINGLE:-0.6}
# You can override strengths explicitly, e.g.:
#   WM_STRENGTHS="0.3 0.6 0.9" bash ...
WM_STRENGTHS=${WM_STRENGTHS:-""}

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
cfg.controller_world_model_fusion = r"""${WM_FUSION}"""

m = WoTEModel(cfg)
pooling = str(r"""${CTRL_POOLING}""" or 'attn').lower()
fusion = str(r"""${WM_FUSION}""" or 'attn').lower()

need = ['controller_style_pooling', 'ctrl_proj', 'ctrl_token_ln', 'controller_encoder']
if pooling == 'attn':
  need += ['ctrl_style_attn']
if fusion == 'attn':
  need += ['ctrl_fuse_attn', 'ctrl_bank_proj', 'ctrl_bank_ln']

missing = [k for k in need if not hasattr(m, k)]
if missing:
  raise RuntimeError('Preflight failed, missing attrs: ' + ','.join(missing))

ctrl_exec = r"""${CTRL_EXEC}"""
if ctrl_exec and (not os.path.isfile(ctrl_exec)):
  print('[WARN] CTRL_EXEC does not exist on disk:', ctrl_exec)
print('[INFO] Preflight OK')
PY

run_one() {
  local wm_strength="$1"; shift
  local exp_suffix="$1"; shift

  echo "============================================================"
  echo "[TRAIN] ${EXP_BASE}/${exp_suffix}"
  echo "  anchors      = ${PLANNER_ANCHORS}"
  echo "  ctrl_ref     = ${CTRL_REF}"
  echo "  ctrl_exec    = ${CTRL_EXEC}"
  echo "  cache_path   = ${CACHE_PATH}"
  echo "  devices      = ${DEVICES} (visible: ${CUDA_VISIBLE_DEVICES})"
  echo "  ctrl->wm     = enable=true fusion=${WM_FUSION} strength=${wm_strength} target=${WM_INJECT_TARGET}"
  echo "  ctrl_pooling = ${CTRL_POOLING}"
  echo "  feature_mode = ${CTRL_FEATURE_MODE}"
  echo "  im_w/metric_w= ${IM_LOSS_W}/${METRIC_LOSS_W}"
  echo "  WOTE_TRAIN_PROFILE=${WOTE_TRAIN_PROFILE}"
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
    experiment_name="${EXP_BASE}/${exp_suffix}" \
    scene_filter=navtrain \
    agent.lr=${LR} \
    agent.config.min_lr=${MIN_LR} \
    agent.config.warmup_epochs=${WARMUP_EPOCHS} \
    ++agent.config.controller_feature_mode=${CTRL_FEATURE_MODE} \
    ++agent.config.controller_style_pooling=${CTRL_POOLING} \
    ++agent.config.controller_condition_on_world_model=true \
    ++agent.config.controller_world_model_fusion=${WM_FUSION} \
    ++agent.config.controller_world_model_strength=${wm_strength} \
    ++agent.config.controller_world_model_inject_target=${WM_INJECT_TARGET} \
    ++agent.config.use_agent_loss=false \
    ++agent.config.use_map_loss=true \
    ++agent.config.bev_semantic_weight=0.0 \
    ++agent.config.fut_bev_semantic_weight=1.0 \
    ++agent.config.traj_offset_loss_weight=0.0 \
    ++agent.config.offset_im_reward_weight=0.0 \
    ++agent.config.im_loss_weight=${IM_LOSS_W} \
    ++agent.config.metric_loss_weight=${METRIC_LOSS_W} \
    "$@"
}

if [[ -z "${WM_STRENGTHS}" ]]; then
  if [[ "${SWEEP_WM_STRENGTH}" == "1" ]]; then
    WM_STRENGTHS="0.6 0.9"
  else
    WM_STRENGTHS="${WM_STRENGTH_SINGLE}"
  fi
fi

echo "[INFO] WM_FUSION=${WM_FUSION} strengths: ${WM_STRENGTHS} (SWEEP_WM_STRENGTH=${SWEEP_WM_STRENGTH})"

for wm_strength in ${WM_STRENGTHS}; do
  suffix_strength="${wm_strength//./p}"
  run_one "${wm_strength}" "${EXP_TAG_BASE}_${EXP_RUN_ID}_wm${suffix_strength}"
done

echo "[DONE] Finished training runs."
