#!/usr/bin/env bash
set -euo pipefail

# Wrapper: controller style bank size = 128
# NOTE: Planner anchors (PLANNER_ANCHORS) stay unchanged; only CTRL_REF/CTRL_EXEC differ.

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}

export CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_128_centered.npy"}
export CTRL_EXEC=${CTRL_EXEC:-"${ROOT}/ControllerExp/generated/128/controller_styles_128.npz"}

# Only train the attn fusion for this experiment (override by env if needed)
export RUN_FUSIONS=${RUN_FUSIONS:-"attn"}

# Distinguish experiment names by default (can override from env)
export EXP_BASE=${EXP_BASE:-"WoTE/cap_controller_bundle_split_ctrl128"}
export EXP_TAG_BASE=${EXP_TAG_BASE:-"ctrlbundle_trainSplit_ctrl128"}

exec bash "${ROOT}/tool/training/0302train_wote_ctrlbundle_split_train_attn.sh" "$@"

#CUDA_VISIBLE_DEVICES=1 NAVSIM_EXP_ROOT=/mnt/data/navsim_workspace/exp bash tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl128.sh