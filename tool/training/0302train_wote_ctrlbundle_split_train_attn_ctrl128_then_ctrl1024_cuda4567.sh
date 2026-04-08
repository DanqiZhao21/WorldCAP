#!/usr/bin/env bash
set -euo pipefail

# Run two experiments sequentially on the same GPUs:
#   1) controller bundle = 128
#   2) controller bundle = 1024
# Both runs train only RUN_FUSIONS=attn by default (defined in the wrapper scripts).
#
# Usage:
#   NAVSIM_EXP_ROOT=/mnt/data/navsim_workspace/exp \
#   CUDA_VISIBLE_DEVICES=4,5,6,7 \
#   bash tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl128_then_ctrl1024_cuda4567.sh

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}

# Default GPU set to 4,5,6,7 if not provided.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}

# Default exp root (can override).
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"/mnt/data/navsim_workspace/exp"}

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT}"

echo "============================================================"
echo "[STAGE 1/2] Train ctrl128 (attn)"
echo "============================================================"
bash "${ROOT}/tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl128.sh" "$@"

echo "============================================================"
echo "[STAGE 2/2] Train ctrl1024 (attn)"
echo "============================================================"
bash "${ROOT}/tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl1024.sh" "$@"

echo "[DONE] ctrl128 -> ctrl1024 finished."
