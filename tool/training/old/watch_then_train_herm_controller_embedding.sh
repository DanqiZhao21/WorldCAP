#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}
WATCH_PID=${WATCH_PID:?WATCH_PID is required}
GPU_IDS=${GPU_IDS:-0,1,2,3}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}

LOG_ROOT=${LOG_ROOT:-"${ROOT}/trainingResult/logs"}
LOG_DIR="${LOG_ROOT}/wote_herm_controller_embedding_4gpu_32ep_${RUN_TS}"
mkdir -p "${LOG_DIR}"

WATCH_LOG="${LOG_DIR}/watcher.log"
TRAIN_LOG="${LOG_DIR}/train.log"

{
  echo "[INFO] $(date '+%F %T %Z') watching PID ${WATCH_PID}"
  while kill -0 "${WATCH_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[INFO] $(date '+%F %T %Z') PID ${WATCH_PID} exited; starting HERM + controller embedding training"
  echo "[INFO] log: ${TRAIN_LOG}"
} >>"${WATCH_LOG}" 2>&1

cd "${ROOT}"
GPU_IDS="${GPU_IDS}" RUN_TS="${RUN_TS}" \
  bash tool/training/0515train_wote_herm_controller_embedding_4gpu_32ep.sh \
  >"${TRAIN_LOG}" 2>&1

echo "[INFO] $(date '+%F %T %Z') follow-up training exited with status $?" >>"${WATCH_LOG}" 2>&1
