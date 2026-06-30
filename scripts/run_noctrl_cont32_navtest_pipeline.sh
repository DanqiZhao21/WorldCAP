#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhaodanqi/clone/WoTE"
BASE_CKPT="/home/zhaodanqi/clone/WoTE/CtrlNew/ckpts/WOTE-epoch=29-step=19950.ckpt"
RUN_ID="${1:-noctrl_cont32_from_epoch29_$(date +%Y%m%d_%H%M%S)}"

export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/data/navsim_workspace/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/data/navsim_workspace/exp}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/data/nuplan/dataset/maps}"
export PYTHONPATH="${ROOT}:${ROOT}/nuplan-devkit:${PYTHONPATH:-}"

LOG_ROOT="${NAVSIM_EXP_ROOT}/runs/WoTE/${RUN_ID}"
mkdir -p "${LOG_ROOT}"
SUMMARY="${LOG_ROOT}/summary.txt"

cd "${ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${SUMMARY}"
}

summarize_csv() {
  local label="$1"
  local csv="$2"
  python - "$label" "$csv" "$SUMMARY" <<'PY'
import sys
import pandas as pd

label, csv, summary = sys.argv[1:4]
df = pd.read_csv(csv)
row = df[df["token"].eq("average")].iloc[0]
fields = [
    "valid",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "score",
]
lines = [f"{label}: {csv}"]
for field in fields:
    if field in row:
        lines.append(f"  {field}: {row[field]}")
with open(summary, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

latest_csv_in() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name '*.csv' -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

run_navtest_eval() {
  local label="$1"
  local ckpt="$2"
  local experiment_name="$3"
  local out_dir="${NAVSIM_EXP_ROOT}/${experiment_name}"

  log "Starting ${label} navtest eval"
  log "  ckpt=${ckpt}"
  log "  output=${out_dir}"

  export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
  python ./navsim/planning/script/run_pdm_score_gpu.py \
    agent=WoTE_agent \
    "agent.checkpoint_path=\"${ckpt}\"" \
    agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
    agent.config.use_controller_wm=false \
    experiment_name="${experiment_name}" \
    split=test \
    scene_filter=navtest \
    worker=sequential \
    verbose=false \
    logger_level=warning \
    '~scorer.anchor_save_dir' \
    '~scorer.anchor_save_name' \
    '~scorer.anchor_overwrite'

  local csv
  csv="$(latest_csv_in "${out_dir}")"
  if [[ -z "${csv}" ]]; then
    log "ERROR: ${label} navtest eval completed but no CSV was found in ${out_dir}"
    exit 1
  fi
  summarize_csv "${label}" "${csv}" | tee -a "${SUMMARY}"
}

log "Pipeline started: ${RUN_ID}"
log "Base ckpt: ${BASE_CKPT}"

run_navtest_eval \
  "base_epoch29" \
  "${BASE_CKPT}" \
  "eval/WoTE/${RUN_ID}_base_navtest"

log "Starting 32-epoch no-controller continuation training"
TRAIN_MARKER="${LOG_ROOT}/train_start.marker"
: > "${TRAIN_MARKER}"

export WOTE_INIT_CKPT="${BASE_CKPT}"
export WOTE_TRAIN_PROFILE="wote_no_controller"
export WOTE_WANDB_RUN_NAME="${RUN_ID}_train32_no_controller"
export WOTE_WANDB_GROUP="noctrl_cont32_control"
export WOTE_WANDB_TAGS="no_controller,control,continue32"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

python ./navsim/planning/script/run_training.py \
  agent=WoTE_agent \
  agent.config.use_controller_wm=false \
  experiment_name="training/WoTE/${RUN_ID}_train32_no_controller" \
  use_cache_without_dataset=true \
  cache_path=/mnt/data/navsim_workspace/exp/training_cache_allWoteTarget \
  trainer.params.max_epochs=32 \
  +trainer.params.devices="${TRAIN_DEVICES:-8}"

NEW_CKPT="$(
  find -L "${ROOT}/trainingResult" -mindepth 2 -maxdepth 2 -type f -name 'last.ckpt' -newer "${TRAIN_MARKER}" -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
)"

if [[ -z "${NEW_CKPT}" ]]; then
  NEW_CKPT="$(
    find -L "${ROOT}/trainingResult" -mindepth 2 -maxdepth 2 -type f -name '*.ckpt' -newer "${TRAIN_MARKER}" -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
fi

if [[ -z "${NEW_CKPT}" ]]; then
  log "ERROR: training completed but no new checkpoint was found"
  exit 1
fi

log "Training completed"
log "New ckpt: ${NEW_CKPT}"

run_navtest_eval \
  "cont32_no_controller" \
  "${NEW_CKPT}" \
  "eval/WoTE/${RUN_ID}_cont32_navtest"

log "Pipeline completed: ${RUN_ID}"
