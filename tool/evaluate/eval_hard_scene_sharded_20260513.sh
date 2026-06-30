#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}
PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
SPLIT=${SPLIT:-test}
OUT_BASE=${OUT_BASE:-eval/hard_scene_sharded_20260513}
WORKER=${WORKER:-sequential}
NUM_SHARDS=${NUM_SHARDS:-4}

export PYTHONPATH="${ROOT}:${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/mnt/data/navsim_workspace/dataset}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/mnt/data/navsim_workspace/exp}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"${ROOT}/nuplan-devkit/nuplan/common/maps"}

CKPT_NEW=${CKPT_NEW:-"${ROOT}/trainingResult/ckpts_20260511_204702/epoch=19-step=21280.ckpt"}
CKPT_BASE=${CKPT_BASE:-"${ROOT}/trainingResult/epoch=29-step=19950.ckpt"}

run_shard() {
  local tag="$1"
  local ckpt="$2"
  local gpu="$3"
  local shard_idx="$4"
  local scene_filter="navtest_hard_composite_p90_shard${shard_idx}"
  local exp_name="${OUT_BASE}/${tag}/${scene_filter}"
  local out_dir="${NAVSIM_EXP_ROOT}/${exp_name}"

  if find "${out_dir}" -maxdepth 1 -name '*.csv' ! -name 'partial_*.csv' -type f 2>/dev/null | grep -q .; then
    echo "[SKIP] ${tag} ${scene_filter}: final CSV already exists in ${out_dir}"
    return 0
  fi

  echo "[RUN] ${tag} ${scene_filter} gpu=${gpu}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${SCRIPT}" \
    agent=WoTE_agent \
    "agent.checkpoint_path='${ckpt}'" \
    "experiment_name=${exp_name}" \
    "split=${SPLIT}" \
    "scene_filter=${scene_filter}" \
    "worker=${WORKER}" \
    "max_number_of_workers=1" \
    "+stream_worker_csv=true" \
    "evaluate_all_trajectories=false" \
    verbose=true \
    2>&1 | tee "${out_dir}/run.log"
}

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
  run_shard "epoch19_step21280" "${CKPT_NEW}" "${shard_idx}" "${shard_idx}" &
  run_shard "epoch29_step19950" "${CKPT_BASE}" "$((shard_idx + NUM_SHARDS))" "${shard_idx}" &
done

wait
