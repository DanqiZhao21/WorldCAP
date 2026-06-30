#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}
PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
SPLIT=${SPLIT:-test}
OUT_BASE=${OUT_BASE:-eval/hard_scene_subsets_20260513}
WORKER=${WORKER:-sequential}
PARALLEL_JOBS=${PARALLEL_JOBS:-4}
GPU_LIST=(${GPU_LIST:-0 1 2 3})

export PYTHONPATH="${ROOT}:${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/mnt/data/navsim_workspace/dataset}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/mnt/data/navsim_workspace/exp}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
# Batch scoring does not load maps unless +enable_eval_visualization=true.
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"${ROOT}/nuplan-devkit/nuplan/common/maps"}

CKPT_NEW=${CKPT_NEW:-"${ROOT}/trainingResult/ckpts_20260511_204702/epoch=19-step=21280.ckpt"}
CKPT_BASE=${CKPT_BASE:-"${ROOT}/trainingResult/epoch=29-step=19950.ckpt"}

SCENE_FILTERS=(
  navtest_hard_curved_p90
  navtest_hard_curved_p95
  navtest_hard_dynamic_p90
  navtest_hard_fast_curve_p90
  navtest_hard_interaction_p90
  navtest_hard_composite_p90
)

run_eval() {
  local tag="$1"
  local ckpt="$2"
  local scene_filter="$3"
  local exp_name="${OUT_BASE}/${tag}/${scene_filter}"
  local out_dir="${NAVSIM_EXP_ROOT}/${exp_name}"

  if find "${out_dir}" -maxdepth 1 -name '*.csv' ! -name 'partial_*.csv' -type f 2>/dev/null | grep -q .; then
    echo "[SKIP] ${tag} ${scene_filter}: final CSV already exists in ${out_dir}"
    return 0
  fi

  local gpu="${4:-0}"
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

main() {
  local task_idx=0
  for scene_filter in "${SCENE_FILTERS[@]}"; do
    for spec in "epoch19_step21280:${CKPT_NEW}" "epoch29_step19950:${CKPT_BASE}"; do
      local tag="${spec%%:*}"
      local ckpt="${spec#*:}"
      local gpu="${GPU_LIST[$((task_idx % ${#GPU_LIST[@]}))]}"
      run_eval "${tag}" "${ckpt}" "${scene_filter}" "${gpu}" &
      task_idx=$((task_idx + 1))
      if (( task_idx % PARALLEL_JOBS == 0 )); then
        wait
      fi
    done
  done
  wait
}

main "$@"
