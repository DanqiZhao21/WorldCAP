#!/usr/bin/env bash
set -euo pipefail

# Multi-simulator evaluation for WoTE checkpoint (offset injection sweep).
# This script runs ONE checkpoint under S01-S10 simulator styles.
#
# Typical usage:
#   OPENSCENE_DATA_ROOT=/path/to/dataset \
#   NAVSIM_EXP_ROOT=/path/to/exp \
#   SPLIT=test SCENE_FILTER=navtest CUDA_VISIBLE_DEVICES=0 \
#   CKPT_MAIN=/abs/path/to/epoch=xx.ckpt OFFSET_INJ_STRENGTH=0.3 \
#   EXP_ROOT=eval/WoTE/ckpt_compare_offset/multi_sim_offset03 \
#   bash tool/evaluate/evaluate_ckpt_offset_multi_sim.sh

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}

CKPT_MAIN=${CKPT_MAIN:-""}
if [[ -z "${CKPT_MAIN}" ]]; then
  echo "[ERROR] CKPT_MAIN is required" >&2
  exit 2
fi

OFFSET_INJ_STRENGTH=${OFFSET_INJ_STRENGTH:-"0.3"}
EXP_ROOT=${EXP_ROOT:-"eval/WoTE/ckpt_compare_offset/multi_sim_offset03"}

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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Worker (Ray by default)
EVAL_WORKER=${EVAL_WORKER:-"ray_distributed_no_torch"}    # ray_distributed_no_torch | single_machine_thread_pool | sequential
EVAL_MAX_WORKERS=${EVAL_MAX_WORKERS:-"8"}                 # only used for single_machine_thread_pool
EVAL_CPU_THREADS=${EVAL_CPU_THREADS:-"1"}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-"${EVAL_CPU_THREADS}"}

echo "[INFO] EVAL_WORKER=${EVAL_WORKER} EVAL_MAX_WORKERS=${EVAL_MAX_WORKERS} EVAL_CPU_THREADS=${EVAL_CPU_THREADS}"

# Runtime env
export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}

# Output root
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}/trainingResult"}

# Dataset root
: "${OPENSCENE_DATA_ROOT:?OPENSCENE_DATA_ROOT is required (dataset root).}"
export OPENSCENE_DATA_ROOT

SPLIT=${SPLIT:-"test"}
SCENE_FILTER=${SCENE_FILTER:-"navtest"}

# Inputs / paths
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}
SIM_REWARD_DICT=${SIM_REWARD_DICT:-"${ROOT}/extra_data/planning_vb/formatted_pdm_score_256.npy"}
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC=${CTRL_EXEC:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}
CTRL_RP_CKPT=${CTRL_RP_CKPT:-"${ROOT}/ControllerExp/generated/controller_response_predictor.pt"}

if [[ ! -f "${CKPT_MAIN}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT_MAIN}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "[ERROR] Missing script: ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" ]]; then
  echo "[ERROR] Missing logs dir: ${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" >&2
  exit 1
fi

run_one() {
  local tag="$1"; shift
  local tracker_style="$1"; shift
  local post_style="$1"; shift
  local heading_scale="$1"; shift
  local speed_scale="$1"; shift
  local heading_bias="$1"; shift
  local speed_bias="$1"; shift
  local noise_std="$1"; shift

  echo "[INFO] Run ${tag} tracker=${tracker_style} post=${post_style} h=${heading_scale}/${heading_bias} s=${speed_scale}/${speed_bias} noise=${noise_std}"

  WORKER_EXTRA=()
  if [[ "${EVAL_WORKER}" == "single_machine_thread_pool" ]]; then
    WORKER_EXTRA+=("worker.max_workers=${EVAL_MAX_WORKERS}")
  fi

  "${PYTHON}" -u "${SCRIPT}" \
    agent=WoTE_agent \
    "agent.checkpoint_path='${CKPT_MAIN}'" \
    "experiment_name=${EXP_ROOT}/${tag}" \
    "worker=${EVAL_WORKER}" \
    "${WORKER_EXTRA[@]}" \
    "+stream_worker_csv=true" \
    "split=${SPLIT}" \
    "scene_filter=${SCENE_FILTER}" \
    "simulator.tracker_style=${tracker_style}" \
    "simulator.post_style=${post_style}" \
    "simulator.post_params.heading_scale=${heading_scale}" \
    "simulator.post_params.heading_bias=${heading_bias}" \
    "simulator.post_params.speed_scale=${speed_scale}" \
    "simulator.post_params.speed_bias=${speed_bias}" \
    "simulator.post_params.noise_std=${noise_std}" \
    "++agent.config.cluster_file_path='${PLANNER_ANCHORS}'" \
    "++agent.config.sim_reward_dict_path='${SIM_REWARD_DICT}'" \
    "++agent.config.controller_ref_traj_path='${CTRL_REF}'" \
    "++agent.config.controller_exec_traj_path='${CTRL_EXEC}'" \
    "++agent.config.controller_response_predictor_path='${CTRL_RP_CKPT}'" \
    "++agent.config.controller_condition_on_offset=true" \
    "++agent.config.controller_offset_inject_strength=${OFFSET_INJ_STRENGTH}" \
    evaluate_all_trajectories=false \
    verbose=true \
    "$@"
}

run_matrix() {
  local scenarios=(
    "S07_unstable_none|unstable|none|1.0|1.0|0|0|0"
    "S08_yaw_scale_12|default|yaw_scale|1.2|1.0|0|0|0"
    "S09_speed_scale_08|default|speed_scale|1.0|0.8|0|0|0"
    "S10_noise_02|default|none|1.0|1.0|0|0|0.2"
    "S01_default_none|default|none|1.0|1.0|0|0|0"
    "S02_default_post1515|default|yaw_speed_extreme|1.5|1.5|0|0|0"
    "S03_aggressive_none|aggressive|none|1.0|1.0|0|0|0"
    "S04_safe_none|safe|none|1.0|1.0|0|0|0"
    "S05_sluggish_none|sluggish|none|1.0|1.0|0|0|0"
    "S06_high_jitter_none|high_jitter|none|1.0|1.0|0|0|0"
  )

  echo "[INFO] Scenario order (as executed):"
  local _spec _tag _t _p _hs _ss _hb _sb _ns
  for _spec in "${scenarios[@]}"; do
    IFS='|' read -r _tag _t _p _hs _ss _hb _sb _ns <<<"${_spec}"
    echo "  - ${_tag}"
  done

  local spec tag tracker_style post_style heading_scale speed_scale heading_bias speed_bias noise_std
  for spec in "${scenarios[@]}"; do
    IFS='|' read -r tag tracker_style post_style heading_scale speed_scale heading_bias speed_bias noise_std <<<"${spec}"
    run_one "${tag}" "${tracker_style}" "${post_style}" "${heading_scale}" "${speed_scale}" "${heading_bias}" "${speed_bias}" "${noise_std}"
  done
}

echo "============================================================"
echo "[EVAL] offset_inject_strength=${OFFSET_INJ_STRENGTH}"
echo "  ckpt=${CKPT_MAIN}"
echo "  split=${SPLIT} scene_filter=${SCENE_FILTER}"
echo "  out=${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"
echo "  OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "============================================================"

run_matrix

echo "[DONE] Outputs under: ${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"