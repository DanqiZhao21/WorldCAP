#!/usr/bin/env bash
set -euo pipefail

# Evaluate 3 WoTE checkpoints on S01-S10 simulator stress matrix.
# - NO base_epoch29 run
# - Runs 10 scenarios (S01..S10) for each ckpt
# - Outputs are written to 3 experiment roots (multi_sim_offset03/06/09)
#
# Required:
#   export OPENSCENE_DATA_ROOT=/path/to/dataset   # contains navsim_logs/<split>, sensor_blobs/<split>
#
# Run:
#   OPENSCENE_DATA_ROOT=/path/to/dataset \
#   NAVSIM_EXP_ROOT=/path/to/exp \
#   SPLIT=test SCENE_FILTER=navtest GPU=0 \
#   EVAL_PROCS=8 \
#   bash tool/evaluate/eval_3ckpts_s01_s10_no_base.sh
#
# Optional overrides:
#   SPLIT=test SCENE_FILTER=navtest CUDA_VISIBLE_DEVICES=0 NAVSIM_EXP_ROOT=/path/to/out \
#   bash tool/evaluate/eval_3ckpts_s01_s10_no_base.sh

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}

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
# By default, force CUDA_VISIBLE_DEVICES=GPU (default 0).
# If you want to keep an externally-set CUDA_VISIBLE_DEVICES, set: RESPECT_CUDA_VISIBLE_DEVICES=1
GPU=${GPU:-0}
RESPECT_CUDA_VISIBLE_DEVICES=${RESPECT_CUDA_VISIBLE_DEVICES:-0}
if [[ "${RESPECT_CUDA_VISIBLE_DEVICES}" == "1" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  :
else
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi
echo "[INFO] GPU=${GPU} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} (RESPECT_CUDA_VISIBLE_DEVICES=${RESPECT_CUDA_VISIBLE_DEVICES})"

# Runtime env
export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}

# Output root (Hydra uses ${NAVSIM_EXP_ROOT})
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}/trainingResult"}

# Dataset root
: "${OPENSCENE_DATA_ROOT:?OPENSCENE_DATA_ROOT is required (dataset root).}"
export OPENSCENE_DATA_ROOT

SPLIT=${SPLIT:-"test"}
SCENE_FILTER=${SCENE_FILTER:-"navtest"}

# Worker (Ray by default)
EVAL_WORKER=${EVAL_WORKER:-"ray_distributed_no_torch"}  # ray_distributed_no_torch | single_machine_thread_pool | sequential
EVAL_CPU_THREADS=${EVAL_CPU_THREADS:-"1"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-"${EVAL_CPU_THREADS}"}

# Record (training-time) loss weights for the offset branch.
# NOTE: these do NOT affect evaluation; they are printed for experiment bookkeeping.
TRAJ_OFFSET_W=${TRAJ_OFFSET_W:-"1.0"}
OFFSET_IM_W=${OFFSET_IM_W:-"0.5"}

# Fix controller world-model conditioning strength during evaluation.
WM_STRENGTH=${WM_STRENGTH:-"0.3"}

# Controller bundle paths (override the absolute defaults in WoTE_agent.yaml)
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC=${CTRL_EXEC:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}
CTRL_RP_CKPT=${CTRL_RP_CKPT:-"${ROOT}/ControllerExp/generated/controller_response_predictor.pt"}

# Optional: force a fixed controller style from the bundle during eval.
# Usage: CTRL_STYLE_IDX=0 bash tool/evaluate/eval_3ckpts_s01_s10_no_base.sh
if [[ -n "${CTRL_STYLE_IDX:-}" ]]; then
  export WOTE_CTRL_STYLE_IDX="${CTRL_STYLE_IDX}"
  export WOTE_CTRL_STYLE_DEBUG=${WOTE_CTRL_STYLE_DEBUG:-"0"}
  echo "[INFO] Forcing WOTE_CTRL_STYLE_IDX=${WOTE_CTRL_STYLE_IDX}"

  # Also force simulator to use the same bundle style (tracker/post params) for matched evaluation.
  export PDM_SIM_BUNDLE_PATH="${CTRL_EXEC}"
  export PDM_SIM_STYLE_IDX="${CTRL_STYLE_IDX}"
  export PDM_SIM_BUNDLE_APPLY=${PDM_SIM_BUNDLE_APPLY:-"all"}
  export PDM_SIM_BUNDLE_DEBUG=${PDM_SIM_BUNDLE_DEBUG:-"0"}
  echo "[INFO] Forcing PDM_SIM_STYLE_IDX=${PDM_SIM_STYLE_IDX} (bundle=${PDM_SIM_BUNDLE_PATH}) apply=${PDM_SIM_BUNDLE_APPLY}"
fi

# Planner anchors (trajectory vocab)
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}
SIM_REWARD_DICT=${SIM_REWARD_DICT:-"${ROOT}/extra_data/planning_vb/formatted_pdm_score_256.npy"}

# 3 checkpoints (main only)
CKPT_MAIN_03=${CKPT_MAIN_03:-"${ROOT}/trainingResult/ckpts_20260211_230908/epoch=39-step=53200.ckpt"}
CKPT_MAIN_06=${CKPT_MAIN_06:-"${ROOT}/trainingResult/ckpts_20260212_130016/epoch=39-step=53200.ckpt"}
CKPT_MAIN_09=${CKPT_MAIN_09:-"${ROOT}/trainingResult/ckpts_20260212_220508/epoch=39-step=53200.ckpt"}

OFFSET_INJ_STRENGTH_03=${OFFSET_INJ_STRENGTH_03:-"0.3"}
OFFSET_INJ_STRENGTH_06=${OFFSET_INJ_STRENGTH_06:-"0.6"}
OFFSET_INJ_STRENGTH_09=${OFFSET_INJ_STRENGTH_09:-"0.9"}

# Output roots (requested names)
EXP_ROOT_03=${EXP_ROOT_03:-"eval/WoTE/ckpt_compare_20260213_offset/multi_sim_offset03"}
EXP_ROOT_06=${EXP_ROOT_06:-"eval/WoTE/ckpt_compare_20260213_offset/multi_sim_offset06"}
EXP_ROOT_09=${EXP_ROOT_09:-"eval/WoTE/ckpt_compare_20260213_offset/multi_sim_offset09"}

for f in "${SCRIPT}" "${CKPT_MAIN_03}" "${CKPT_MAIN_06}" "${CKPT_MAIN_09}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERR] Missing file: ${f}" >&2
    exit 1
  fi
done

if [[ ! -d "${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" ]]; then
  echo "[ERR] Missing logs dir: ${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" >&2
  exit 1
fi

run_one() {
  local ckpt_path="$1"; shift
  local offset_strength="$1"; shift
  local exp_root="$1"; shift
  local scen_tag="$1"; shift
  local tracker_style="$1"; shift
  local post_style="$1"; shift
  local heading_scale="$1"; shift
  local speed_scale="$1"; shift
  local heading_bias="$1"; shift
  local speed_bias="$1"; shift
  local noise_std="$1"; shift

  echo "[INFO] ${scen_tag} off=${offset_strength} tracker=${tracker_style} post=${post_style} h=${heading_scale}/${heading_bias} s=${speed_scale}/${speed_bias} noise=${noise_std}"

  WORKER_EXTRA=()
  if [[ "${EVAL_WORKER}" == "single_machine_thread_pool" ]]; then
    # Optional: when not using Ray.
    WORKER_EXTRA+=("worker.max_workers=${EVAL_MAX_WORKERS:-32}")
  fi

  "${PYTHON}" -u "${SCRIPT}" \
    agent=WoTE_agent \
    "agent.checkpoint_path='${ckpt_path}'" \
    "experiment_name=${exp_root}/${scen_tag}" \
    "worker=${EVAL_WORKER}" \
    "${WORKER_EXTRA[@]}" \
    "+stream_worker_csv=true" \
    "split=${SPLIT}" \
    "scene_filter=${SCENE_FILTER}" \
    "simulator.tracker_style=${tracker_style}" \
    "simulator.post_style=${post_style}" \
    "+simulator.post_params.apply_mode=online" \
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
    "++agent.config.controller_condition_on_world_model=true" \
    "++agent.config.controller_world_model_strength=${WM_STRENGTH}" \
    "++agent.config.controller_condition_on_offset=true" \
    "++agent.config.controller_offset_inject_strength=${offset_strength}" \
    evaluate_all_trajectories=false \
    verbose=true
}

run_matrix() {
  local ckpt_path="$1"; shift
  local offset_strength="$1"; shift
  local exp_root="$1"; shift

  # Keep same ordering as existing multi_sim script for easier comparison.
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
  local _spec _tag
  for _spec in "${scenarios[@]}"; do
    IFS='|' read -r _tag _ <<<"${_spec}"
    echo "  - ${_tag}"
  done

  local spec tag tracker_style post_style heading_scale speed_scale heading_bias speed_bias noise_std
  for spec in "${scenarios[@]}"; do
    IFS='|' read -r tag tracker_style post_style heading_scale speed_scale heading_bias speed_bias noise_std <<<"${spec}"
    run_one "${ckpt_path}" "${offset_strength}" "${exp_root}" "${tag}" "${tracker_style}" "${post_style}" "${heading_scale}" "${speed_scale}" "${heading_bias}" "${speed_bias}" "${noise_std}"
  done
}

echo "============================================================"
echo "[EVAL] 3 ckpts x 10 scenarios (S01-S10), no base"
echo "  split=${SPLIT} scene_filter=${SCENE_FILTER}"
echo "  out03=${NAVSIM_EXP_ROOT}/${EXP_ROOT_03}/"
echo "  out06=${NAVSIM_EXP_ROOT}/${EXP_ROOT_06}/"
echo "  out09=${NAVSIM_EXP_ROOT}/${EXP_ROOT_09}/"
echo "  OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "  CTRL_REF=${CTRL_REF}"
echo "  CTRL_EXEC=${CTRL_EXEC}"
echo "  wm_strength=${WM_STRENGTH}"
echo "  (train-loss weights record only) traj_offset_w=${TRAJ_OFFSET_W} offset_im_w=${OFFSET_IM_W}"
echo "  EVAL_WORKER=${EVAL_WORKER}"
echo "============================================================"

run_matrix "${CKPT_MAIN_03}" "${OFFSET_INJ_STRENGTH_03}" "${EXP_ROOT_03}"
run_matrix "${CKPT_MAIN_06}" "${OFFSET_INJ_STRENGTH_06}" "${EXP_ROOT_06}"
run_matrix "${CKPT_MAIN_09}" "${OFFSET_INJ_STRENGTH_09}" "${EXP_ROOT_09}"

echo "[DONE] Outputs under:"
echo "  - ${NAVSIM_EXP_ROOT}/${EXP_ROOT_03}/"
echo "  - ${NAVSIM_EXP_ROOT}/${EXP_ROOT_06}/"
echo "  - ${NAVSIM_EXP_ROOT}/${EXP_ROOT_09}/"
