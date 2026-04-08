#!/usr/bin/env bash
set -euo pipefail

# Evaluate a WoTE checkpoint on the S01-S10 simulator stress matrix.
#
# Key features:
# - Explicit online post processing (yaw/speed scale affects x/y): +simulator.post_params.apply_mode=online
# - Controller-conditioned world model fusion via attention:
#     ++agent.config.controller_world_model_fusion=attn
#
# Required:
#   export OPENSCENE_DATA_ROOT=/path/to/dataset
#
# Typical usage:
#   OPENSCENE_DATA_ROOT=/mnt/data/navsim_workspace/dataset \
#   NAVSIM_EXP_ROOT=/mnt/data/navsim_workspace/exp \
#   CUDA_VISIBLE_DEVICES=0 \
#   CKPT_MAIN=/path/to/new.ckpt \
#   bash tool/evaluate/evaluate_ckpt_attn_s01_s10_onlinepost.sh

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
GEN_ANCHOR_SCRIPT=${GEN_ANCHOR_SCRIPT:-"${ROOT}/tool/evaluate/generate_exec_anchor_for_sim_style.py"}

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

# Runtime env (make sure local packages are importable without pip install -e)
export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}

# Where to write evaluation outputs (Hydra uses ${NAVSIM_EXP_ROOT})
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}/trainingResult"}

# Dataset root
: "${OPENSCENE_DATA_ROOT:?OPENSCENE_DATA_ROOT is required (dataset root).}"
export OPENSCENE_DATA_ROOT

SPLIT=${SPLIT:-"test"}
SCENE_FILTER=${SCENE_FILTER:-"navtest"}

# Worker
EVAL_WORKER=${EVAL_WORKER:-"ray_distributed_no_torch"}  # ray_distributed_no_torch | single_machine_thread_pool | sequential
EVAL_CPU_THREADS=${EVAL_CPU_THREADS:-"1"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-"${EVAL_CPU_THREADS}"}

# Controller-conditioning knobs
CTRL_FEATURE_MODE=${CTRL_FEATURE_MODE:-"full"}   # full | lateral_only
CTRL_POOLING=${CTRL_POOLING:-"attn"}            # attn | mean
WM_FUSION=${WM_FUSION:-"attn"}                  # attn | add
WM_INJECT_TARGET=${WM_INJECT_TARGET:-"all"}     # all | ego

# Checkpoint to evaluate
CKPT_MAIN=${CKPT_MAIN:-"/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260219_223113/epoch=39-step=106400.ckpt"}
if [[ -z "${CKPT_MAIN}" ]]; then
  echo "[ERR] CKPT_MAIN is required. Example: CKPT_MAIN=/path/to/epoch=xx-step=yy.ckpt" >&2
  exit 1
fi
if [[ ! -f "${CKPT_MAIN}" ]]; then
  echo "[ERR] Missing checkpoint: ${CKPT_MAIN}" >&2
  exit 1
fi

CKPT_DIR_TAG=${CKPT_DIR_TAG:-"$(basename "$(dirname "${CKPT_MAIN}")")"}
CKPT_FILE_TAG=${CKPT_FILE_TAG:-"$(basename "${CKPT_MAIN}" .ckpt)"}
CKPT_TAG=${CKPT_TAG:-"${CKPT_DIR_TAG}/${CKPT_FILE_TAG}"}
CKPT_TAG_SAFE=${CKPT_TAG_SAFE:-"${CKPT_TAG//\//_}"}

# Experiment output root (under ${NAVSIM_EXP_ROOT})
EXP_ROOT=${EXP_ROOT:-"eval/WoTE/${CKPT_TAG_SAFE}_wmfusion-${WM_FUSION}_s01_s10/${SPLIT}"}

# Paths used by WoTE agent
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}
SIM_REWARD_DICT=${SIM_REWARD_DICT:-"${ROOT}/extra_data/planning_vb/formatted_pdm_score_256.npy"}

CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_RP_CKPT=${CTRL_RP_CKPT:-"${ROOT}/ControllerExp/generated/controller_response_predictor.pt"}

# Optional: force a fixed controller style from the bundle during eval.
# Usage: CTRL_STYLE_IDX=0 bash tool/evaluate/evaluate_ckpt_attn_s01_s10_onlinepost.sh
if [[ -n "${CTRL_STYLE_IDX:-}" ]]; then
  export WOTE_CTRL_STYLE_IDX="${CTRL_STYLE_IDX}"
  export WOTE_CTRL_STYLE_DEBUG=${WOTE_CTRL_STYLE_DEBUG:-"0"}
  echo "[INFO] Forcing WOTE_CTRL_STYLE_IDX=${WOTE_CTRL_STYLE_IDX}"

  # Also force simulator to use the same bundle style (tracker/post params) for matched evaluation.
  # Bundle path is the WoTE controller exec path when using a .npz bundle.
  CTRL_EXEC_BUNDLE=${CTRL_EXEC_BUNDLE:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}
  export PDM_SIM_BUNDLE_PATH="${CTRL_EXEC_BUNDLE}"
  export PDM_SIM_STYLE_IDX="${CTRL_STYLE_IDX}"
  export PDM_SIM_BUNDLE_APPLY=${PDM_SIM_BUNDLE_APPLY:-"all"}
  export PDM_SIM_BUNDLE_DEBUG=${PDM_SIM_BUNDLE_DEBUG:-"0"}
  echo "[INFO] Forcing PDM_SIM_STYLE_IDX=${PDM_SIM_STYLE_IDX} (bundle=${PDM_SIM_BUNDLE_PATH}) apply=${PDM_SIM_BUNDLE_APPLY}"
fi

# Whether to generate per-scenario exec anchors and inject controller into WM.
INJECT_CONTROLLER=${INJECT_CONTROLLER:-"1"}      # 1|0

run_one() {
  local tag="$1"; shift
  local tracker_style="$1"; shift
  local post_style="$1"; shift
  local heading_scale="$1"; shift
  local speed_scale="$1"; shift
  local heading_bias="$1"; shift
  local speed_bias="$1"; shift
  local noise_std="$1"; shift

  echo "[INFO] Run: ${tag} tracker=${tracker_style} post=${post_style} h=${heading_scale}/${heading_bias} s=${speed_scale}/${speed_bias} noise=${noise_std}"

  local exec_anchor_path="${NAVSIM_EXP_ROOT}/${EXP_ROOT}/${tag}/controller_exec_anchor.npy"

  if [[ "${INJECT_CONTROLLER}" == "1" ]]; then
    mkdir -p "$(dirname "${exec_anchor_path}")"
    if [[ ! -f "${exec_anchor_path}" || "${REGEN_ANCHOR:-0}" == "1" ]]; then
      echo "[INFO] Generating exec anchor -> ${exec_anchor_path}"
      FORCE_FLAG=()
      if [[ "${REGEN_ANCHOR:-0}" == "1" ]]; then
        FORCE_FLAG=(--force)
      fi
      "${PYTHON}" -u "${GEN_ANCHOR_SCRIPT}" \
        --ref "${CTRL_REF}" \
        --out "${exec_anchor_path}" \
        --tracker-style "${tracker_style}" \
        --post-style "${post_style}" \
        --apply-mode online \
        --heading-scale "${heading_scale}" \
        --heading-bias "${heading_bias}" \
        --speed-scale "${speed_scale}" \
        --speed-bias "${speed_bias}" \
        --noise-std "${noise_std}" \
        --seed "${ANCHOR_SEED:-42}" \
        "${FORCE_FLAG[@]}"
    else
      echo "[INFO] Reusing exec anchor: ${exec_anchor_path}"
    fi
  fi

  WORKER_EXTRA=()
  if [[ "${EVAL_WORKER}" == "single_machine_thread_pool" ]]; then
    WORKER_EXTRA+=("worker.max_workers=${EVAL_MAX_WORKERS:-32}")
  fi

  HYDRA_ARGS=(
    agent=WoTE_agent
    "agent.checkpoint_path='${CKPT_MAIN}'"
    "experiment_name='${EXP_ROOT}/${tag}'"
    "worker=${EVAL_WORKER}"
    "${WORKER_EXTRA[@]}"
    "+stream_worker_csv=true"
    "split='${SPLIT}'"
    "scene_filter='${SCENE_FILTER}'"
    "simulator.tracker_style='${tracker_style}'"
    "simulator.post_style='${post_style}'"
    "+simulator.post_params.apply_mode=online"
    "simulator.post_params.heading_scale=${heading_scale}"
    "simulator.post_params.heading_bias=${heading_bias}"
    "simulator.post_params.speed_scale=${speed_scale}"
    "simulator.post_params.speed_bias=${speed_bias}"
    "simulator.post_params.noise_std=${noise_std}"
    "++agent.config.cluster_file_path='${PLANNER_ANCHORS}'"
    "++agent.config.sim_reward_dict_path='${SIM_REWARD_DICT}'"
    "++agent.config.controller_ref_traj_path='${CTRL_REF}'"
    "++agent.config.controller_response_predictor_path='${CTRL_RP_CKPT}'"
    "++agent.config.controller_feature_mode='${CTRL_FEATURE_MODE}'"
    "++agent.config.controller_style_pooling='${CTRL_POOLING}'"
    "++agent.config.controller_condition_on_world_model=true"
    "++agent.config.controller_world_model_fusion='${WM_FUSION}'"
    "++agent.config.controller_world_model_inject_target='${WM_INJECT_TARGET}'"
    evaluate_all_trajectories=false
    verbose=true
  )

  if [[ "${INJECT_CONTROLLER}" == "1" ]]; then
    HYDRA_ARGS+=(
      "++agent.config.controller_exec_traj_path='${exec_anchor_path}'"
    )
  else
    HYDRA_ARGS+=(
      "++agent.config.controller_exec_traj_path='${CTRL_REF}'"
      "++agent.config.controller_condition_on_world_model=false"
    )
  fi

  "${PYTHON}" -u "${SCRIPT}" "${HYDRA_ARGS[@]}"
}

scenarios=(
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

echo "============================================================"
echo "[EVAL] ckpt=${CKPT_MAIN}"
echo "  split=${SPLIT} scene_filter=${SCENE_FILTER}"
echo "  out=${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"
echo "  INJECT_CONTROLLER=${INJECT_CONTROLLER} WM_FUSION=${WM_FUSION} WM_INJECT_TARGET=${WM_INJECT_TARGET}"
echo "============================================================"

for spec in "${scenarios[@]}"; do
  IFS='|' read -r tag tracker_style post_style heading_scale speed_scale heading_bias speed_bias noise_std <<<"${spec}"
  run_one "${tag}" "${tracker_style}" "${post_style}" "${heading_scale}" "${speed_scale}" "${heading_bias}" "${speed_bias}" "${noise_std}"
done

echo "[DONE] Outputs under: ${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"
