#!/usr/bin/env bash
set -euo pipefail

# Multi-simulator evaluation for WoTE (controller-conditioned WM) checkpoint.
# This script runs the same ckpt under several simulator tracker/post styles.
#
# You can override at runtime, e.g.:
#   OPENSCENE_DATA_ROOT=/mnt/data/navsim_workspace/dataset \
#   NAVSIM_EXP_ROOT=/mnt/data/navsim_workspace/exp \
#   SPLIT=test SCENE_FILTER=navtest CUDA_VISIBLE_DEVICES=0 \
#   bash tool/evaluate/evaluate_ckpt_20260207_000522_wm_multi_sim.sh

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
GEN_ANCHOR_SCRIPT=${GEN_ANCHOR_SCRIPT:-"${ROOT}/tool/evaluate/generate_exec_anchor_for_sim_style.py"}

# Compare two checkpoints:
# - CKPT_MAIN: your latest trained ckpt
# - CKPT_BASE: original/base ckpt for comparison
CKPT_MAIN=${CKPT_MAIN:-"${ROOT}/trainingResult/ckpts_20260209_105105/epoch=39-step=42560.ckpt"}
CKPT_BASE=${CKPT_BASE:-"${ROOT}/trainingResult/epoch=29-step=19950.ckpt"}

# Controller -> WM knobs (New06 defaults)
# You can override at runtime, e.g.:
#   WOTE_WM_STRENGTH=0.6 WOTE_INJ_STRENGTH=0.25 bash tool/evaluate/evaluate_ckpt_20260207_000522_wm_multi_sim——NEW.sh
WOTE_WM_STRENGTH=${WOTE_WM_STRENGTH:-"0.6"}
WOTE_INJ_STRENGTH=${WOTE_INJ_STRENGTH:-"0.25"}

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

# --------------------------------------
# Concurrency / resource limits (tuning)
# --------------------------------------
# By default, evaluation can spawn many CPU threads (ray/threadpool + BLAS).
# These knobs help avoid CPU/memory spikes on shared servers.
EVAL_WORKER=${EVAL_WORKER:-"sequential"}          # sequential | single_machine_thread_pool | ray_distributed_no_torch
EVAL_SINGLE_EVAL=${EVAL_SINGLE_EVAL:-"true"}      # true -> bypass worker_map parallelism
EVAL_CPU_THREADS=${EVAL_CPU_THREADS:-"1"}         # limit BLAS/OMP threads

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-"${EVAL_CPU_THREADS}"}

echo "[INFO] EVAL_WORKER=${EVAL_WORKER} EVAL_SINGLE_EVAL=${EVAL_SINGLE_EVAL} EVAL_CPU_THREADS=${EVAL_CPU_THREADS}"
echo "[INFO] OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"

# Runtime env (make sure local packages are importable without pip install -e)
export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}

# Where to write evaluation outputs (Hydra uses ${NAVSIM_EXP_ROOT})
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}/trainingResult"}

# Dataset split to evaluate (folder under ${OPENSCENE_DATA_ROOT}/navsim_logs/<split>)
SPLIT=${SPLIT:-"test"}
# Scene filter config name (Hydra config). For navtest benchmark use: navtest
SCENE_FILTER=${SCENE_FILTER:-"navtest"}

# Optional: fix controller style index for deterministic eval
export WOTE_CTRL_STYLE_IDX=${WOTE_CTRL_STYLE_IDX:-"0"}

# Dataset root
: "${OPENSCENE_DATA_ROOT:="${ROOT}/extra_data"}"
export OPENSCENE_DATA_ROOT

for _ckpt in "${CKPT_MAIN}" "${CKPT_BASE}"; do
  if [[ ! -f "${_ckpt}" ]]; then
    echo "[ERROR] Checkpoint not found: ${_ckpt}" >&2
    exit 1
  fi
done

if [[ ! -d "${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" ]]; then
  echo "[ERROR] Missing logs dir: ${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" >&2
  echo "        Tip: set OPENSCENE_DATA_ROOT=/path/to/dataset and SPLIT=test (or mini/train/val)." >&2
  exit 1
fi

CTRL_NPZ=${CTRL_NPZ:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}
if [[ ! -f "${CTRL_NPZ}" ]]; then
  echo "[WARN] controller_styles.npz missing at ${CTRL_NPZ} (controller exec sampling may fail)" >&2
fi

# Controller reference anchors used to compute style embedding (phi)
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
if [[ ! -f "${CTRL_REF}" ]]; then
  echo "[WARN] controller_ref_traj_path missing at ${CTRL_REF} (controller embedding may fail)" >&2
fi

if [[ ! -f "${GEN_ANCHOR_SCRIPT}" ]]; then
  echo "[ERROR] Missing generator script: ${GEN_ANCHOR_SCRIPT}" >&2
  exit 1
fi

# Output root (under ${NAVSIM_EXP_ROOT})
EXP_ROOT=${EXP_ROOT:-"eval/WoTE/ckpt_compare_20260207_wm/multi_simNew06"}

echo "============================================================"
echo "[EVAL] ckpt_main=${CKPT_MAIN}"
echo "      ckpt_base=${CKPT_BASE}"
echo "      split=${SPLIT}"
echo "      scene_filter=${SCENE_FILTER}"
echo "      out=${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"
echo "      OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "      CTRL_REF=${CTRL_REF}"
echo "      CTRL_NPZ=${CTRL_NPZ}"
echo "      WOTE_WM_STRENGTH=${WOTE_WM_STRENGTH}"
echo "      WOTE_INJ_STRENGTH=${WOTE_INJ_STRENGTH}"
echo "============================================================"

run_one() {
  local ckpt_tag="$1"; shift
  local ckpt_path="$1"; shift
  local tag="$1"; shift
  local tracker_style="$1"; shift
  local post_style="$1"; shift
  local heading_scale="$1"; shift
  local speed_scale="$1"; shift
  local heading_bias="$1"; shift
  local speed_bias="$1"; shift
  local noise_std="$1"; shift

  # For main ckpt we generate exec anchor per simulator style and save it into the tag folder.
  # For base ckpt we do NOT inject controller and do NOT generate exec anchors.
  local exec_anchor_path="${NAVSIM_EXP_ROOT}/${EXP_ROOT}/${ckpt_tag}/${tag}/controller_exec_anchor.npy"
  local inject_controller="1"
  if [[ "${ckpt_tag}" == "base_epoch29" ]]; then
    inject_controller="0"
  fi

  echo "[INFO] Run: ckpt=${ckpt_tag} ${tag} tracker=${tracker_style} post=${post_style} h=${heading_scale}/${heading_bias} s=${speed_scale}/${speed_bias} noise=${noise_std}"

  if [[ "${inject_controller}" == "1" ]]; then
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

  HYDRA_ARGS=(
    agent=WoTE_agent
    "agent.checkpoint_path='${ckpt_path}'"
    "experiment_name=${EXP_ROOT}/${ckpt_tag}/${tag}"
    "worker=${EVAL_WORKER}"
    "+single_eval=${EVAL_SINGLE_EVAL}"
    "split=${SPLIT}"
    "scene_filter=${SCENE_FILTER}"
    "simulator.tracker_style=${tracker_style}"
    "simulator.post_style=${post_style}"
    "simulator.post_params.heading_scale=${heading_scale}"
    "simulator.post_params.heading_bias=${heading_bias}"
    "simulator.post_params.speed_scale=${speed_scale}"
    "simulator.post_params.speed_bias=${speed_bias}"
    "simulator.post_params.noise_std=${noise_std}"
    agent.config.controller_feature_mode=full
    "agent.config.controller_ref_traj_path='${CTRL_REF}'"
    evaluate_all_trajectories=false
    verbose=true
  )

  if [[ "${inject_controller}" == "1" ]]; then
    HYDRA_ARGS+=(
      "agent.config.controller_exec_traj_path='${exec_anchor_path}'"
      agent.config.controller_condition_on_world_model=true
      "agent.config.controller_world_model_strength=${WOTE_WM_STRENGTH}"
      agent.config.controller_injection_mode=film
      "agent.config.controller_injection_strength=${WOTE_INJ_STRENGTH}"
      agent.config.controller_condition_on_traj_feature=true
      agent.config.controller_condition_on_offset=true
      agent.config.controller_condition_on_reward_feature=true
    )
  else
    HYDRA_ARGS+=(
      "agent.config.controller_exec_traj_path='${CTRL_REF}'"
      agent.config.controller_condition_on_world_model=false
      agent.config.controller_world_model_strength=0.0
      agent.config.controller_injection_mode=none
      agent.config.controller_injection_strength=0.0
      agent.config.controller_condition_on_traj_feature=false
      agent.config.controller_condition_on_offset=false
      agent.config.controller_condition_on_reward_feature=false
    )
  fi

  "${PYTHON}" -u "${SCRIPT}" "${HYDRA_ARGS[@]}" "$@"
}

run_matrix_interleaved() {
  # 10 scenarios (5-10 requested): tracker styles + post styles + noise stress.
  # Run main then base for each scenario to avoid waiting for all main to finish.
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

  local spec tag tracker_style post_style heading_scale speed_scale heading_bias speed_bias noise_std
  for spec in "${scenarios[@]}"; do
    IFS='|' read -r tag tracker_style post_style heading_scale speed_scale heading_bias speed_bias noise_std <<<"${spec}"
    run_one "main" "${CKPT_MAIN}" "${tag}" "${tracker_style}" "${post_style}" "${heading_scale}" "${speed_scale}" "${heading_bias}" "${speed_bias}" "${noise_std}"
    run_one "base_epoch29" "${CKPT_BASE}" "${tag}" "${tracker_style}" "${post_style}" "${heading_scale}" "${speed_scale}" "${heading_bias}" "${speed_bias}" "${noise_std}"
  done
}

run_matrix_interleaved

echo "[DONE] Outputs under: ${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"

