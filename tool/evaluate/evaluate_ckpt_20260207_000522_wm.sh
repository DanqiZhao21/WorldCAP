#!/usr/bin/env bash
set -euo pipefail

# Evaluate WoTEAgent (controller-conditioned WM) on a given checkpoint
# Checkpoint: /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260207_000522/epoch=39-step=53200.ckpt

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
CKPT=${CKPT:-"${ROOT}/trainingResult/ckpts_20260207_000522/epoch=39-step=53200.ckpt"}

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

# Runtime env (make sure local packages are importable without pip install -e)
export PYTHONPATH="${ROOT}/navsim:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}

# Where to write evaluation outputs (Hydra uses ${NAVSIM_EXP_ROOT})
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}/trainingResult"}

# Dataset split to evaluate (folder under ${OPENSCENE_DATA_ROOT}/navsim_logs/<split>)
# Common: train|val|test|mini
SPLIT=${SPLIT:-"test"}

# Scene filter config (Hydra config name). For navtest benchmark use: navtest
SCENE_FILTER=${SCENE_FILTER:-"navtest"}

# Fix controller style index for deterministic eval (optional)
export WOTE_CTRL_STYLE_IDX=${WOTE_CTRL_STYLE_IDX:-"0"}

# Optional: if OPENSCENE_DATA_ROOT isn't set, try a local default
: "${OPENSCENE_DATA_ROOT:="${ROOT}/extra_data"}"
export OPENSCENE_DATA_ROOT

if [[ ! -d "${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" ]]; then
  echo "[ERROR] Missing logs dir: ${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" >&2
  echo "        Tip: set OPENSCENE_DATA_ROOT=/path/to/dataset and SPLIT=test (or mini/train/val)." >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] Checkpoint not found: $CKPT" >&2
  exit 1
fi

CTRL_NPZ="${ROOT}/ControllerExp/generated/controller_styles.npz"
if [[ ! -f "$CTRL_NPZ" ]]; then
  echo "[WARN] controller_styles.npz missing at $CTRL_NPZ (style sampling is disabled in eval; proceeding)"
fi

# Output root (under ${NAVSIM_EXP_ROOT})
EXP_ROOT=${EXP_ROOT:-"eval/WoTE/ckpt_20260207_000522/wm_controller"}

echo "============================================================"
echo "[EVAL] ckpt=${CKPT}"
echo "      split=${SPLIT}"
echo "      scene_filter=${SCENE_FILTER}"
echo "      out=${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"
echo "      OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "      PYTHONPATH=${PYTHONPATH}"
echo "============================================================"

run_eval() {
  local tracker_style="$1"    # default|aggressive|sluggish|drunk|safe|high_jitter|highhigh_jitter|unstable
  local post_style="$2"       # none|yaw_scale|speed_scale|yaw_speed_extreme|aggressive_post
  local exp_name_suffix="$3"  # label for experiment_name
  local heading_scale="${4:-1.0}"
  local speed_scale="${5:-1.0}"

  local EXP_NAME="${EXP_ROOT}/sim_${exp_name_suffix}"

  echo "[INFO] Running eval: tracker=$tracker_style post=$post_style split=$SPLIT exp=$EXP_NAME"
  "${PYTHON}" -u "${SCRIPT}" \
    experiment_name="$EXP_NAME" \
    split="$SPLIT" \
    scene_filter="$SCENE_FILTER" \
    agent=WoTE_agent \
    "agent.checkpoint_path='${CKPT}'" \
    agent.config.controller_feature_mode=full \
    "agent.config.controller_exec_traj_path='${CTRL_NPZ}'" \
    simulator.tracker_style="$tracker_style" \
    simulator.post_style="$post_style" \
    simulator.post_params.heading_scale="$heading_scale" \
    simulator.post_params.heading_bias=0 \
    simulator.post_params.speed_scale="$speed_scale" \
    simulator.post_params.speed_bias=0 \
    simulator.post_params.noise_std=0 \
    evaluate_all_trajectories=false \
    verbose=true
}

# 1) default
run_eval default none default

# 2) 1515 (POSTstyle_1515): yaw_speed_extreme with 1.5 scaling
run_eval default yaw_speed_extreme 1515 1.5 1.5

# 3) aggressive
run_eval aggressive none aggressive

echo "[DONE] Outputs under: ${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"
