#!/usr/bin/env bash
# EVAL_CPU_THREADS=null EVAL_RAY_THREADS_PER_NODE=null bash tool/evaluate/eval_ckpts_20260225_valstyles.sh
set -euo pipefail

# Eval script for 3 checkpoints (baseline/default, attn, film03) using VAL styles from a controller bundle.
# Two modes:
#  1) Mix: sample a random VAL style each planning step (uniform), matched between agent+sim.
#  2) Fixed: run a fixed set of style indices (post-dyn 5 + tracker-level 5), each style per run.
#
# Outputs:
#   ${NAVSIM_EXP_ROOT}/eval/WoTE/ckpt_20260225/
#     mix/{default,attn,film03}/
#     0189/{default,attn,film03}/ ...
#     0002/{default,attn,film03}/ ...
#
# Required:
#   OPENSCENE_DATA_ROOT=/path/to/dataset
#
# Example:
#   OPENSCENE_DATA_ROOT=/data/navsim \
#   SPLIT=test SCENE_FILTER=navtest GPU=0 EVAL_WORKER=ray_distributed_no_torch EVAL_CPU_THREADS=1 \
#   bash tool/evaluate/eval_ckpts_20260225_valstyles.sh

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

# Output root
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${ROOT}/trainingResult"}

# Dataset root
: "${OPENSCENE_DATA_ROOT:?OPENSCENE_DATA_ROOT is required (dataset root).}"
export OPENSCENE_DATA_ROOT

SPLIT=${SPLIT:-"test"}
SCENE_FILTER=${SCENE_FILTER:-"navtest"}

# Worker
EVAL_WORKER=${EVAL_WORKER:-"ray_distributed_no_torch"}
# NOTE: Do NOT default to very high parallelism; ray may kill workers due to memory pressure
# (MemAvailable drops because many tasks load logs/maps simultaneously).
# Start with 8/16/32 and scale up if stable.
EVAL_CPU_THREADS=${EVAL_CPU_THREADS:-"64"}

# Set to an integer to cap ray parallelism per node.
# Set to 'null' to use the worker config default (null -> all available threads).
EVAL_RAY_THREADS_PER_NODE=${EVAL_RAY_THREADS_PER_NODE:-"${EVAL_CPU_THREADS}"}

# Optional Ray safety knobs (only effective when raylet starts).
# - Set RAY_STOP_FIRST=1 to stop any existing local ray before running.
RAY_STOP_FIRST=${RAY_STOP_FIRST:-0}

is_nullish() {
  local v="${1:-}"
  [[ -z "${v}" || "${v,,}" == "null" || "${v,,}" == "none" ]]
}

# Threading env vars (BLAS/NumPy). If EVAL_CPU_THREADS is null, do not force a cap.
if ! is_nullish "${EVAL_CPU_THREADS}"; then
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
  export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
  export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
  export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-"${EVAL_CPU_THREADS}"}
fi

# Controller bundle paths
CTRL_REF=${CTRL_REF:-"${ROOT}/ControllerExp/Anchors_Original_256_centered.npy"}
CTRL_EXEC=${CTRL_EXEC:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}
CTRL_RP_CKPT=${CTRL_RP_CKPT:-"${ROOT}/ControllerExp/generated/controller_response_predictor.pt"}

# Planner anchors
# Choose which anchor vocab to evaluate with.
# - If you explicitly export PLANNER_ANCHORS/SIM_REWARD_DICT, those win.
# - Otherwise, select by ANCHOR_SET.
ANCHOR_SET=${ANCHOR_SET:-"256"}  # 256 | 128 | 64

if [[ -z "${PLANNER_ANCHORS:-}" || -z "${SIM_REWARD_DICT:-}" ]]; then
  case "${ANCHOR_SET}" in
    256)
      PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}
      SIM_REWARD_DICT=${SIM_REWARD_DICT:-"${ROOT}/extra_data/planning_vb/formatted_pdm_score_256.npy"}
      ;;
    128)
      PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/Anchors_KMeans_128.npy"}
      SIM_REWARD_DICT=${SIM_REWARD_DICT:-"${ROOT}/extra_data/planning_vb/PDMScore_KMeans_128.npy"}
      ;;
    64)
      PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/Anchors_KMeans_64.npy"}
      SIM_REWARD_DICT=${SIM_REWARD_DICT:-"${ROOT}/extra_data/planning_vb/PDMScore_KMeans_64.npy"}
      ;;
    *)
      echo "[ERR] Unknown ANCHOR_SET='${ANCHOR_SET}'. Use 256|128|64, or export PLANNER_ANCHORS/SIM_REWARD_DICT explicitly." >&2
      exit 2
      ;;
  esac
fi

#FIXME:
# Checkpoints这些是主实验部分的akpt
CKPT_BASELINE=${CKPT_BASELINE:-"${ROOT}/trainingResult/epoch=29-step=19950.ckpt"}
CKPT_ATTN=${CKPT_ATTN:-"${ROOT}/trainingResult/ckpts_20260225_003446/epoch=39-step=53200.ckpt"}
CKPT_FILM03=${CKPT_FILM03:-"${ROOT}/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt"}

# Ablations (defaults reuse the closest baseline unless overridden explicitly)
# - attn_s0: runtime override only (inject controller token only at world-model step 0)
CKPT_ATTN_FIRSTSTEP=${CKPT_ATTN_FIRSTSTEP:-"${ROOT}/trainingResult/ckpts_20260301_113743/epoch=39-step=53200.ckpt"}
# - film06/film09: strength ablations; by default reuse film03 checkpoint unless you provide separate ckpts.
CKPT_FILM06=${CKPT_FILM06:-"${ROOT}/trainingResult/ckpts_20260301_164244/epoch=39-step=53200.ckpt"}
CKPT_FILM09=${CKPT_FILM09:-"${ROOT}/trainingResult/ckpts_20260301_221622/epoch=29-step=39900.ckpt"}



#FIXME:
# Output folder root (relative to NAVSIM_EXP_ROOT) # Default follows ANCHOR_SET to avoid accidentally overwriting results.
# OUT_ROOT=${OUT_ROOT:-"eval/WoTE/ckpt_20260228/AblationForDifferentAnchorsInReference/anchor${ANCHOR_SET}"}
OUT_ROOT=${OUT_ROOT:-"eval/WoTE/ckpt_20260302/attn_s0_film06_film09"}


# Fixed styles
POST_STYLE_IDXS=(0189 0195 0177 0159 0087)

# TRK_STYLE_IDXS=(0002 0025 0041 0057 0073)

TRK_STYLE_IDXS=(0000 )


# Film strength for eval (only used by fusion=film*)
FILM_WM_STRENGTH=${FILM_WM_STRENGTH:-"0.3"}
FILM_WM_STRENGTH_06=${FILM_WM_STRENGTH_06:-"0.6"}
FILM_WM_STRENGTH_09=${FILM_WM_STRENGTH_09:-"0.9"}

# Set to 1 to allow bundles without train/val split indices.
ALLOW_BUNDLE_NO_SPLIT=${ALLOW_BUNDLE_NO_SPLIT:-0}

preflight() {
  if [[ "${RAY_STOP_FIRST}" == "1" && "${EVAL_WORKER}" == ray_distributed* ]]; then
    if command -v ray >/dev/null 2>&1; then
      echo "[INFO] RAY_STOP_FIRST=1 -> ray stop --force"
      ray stop --force || true
    else
      echo "[WARN] RAY_STOP_FIRST=1 but 'ray' command not found" >&2
    fi
  fi

  local f
  for f in "${SCRIPT}" "${CKPT_BASELINE}" "${CKPT_ATTN}" "${CKPT_FILM03}" "${CKPT_ATTN_FIRSTSTEP}" "${CKPT_FILM06}" "${CKPT_FILM09}" "${CTRL_REF}" "${CTRL_EXEC}"; do
    if [[ ! -f "${f}" ]]; then
      echo "[ERR] Missing file: ${f}" >&2
      exit 1
    fi
  done
  if [[ ! -d "${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" ]]; then
    echo "[ERR] Missing logs dir: ${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" >&2
    exit 1
  fi

  echo "[INFO] CTRL_EXEC=${CTRL_EXEC}"
  echo "[INFO] ANCHOR_SET=${ANCHOR_SET}"
  echo "[INFO] PLANNER_ANCHORS=${PLANNER_ANCHORS}"
  echo "[INFO] SIM_REWARD_DICT=${SIM_REWARD_DICT}"
  local post_idxs_str trk_idxs_str
  post_idxs_str="${POST_STYLE_IDXS[*]:-}"
  trk_idxs_str="${TRK_STYLE_IDXS[*]:-}"
  CTRL_EXEC="${CTRL_EXEC}" \
  PLANNER_ANCHORS="${PLANNER_ANCHORS}" \
  SIM_REWARD_DICT="${SIM_REWARD_DICT}" \
  ALLOW_BUNDLE_NO_SPLIT="${ALLOW_BUNDLE_NO_SPLIT}" \
  POST_STYLE_IDXS_STR="${post_idxs_str}" \
  TRK_STYLE_IDXS_STR="${trk_idxs_str}" \
  "${PYTHON}" - <<'PY'
import os
import numpy as np

bundle = os.environ.get('CTRL_EXEC')
if not bundle:
    raise SystemExit('[ERR] CTRL_EXEC env not set (unexpected).')

d = np.load(bundle, allow_pickle=True)
keys = set(d.keys())

def num_styles(dd):
    if 'style_names' in dd:
        return int(dd['style_names'].shape[0])
    if 'lqr_params' in dd:
        return int(dd['lqr_params'].shape[0])
    if 'post_params' in dd:
        return int(dd['post_params'].shape[0])
    return 0

n = num_styles(d)
has_train = 'train_style_indices' in keys
has_val = 'val_style_indices' in keys

print(f"[INFO] bundle keys: {sorted(list(keys))}")
print(f"[INFO] bundle num_styles={n} has_train_split={has_train} has_val_split={has_val}")

need_split = os.environ.get('ALLOW_BUNDLE_NO_SPLIT', '0') != '1'
if need_split and (not has_train or not has_val):
    raise SystemExit(
        "[ERR] CTRL_EXEC bundle does not contain train_style_indices/val_style_indices. "
        "You are probably pointing at an older bundle like 'controller_styles copy.npz'. "
        "Set CTRL_EXEC=ControllerExp/generated/controller_styles.npz (the one with splits), "
        "or set ALLOW_BUNDLE_NO_SPLIT=1 to bypass."
    )

def parse_idx_list(s: str):
  s = (s or '').strip()
  if not s:
    return []
  out = []
  for tok in s.split():
    tok = tok.strip()
    if not tok:
      continue
    out.append(int(tok, 10))
  return out

post_s = os.environ.get('POST_STYLE_IDXS_STR', '')
trk_s = os.environ.get('TRK_STYLE_IDXS_STR', '')
fixed_idxs = parse_idx_list(post_s) + parse_idx_list(trk_s)
if fixed_idxs:
  mx = max(fixed_idxs)
  if n <= mx:
    raise SystemExit(
      f"[ERR] CTRL_EXEC num_styles={n} but fixed idx max={mx}. "
      "This would clamp indices and silently mismatch styles. "
      "Use the newer controller_styles.npz bundle (213 styles), or reduce fixed idx list."
    )
  print(f"[INFO] fixed styles: post=[{post_s}] trk=[{trk_s}]")
else:
  print('[INFO] fixed styles: <none> (skipping fixed idx range check)')

anchors = os.environ.get('PLANNER_ANCHORS', '')
score = os.environ.get('SIM_REWARD_DICT', '')
if anchors:
  a = np.load(anchors)
  print(f"[INFO] planner anchors shape={getattr(a, 'shape', None)} dtype={getattr(a, 'dtype', None)} path={anchors}")
if score:
  # Some reward dict .npy files may be saved as object arrays; allow_pickle keeps preflight robust.
  s = np.load(score, allow_pickle=True)
  print(f"[INFO] sim reward dict shape={getattr(s, 'shape', None)} dtype={getattr(s, 'dtype', None)} path={score}")
PY
}

# local var="$1"; shift → 获取函数的第一个参数赋给变量 var，然后 shift 把参数列表左移，剩下的参数依次成为 $1, $2,...。
run_eval() {
  local ckpt_path="$1"; shift
  local exp_name="$1"; shift
  local model_tag="$1"; shift  # default|attn|film03 (for logging)
  local ctrl_enable="$1"; shift # 0|1
  local wm_fusion="$1"; shift   # attn|film03
  local wm_strength="$1"; shift

  WORKER_EXTRA=()
  if [[ "${EVAL_WORKER}" == "single_machine_thread_pool" ]]; then
    WORKER_EXTRA+=("worker.max_workers=${EVAL_MAX_WORKERS:-32}")
  elif [[ "${EVAL_WORKER}" == ray_distributed* ]]; then
    # Prevent ray from using all CPU threads by default (threads_per_node=null).
    # This reduces memory pressure and avoids ray OOM killing workers.
    if ! is_nullish "${EVAL_RAY_THREADS_PER_NODE}"; then
      WORKER_EXTRA+=("worker.threads_per_node=${EVAL_RAY_THREADS_PER_NODE}")
    fi
  fi

  echo "============================================================"
  echo "[EVAL] ${model_tag} exp=${exp_name}"
  echo "  ckpt=${ckpt_path}"
  echo "  ctrl_enable=${ctrl_enable} wm_fusion=${wm_fusion} wm_strength=${wm_strength}"
  echo "  bundle=${CTRL_EXEC}"
  echo "============================================================"

  "${PYTHON}" -u "${SCRIPT}" \
    agent=WoTE_agent \
    "agent.checkpoint_path='${ckpt_path}'" \
    "experiment_name=${exp_name}" \
    "worker=${EVAL_WORKER}" \
    "${WORKER_EXTRA[@]}" \
    "+stream_worker_csv=true" \
    "split=${SPLIT}" \
    "scene_filter=${SCENE_FILTER}" \
    "+simulator.post_params.apply_mode=online" \
    "++agent.config.cluster_file_path='${PLANNER_ANCHORS}'" \
    "++agent.config.sim_reward_dict_path='${SIM_REWARD_DICT}'" \
    "++agent.config.controller_ref_traj_path='${CTRL_REF}'" \
    "++agent.config.controller_exec_traj_path='${CTRL_EXEC}'" \
    "++agent.config.controller_response_predictor_path='${CTRL_RP_CKPT}'" \
    "++agent.config.controller_condition_on_world_model=${ctrl_enable}" \
    "++agent.config.controller_world_model_fusion=${wm_fusion}" \
    "++agent.config.controller_world_model_strength=${wm_strength}" \
    verbose=true \
    evaluate_all_trajectories=false \
    "$@"
}

#FIXME: ---- Mode 1: Mix (random val style per step, matched agent+sim) ----

# run_mix() {
#   local model_tag="$1"; shift
#   local ckpt_path="$1"; shift
#   local ctrl_enable="$1"; shift
#   local wm_fusion="$1"; shift
#   local wm_strength="$1"; shift

#   # Enable eval-time sampling from VAL split
#   export WOTE_CTRL_STYLE_SPLIT=val
#   export WOTE_CTRL_EVAL_SAMPLE=1

#   # Simulator: sample from VAL split, and keep it synced via PDM_SIM_STYLE_IDX
#   export PDM_SIM_BUNDLE_PATH="${CTRL_EXEC}"
#   export PDM_SIM_BUNDLE_APPLY=all
#   export PDM_SIM_STYLE_SPLIT=val
#   export PDM_SIM_EVAL_SAMPLE=1
#   export PDM_SIM_BUNDLE_DEBUG=${PDM_SIM_BUNDLE_DEBUG:-0}

#   unset WOTE_CTRL_STYLE_IDX || true
#   unset PDM_SIM_STYLE_IDX || true

#   run_eval "${ckpt_path}" "${OUT_ROOT}/mix/${model_tag}" "${model_tag}" "${ctrl_enable}" "${wm_fusion}" "${wm_strength}"
# }

# ---- Mode 2: Fixed style idx (entire eval uses one simulator style) ----
run_fixed_style() {
  local style_idx="$1"; shift
  local model_tag="$1"; shift
  local ckpt_path="$1"; shift
  local ctrl_enable="$1"; shift
  local wm_fusion="$1"; shift
  local wm_strength="$1"; shift
  local wm_first_step_only="${1:-}"; shift || true

  export WOTE_CTRL_STYLE_SPLIT=val
  export WOTE_CTRL_EVAL_SAMPLE=0

  export WOTE_CTRL_STYLE_IDX="${style_idx#0}"  # allow leading zeros
  export PDM_SIM_BUNDLE_PATH="${CTRL_EXEC}"
  export PDM_SIM_BUNDLE_APPLY=all
  export PDM_SIM_EVAL_SAMPLE=0
  export PDM_SIM_STYLE_IDX="${style_idx#0}"
  export PDM_SIM_BUNDLE_DEBUG=${PDM_SIM_BUNDLE_DEBUG:-0}

  EXTRA_OVERRIDES=()
  if [[ -n "${wm_first_step_only}" ]]; then
    EXTRA_OVERRIDES+=("++agent.config.controller_world_model_inject_first_step_only=${wm_first_step_only}")
  fi

  run_eval "${ckpt_path}" "${OUT_ROOT}/${style_idx}/${model_tag}" "${model_tag}" "${ctrl_enable}" "${wm_fusion}" "${wm_strength}" "${EXTRA_OVERRIDES[@]}"
}

main() {
  preflight

  echo "============================================================"
  echo "[EVAL] ckpt_20260225"
  echo "  OUT_ROOT=${NAVSIM_EXP_ROOT}/${OUT_ROOT}"
  echo "  bundle=${CTRL_EXEC} (VAL sampling)"
  echo "  SPLIT=${SPLIT} SCENE_FILTER=${SCENE_FILTER}"
  echo "  worker=${EVAL_WORKER} cpu_threads=${EVAL_CPU_THREADS}"
  echo "============================================================"

  # Mix: default (baseline, ctrl off) -> attn -> film03
  # run_mix default "${CKPT_BASELINE}" 0 attn 0.0
  # run_mix attn "${CKPT_ATTN}" 1 attn 0.0
  # run_mix film03 "${CKPT_FILM03}" 1 film03 "${FILM_WM_STRENGTH}"

  # Fixed: post-dynamics 5 + tracker-level 5
  # for s in "${POST_STYLE_IDXS[@]}" "${TRK_STYLE_IDXS[@]}"; do
  for s in "${TRK_STYLE_IDXS[@]}"; do
    # Baseline (controller->WM off)
    # run_fixed_style "${s}" default "${CKPT_BASELINE}" 0 attn 0.0 false

    # # ATTn (controller->WM on), inject every WM step (historical)
    # run_fixed_style "${s}" attn "${CKPT_ATTN}" 1 attn 0.0 false

    # ATTn single-step injection (new runtime flag): only inject at WM step 0
    run_fixed_style "${s}" attn_s0 "${CKPT_ATTN_FIRSTSTEP}" 1 attn 0.0 true

    # FiLM ablations
    # run_fixed_style "${s}" film03 "${CKPT_FILM03}" 1 film03 "${FILM_WM_STRENGTH}" false
    run_fixed_style "${s}" film06 "${CKPT_FILM06}" 1 film03 "${FILM_WM_STRENGTH_06}" false
    run_fixed_style "${s}" film09 "${CKPT_FILM09}" 1 film03 "${FILM_WM_STRENGTH_09}" false
  done

  echo "[DONE] Outputs under: ${NAVSIM_EXP_ROOT}/${OUT_ROOT}/"
}

main "$@"
