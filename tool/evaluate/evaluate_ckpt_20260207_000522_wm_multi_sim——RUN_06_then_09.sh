#!/usr/bin/env bash
set -euo pipefail

# Driver script: run New06 first, then New09 automatically.
#
# Typical usage:
#   OPENSCENE_DATA_ROOT=/path/to/dataset \
#   NAVSIM_EXP_ROOT=/path/to/exp \
#   SPLIT=test SCENE_FILTER=navtest CUDA_VISIBLE_DEVICES=0 \
#   CKPT_MAIN=/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260209_170436/epoch=39-step=42560.ckpt \
#   bash tool/evaluate/evaluate_ckpt_20260207_000522_wm_multi_sim——RUN_06_then_09.sh

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}

SCRIPT_06=${SCRIPT_06:-"${ROOT}/tool/evaluate/evaluate_ckpt_20260207_000522_wm_multi_sim——NEW.sh"}
SCRIPT_09=${SCRIPT_09:-"${ROOT}/tool/evaluate/evaluate_ckpt_20260207_000522_wm_multi_sim——NEW09.sh"}

if [[ ! -f "${SCRIPT_06}" ]]; then
  echo "[ERROR] Missing script: ${SCRIPT_06}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_09}" ]]; then
  echo "[ERROR] Missing script: ${SCRIPT_09}" >&2
  exit 1
fi

# Allow using one CKPT_MAIN for both runs, or override per-run.
CKPT_MAIN=${CKPT_MAIN:-"${ROOT}/trainingResult/ckpts_20260209_170436/epoch=39-step=42560.ckpt"}
CKPT_BASE=${CKPT_BASE:-"${ROOT}/trainingResult/epoch=29-step=19950.ckpt"}
CKPT_MAIN_06=${CKPT_MAIN_06:-"${CKPT_MAIN}"}
CKPT_MAIN_09=${CKPT_MAIN_09:-"${CKPT_MAIN}"}

EXP_ROOT_06=${EXP_ROOT_06:-"eval/WoTE/ckpt_compare_20260207_wm/multi_simNew06"}
EXP_ROOT_09=${EXP_ROOT_09:-"eval/WoTE/ckpt_compare_20260207_wm/multi_simNew09"}

WOTE_INJ_STRENGTH_06=${WOTE_INJ_STRENGTH_06:-"0.25"}
WOTE_INJ_STRENGTH_09=${WOTE_INJ_STRENGTH_09:-"0.25"}

echo "============================================================"
echo "[RUN] 06 then 09"
echo "  CKPT_MAIN_06=${CKPT_MAIN_06}"
echo "  CKPT_MAIN_09=${CKPT_MAIN_09}"
echo "  CKPT_BASE=${CKPT_BASE}"
echo "  EXP_ROOT_06=${EXP_ROOT_06}"
echo "  EXP_ROOT_09=${EXP_ROOT_09}"
echo "  inj06=${WOTE_INJ_STRENGTH_06} inj09=${WOTE_INJ_STRENGTH_09}"
echo "  EVAL_WORKER=${EVAL_WORKER:-<inherit>} EVAL_SINGLE_EVAL=${EVAL_SINGLE_EVAL:-<inherit>} EVAL_CPU_THREADS=${EVAL_CPU_THREADS:-<inherit>}"
echo "  EVAL_NICE=${EVAL_NICE:-<none>}"
echo "============================================================"

NICE_PREFIX=()
if [[ -n "${EVAL_NICE:-}" ]]; then
  NICE_PREFIX=(nice -n "${EVAL_NICE}")
fi

echo "[STEP] Running New06 (wm_strength=0.6) ..."
CKPT_MAIN="${CKPT_MAIN_06}" \
CKPT_BASE="${CKPT_BASE}" \
EXP_ROOT="${EXP_ROOT_06}" \
WOTE_WM_STRENGTH=0.6 \
WOTE_INJ_STRENGTH="${WOTE_INJ_STRENGTH_06}" \
"${NICE_PREFIX[@]}" bash "${SCRIPT_06}"

echo "[STEP] Running New09 (wm_strength=0.9) ..."
CKPT_MAIN="${CKPT_MAIN_09}" \
CKPT_BASE="${CKPT_BASE}" \
EXP_ROOT="${EXP_ROOT_09}" \
WOTE_WM_STRENGTH=0.9 \
WOTE_INJ_STRENGTH="${WOTE_INJ_STRENGTH_09}" \
"${NICE_PREFIX[@]}" bash "${SCRIPT_09}"

echo "[DONE] Both runs finished."
