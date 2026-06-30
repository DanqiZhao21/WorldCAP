#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}
PYTHON=${PYTHON:-/home/zhaodanqi/anaconda3/envs/wotenewnew/bin/python}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${ROOT}/tool/training/0526train_wote_candidate_aligned_offset_wm_4gpu_32ep.sh"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
EXP_ROOT=${NAVSIM_EXP_ROOT:-/mnt/data/navsim_workspace/exp}

RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}
LOG_DIR=${LOG_DIR:-"${ROOT}/trainingResult/logs/candidate_aligned_${RUN_TS}"}
mkdir -p "${LOG_DIR}"

export RUN_TS
export NAVSIM_EXP_ROOT="${EXP_ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/navsim:${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-online}
export WOTE_WANDB_UPLOAD_CKPT=${WOTE_WANDB_UPLOAD_CKPT:-0}

source "${ROOT}/tool/common/worldcap_newctrl_paths.sh"

BASE_CKPT=${BASE_CKPT:-"${ROOT}/trainingResult/ckpts_20260514_213849+HERM/epoch=31-step=42560.ckpt"}
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}
CTRL_REF=${CTRL_REF:-"${WORLDCAP_CTRL_REF_1024}"}
CTRL_EXEC=${CTRL_EXEC:-"${WORLDCAP_CTRL_EXEC_1024}"}
OUT_BASE=${OUT_BASE:-"eval/candidate_aligned_offset_wm_${RUN_TS}"}
SPLIT=${SPLIT:-test}
GPU=${GPU:-0}
EVAL_WORKER=${EVAL_WORKER:-ray_distributed_no_torch}
EVAL_CPU_THREADS=${EVAL_CPU_THREADS:-32}
SCENE_FILTERS_STR=${SCENE_FILTERS_STR:-"navtest navtest_hard_composite_p90"}

exec > >(tee -a "${LOG_DIR}/train_then_eval.log") 2>&1

echo "[$(date '+%F %T')] candidate-aligned train-then-eval started"
echo "[$(date '+%F %T')] RUN_TS=${RUN_TS}"
echo "[$(date '+%F %T')] LOG_DIR=${LOG_DIR}"

for path in "${TRAIN_SCRIPT}" "${BASE_CKPT}" "${PLANNER_ANCHORS}" "${CTRL_REF}" "${CTRL_EXEC}"; do
  [[ -e "${path}" ]] || { echo "[ERR] missing ${path}" >&2; exit 1; }
done

echo "[$(date '+%F %T')] starting training: ${TRAIN_SCRIPT}"
GPU_IDS=${GPU_IDS:-0,1,2,3} WANDB_MODE="${WANDB_MODE}" bash "${TRAIN_SCRIPT}"

CKPT_DIR=$(ls -td "${ROOT}"/trainingResult/ckpts_* 2>/dev/null | head -n 1 || true)
if [[ -z "${CKPT_DIR}" || ! -d "${CKPT_DIR}" ]]; then
  echo "[ERR] no checkpoint dir found under ${ROOT}/trainingResult" >&2
  exit 1
fi

CUR_CKPT=$(ls -1 "${CKPT_DIR}"/epoch=31-step=*.ckpt 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -z "${CUR_CKPT}" ]]; then
  CUR_CKPT="${CKPT_DIR}/last.ckpt"
fi
if [[ ! -f "${CUR_CKPT}" ]]; then
  echo "[ERR] current checkpoint not found: ${CUR_CKPT}" >&2
  exit 1
fi

echo "[$(date '+%F %T')] training finished"
echo "[$(date '+%F %T')] CKPT_DIR=${CKPT_DIR}"
echo "[$(date '+%F %T')] CUR_CKPT=${CUR_CKPT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/mnt/data/navsim_workspace/dataset}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"${ROOT}/nuplan-devkit/nuplan/common/maps"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

run_eval() {
  local tag="$1"
  local ckpt="$2"
  local scene_filter="$3"
  local exp_name="${OUT_BASE}/${tag}/${scene_filter}"
  local out_dir="${EXP_ROOT}/${exp_name}"

  if find "${out_dir}" -maxdepth 1 -name '*.csv' ! -name 'partial_*.csv' -type f 2>/dev/null | grep -q .; then
    echo "[SKIP] ${tag} ${scene_filter}: final CSV already exists"
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] RUN eval tag=${tag} scene=${scene_filter}"
  "${PYTHON}" -u "${EVAL_SCRIPT}" \
    agent=WoTE_agent \
    "agent.checkpoint_path='${ckpt}'" \
    "experiment_name=${exp_name}" \
    "split=${SPLIT}" \
    "scene_filter=${scene_filter}" \
    "worker=${EVAL_WORKER}" \
    "worker.threads_per_node=${EVAL_CPU_THREADS}" \
    "+stream_worker_csv=true" \
    "evaluate_all_trajectories=false" \
    "++agent.config.cluster_file_path='${PLANNER_ANCHORS}'" \
    "++agent.config.controller_ref_traj_path='${CTRL_REF}'" \
    "++agent.config.controller_exec_traj_path='${CTRL_EXEC}'" \
    "++agent.config.herm_enable=false" \
    "++agent.config.herm_apply_in_eval=false" \
    "++agent.config.use_offset_candidates_in_train=true" \
    "++agent.config.detach_offset_candidates_in_train=true" \
    "++agent.config.use_scored_candidates_for_fut_bev_target=true" \
    "++agent.config.use_scored_candidates_for_im_loss=true" \
    "++agent.config.controller_condition_on_world_model=true" \
    "++agent.config.controller_world_model_fusion=attn_film" \
    "++agent.config.controller_world_model_inject_target=all" \
    "++agent.config.controller_feature_mode=full" \
    "++agent.config.controller_style_pooling=attn" \
    "++agent.config.use_agent_loss=false" \
    "++agent.config.use_map_loss=true" \
    "++agent.config.bev_semantic_weight=0.5" \
    "++agent.config.fut_bev_semantic_weight=1.0" \
    "++agent.config.traj_offset_loss_weight=0.2" \
    "++agent.config.offset_im_reward_weight=0.1" \
    verbose=true \
    2>&1 | tee "${out_dir}/run.log"
}

for scene in ${SCENE_FILTERS_STR}; do
  run_eval base_epoch31_42560 "${BASE_CKPT}" "${scene}"
  run_eval cur_candidate_aligned "${CUR_CKPT}" "${scene}"
done

SUMMARY_CSV="${LOG_DIR}/summary_metrics.csv"
SUMMARY_MD="${LOG_DIR}/summary_metrics.md"
echo "subset,tag,score,no_at_fault_collisions,drivable_area_compliance,driving_direction_compliance,ego_progress,time_to_collision_within_bound,comfort" > "${SUMMARY_CSV}"
for scene in ${SCENE_FILTERS_STR}; do
  for tag in base_epoch31_42560 cur_candidate_aligned; do
    dir="${EXP_ROOT}/${OUT_BASE}/${tag}/${scene}"
    csv=$(find "${dir}" -maxdepth 1 -name '*.csv' ! -name 'partial_*.csv' -type f | sort | tail -n 1 || true)
    [[ -n "${csv}" ]] || continue
    "${PYTHON}" - "${scene}" "${tag}" "${csv}" >> "${SUMMARY_CSV}" <<'PY'
import csv, sys
scene, tag, path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    avg = None
    for row in csv.DictReader(f):
        if row.get("token") == "average":
            avg = row
    if avg is None:
        raise SystemExit(0)
keys = ["score", "no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]
print(",".join([scene, tag] + [avg.get(k, "") for k in keys]))
PY
  done
done

"${PYTHON}" - "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import pandas as pd, sys
inp, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(inp)
if df.empty:
    print("no rows")
    raise SystemExit(0)
pivot = df.pivot(index="subset", columns="tag", values="score")
if {"cur_candidate_aligned", "base_epoch31_42560"}.issubset(pivot.columns):
    delta = (pivot["cur_candidate_aligned"] - pivot["base_epoch31_42560"]).rename("delta_cur_minus_base")
    res = pd.concat([pivot, delta], axis=1).reset_index()
else:
    res = pivot.reset_index()
with open(out, "w") as f:
    f.write(res.to_markdown(index=False, floatfmt=".4f"))
print(res.to_markdown(index=False, floatfmt=".4f"))
PY

echo "[$(date '+%F %T')] all done"
echo "[$(date '+%F %T')] summary csv: ${SUMMARY_CSV}"
echo "[$(date '+%F %T')] summary md: ${SUMMARY_MD}"
