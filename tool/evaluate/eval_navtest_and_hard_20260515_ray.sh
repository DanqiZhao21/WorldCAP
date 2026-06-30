#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}
source "${ROOT}/tool/common/worldcap_newctrl_paths.sh"

PYTHON=${PYTHON:-/home/zhaodanqi/anaconda3/envs/wotenewnew/bin/python}
SCRIPT=${SCRIPT:-"${ROOT}/navsim/planning/script/run_pdm_score_multiTraj.py"}
OUT_BASE=${OUT_BASE:-eval/navtest_and_hard_20260515_ray}
SPLIT=${SPLIT:-test}
GPU=${GPU:-0}
EVAL_WORKER=${EVAL_WORKER:-ray_distributed_no_torch}
EVAL_CPU_THREADS=${EVAL_CPU_THREADS:-32}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU}}
export PYTHONPATH="${ROOT}:${ROOT}/navsim:${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-/mnt/data/navsim_workspace/dataset}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/mnt/data/navsim_workspace/exp}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"${ROOT}/nuplan-devkit/nuplan/common/maps"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

BASE_CKPT=${BASE_CKPT:-"${ROOT}/trainingResult/ckpts_20260514_213849/epoch=31-step=42560.ckpt"}
CUR_CKPT=${CUR_CKPT:-"${ROOT}/trainingResult/ckpts_20260515_162313/epoch=31-step=34048.ckpt"}
HERM_CKPT=${HERM_CKPT:-"${ROOT}/trainingResult/HERM/herm_support_1024_768_256_convattn_200ep.pt"}
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}
CTRL_REF=${CTRL_REF:-"${WORLDCAP_CTRL_REF_1024}"}
CTRL_EXEC=${CTRL_EXEC:-"${WORLDCAP_CTRL_EXEC_1024}"}

SCENE_FILTERS=(
  navtest
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
    echo "[SKIP] ${tag} ${scene_filter}: final CSV already exists"
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[RUN] ${tag} ${scene_filter}"
  "${PYTHON}" -u "${SCRIPT}" \
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
    "++agent.config.herm_enable=true" \
    "++agent.config.herm_checkpoint_path='${HERM_CKPT}'" \
    "++agent.config.herm_support_size=768" \
    "++agent.config.herm_support_seed=0" \
    "++agent.config.herm_query_chunk_size=256" \
    "++agent.config.herm_apply_in_eval=true" \
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

main() {
  for path in "${BASE_CKPT}" "${CUR_CKPT}" "${HERM_CKPT}" "${PLANNER_ANCHORS}" "${CTRL_REF}" "${CTRL_EXEC}"; do
    [[ -e "${path}" ]] || { echo "[ERR] missing ${path}" >&2; exit 1; }
  done

  for scene_filter in "${SCENE_FILTERS[@]}"; do
    run_eval base_epoch31_42560 "${BASE_CKPT}" "${scene_filter}"
    run_eval cur_epoch31_34048 "${CUR_CKPT}" "${scene_filter}"
  done
}

main "$@"
