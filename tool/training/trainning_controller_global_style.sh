#!/bin/bash
set -euo pipefail

# =====================
# WoTE training (controller-bank -> global style token)
# - controller-bank anchors/ref+exec can be arbitrary (NOT required to align with planner anchors)
# - planner anchors are from agent.config.cluster_file_path
# - controller is injected as a global style/preference condition
# =====================

export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH

# GPU selection (modify as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional: controller style sampling probs when controller_exec_traj_path is a bundle (.npz)
# export WOTE_STYLE_PROBS="0.4,0.4,0.2"

# ---------------------
# Paths (edit these)
# ---------------------
# Planner candidate anchors (THIS defines the planning trajectory set)
PLANNER_ANCHORS="/home/zhaodanqi/clone/WoTE/extra_data/planning_vb/trajectory_anchors_256.npy"

# Controller bank (used only to extract controller style / preference)
CTRL_REF="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy"
CTRL_EXEC="/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz"

# Cache
CACHE_PATH="${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114"

# Experiment
EXP_BASE="WoTE/controller_global_style"

# ---------------------
# Quick key experiments (run all)
# ---------------------
# Notes:
# - We keep the same CACHE_PATH across runs; cache is independent of controller style.
# - Each run uses a unique experiment_name suffix to avoid overwriting.
# - Start with small strength (0.05~0.2). Strength=1.0 is usually too aggressive.

# ---------------------
# Training hyperparams
# ---------------------
BATCH_SIZE=16
MAX_EPOCHS=20
LR="1e-4"
MIN_LR="1e-5"
WARMUP_EPOCHS=5

run_one() {
  local EXP_SUFFIX="$1"; shift
  echo "============================================================"
  echo "[RUN] ${EXP_BASE}/${EXP_SUFFIX}"
  echo "============================================================"
  python ./navsim/planning/script/run_training.py \
    agent=WoTE_agent \
    agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
    ++agent.config.cluster_file_path="${PLANNER_ANCHORS}" \
    agent.config.controller_ref_traj_path="${CTRL_REF}" \
    agent.config.controller_exec_traj_path="${CTRL_EXEC}" \
    use_cache_without_dataset=true \
    cache_path="${CACHE_PATH}" \
    dataloader.params.batch_size=${BATCH_SIZE} \
    trainer.params.max_epochs=${MAX_EPOCHS} \
    split=trainval \
    experiment_name="${EXP_BASE}/${EXP_SUFFIX}" \
    scene_filter=navtrain \
    agent.lr=${LR} \
    agent.config.min_lr=${MIN_LR} \
    agent.config.warmup_epochs=${WARMUP_EPOCHS} \
    "$@"
}

# # 0) Baseline: no controller
# run_one "baseline_none" \
#   agent.config.controller_injection_mode=none \
#   agent.config.controller_injection_strength=0.0 \
#   +agent.config.controller_condition_on_reward_feature=false \
#   +agent.config.controller_condition_on_offset=false

# 1) Global style token + mean pooling + Add (most stable simplest)
run_one "global_mean_add_s0p1" \
  ++agent.config.controller_condition_scope=global \
  ++agent.config.controller_style_pooling=mean \
  agent.config.controller_injection_mode=add \
  agent.config.controller_injection_strength=0.1 \
  agent.config.controller_inject_every_step=false

# 2) Global style token + mean pooling + FiLM (recommended)
run_one "global_mean_film_s0p1" \
  ++agent.config.controller_condition_scope=global \
  ++agent.config.controller_style_pooling=mean \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=0.1 \
  agent.config.controller_inject_every_step=false

# 2c) CAP: controller-aware planning via trajectory-feature conditioning (recommended)
#     Disable BEV-token injection to avoid domain mismatch; controller affects per-trajectory features.
run_one "cap_trajfeat_mean_film_s0p1" \
  ++agent.config.controller_condition_scope=global \
  ++agent.config.controller_style_pooling=mean \
  ++agent.config.controller_condition_on_traj_feature=true \
  ++agent.config.controller_traj_condition_strength=0.1 \
  ++agent.config.controller_condition_on_bev_tokens=false \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=0.1 \
  agent.config.controller_inject_every_step=false

# 2b) Same as (2) + controller-conditioned imitation supervision (small bias)
run_one "global_mean_film_s0p1_imtargetbias_a0p05" \
  ++agent.config.controller_condition_scope=global \
  ++agent.config.controller_style_pooling=mean \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=0.1 \
  agent.config.controller_inject_every_step=false \
  ++agent.config.controller_im_target_bias_alpha=0.05 \
  ++agent.config.controller_pref_temperature=3.0

# 3) Global style token + attn pooling + FiLM (stronger, may overfit)
run_one "global_attn_film_s0p1" \
  ++agent.config.controller_condition_scope=global \
  ++agent.config.controller_style_pooling=attn \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=0.1 \
  agent.config.controller_inject_every_step=false

# 4) Condition reward_feature directly (non-WM injection point)
run_one "global_mean_film_rewardfeat_s0p1" \
  ++agent.config.controller_condition_scope=global \
  ++agent.config.controller_style_pooling=mean \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=0.1 \
  agent.config.controller_inject_every_step=false \
  ++agent.config.controller_condition_on_reward_feature=true \
  ++agent.config.controller_reward_condition_strength=0.1

# 5) Condition offset branch (optional; try only if (4) helps)
run_one "global_mean_film_offset_s0p1" \
  ++agent.config.controller_condition_scope=global \
  ++agent.config.controller_style_pooling=mean \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=0.1 \
  agent.config.controller_inject_every_step=false \
  ++agent.config.controller_condition_on_offset=true \
  ++agent.config.controller_offset_condition_strength=0.1

echo "All runs finished. Results under: ${NAVSIM_EXP_ROOT}/exp/training/${EXP_BASE} (or your configured output root)."

#都是add/film 注入强度都是0.1 都是仅仅第一步注入
#不同的：add mean //film mean //film mean im_target_bias_alpha 
#film attn 
# //film mean reward feat // film mean offset