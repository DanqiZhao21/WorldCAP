#!/bin/bash
set -euo pipefail

##################
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

SCRIPT=/home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py

# Planner anchors are already configured in WoTEConfig by default.
# If you need to force a different anchor set, add:
#   agent.config.cluster_file_path="/path/to/trajectory_anchors_256.npy"

CTRL_REF="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy"
CTRL_EXEC_DEFAULT="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy"
CTRL_EXEC_POST1515="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy"
CTRL_EXEC_AGGRESSIVE="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy"

# Optional: enable CAP risk penalty via Controller Response Predictor.
# Usage example:
#   export WOTE_USE_CTRL_RISK=1
#   export WOTE_CTRL_RP_CKPT=/abs/path/to/controller_response_predictor.pt
#   export WOTE_CTRL_RISK_W=0.2
WOTE_USE_CTRL_RISK=${WOTE_USE_CTRL_RISK:-0}
WOTE_CTRL_RP_CKPT=${WOTE_CTRL_RP_CKPT:-""}
WOTE_CTRL_RISK_W=${WOTE_CTRL_RISK_W:-0.0}
WOTE_CTRL_RISK_XY_W=${WOTE_CTRL_RISK_XY_W:-1.0}
WOTE_CTRL_RISK_YAW_W=${WOTE_CTRL_RISK_YAW_W:-0.2}

# Put all outputs under a date-stamped folder to avoid overwriting hydra output dirs.
EXP_ROOT="eval/WoTE/0202"

run_eval_3styles() {
  local MODEL_TAG="$1"; shift
  local CKPT="$1"; shift
  local INJ_MODE="$1"; shift
  local POOLING="$1"; shift
  local EXTRA_ARGS=("$@")

  local CAP_ARGS=()
  if [[ "${WOTE_USE_CTRL_RISK}" == "1" ]]; then
    if [[ -z "${WOTE_CTRL_RP_CKPT}" ]]; then
      echo "[WARN] WOTE_USE_CTRL_RISK=1 but WOTE_CTRL_RP_CKPT is empty; skipping CAP risk args"
    else
      CAP_ARGS=(
        ++agent.config.controller_use_response_predictor=true
        ++agent.config.controller_response_predictor_path=\"${WOTE_CTRL_RP_CKPT}\"
        ++agent.config.controller_risk_weight=${WOTE_CTRL_RISK_W}
        ++agent.config.controller_risk_xy_weight=${WOTE_CTRL_RISK_XY_W}
        ++agent.config.controller_risk_yaw_weight=${WOTE_CTRL_RISK_YAW_W}
      )
    fi
  fi

  # 1) default tracker + no post
  python "${SCRIPT}" \
    agent=WoTE_agent \
    agent.checkpoint_path=\"${CKPT}\" \
    experiment_name="${EXP_ROOT}/${MODEL_TAG}/sim_default_none" \
    split=test \
    scene_filter=navtest \
    simulator.tracker_style=default \
    simulator.post_style=none \
    simulator.post_params.heading_scale=1.0 \
    simulator.post_params.heading_bias=0 \
    simulator.post_params.speed_scale=1.0 \
    simulator.post_params.speed_bias=0 \
    simulator.post_params.noise_std=0 \
    agent.config.controller_ref_traj_path="${CTRL_REF}" \
    agent.config.controller_exec_traj_path="${CTRL_EXEC_DEFAULT}" \
    ++agent.config.controller_condition_scope=global \
    ++agent.config.controller_style_pooling="${POOLING}" \
    ++agent.config.controller_condition_on_traj_feature=false \
    ++agent.config.controller_condition_on_bev_tokens=true \
    agent.config.controller_injection_mode="${INJ_MODE}" \
    agent.config.controller_injection_strength=0.1 \
    agent.config.controller_inject_every_step=false \
    "${CAP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

  # 2) default tracker + poststyle yaw/speed extreme 1.5
  python "${SCRIPT}" \
    agent=WoTE_agent \
    agent.checkpoint_path=\"${CKPT}\" \
    experiment_name="${EXP_ROOT}/${MODEL_TAG}/sim_default_post1515" \
    split=test \
    scene_filter=navtest \
    simulator.tracker_style=default \
    simulator.post_style=yaw_speed_extreme \
    simulator.post_params.heading_scale=1.5 \
    simulator.post_params.heading_bias=0 \
    simulator.post_params.speed_scale=1.5 \
    simulator.post_params.speed_bias=0 \
    simulator.post_params.noise_std=0 \
    agent.config.controller_ref_traj_path="${CTRL_REF}" \
    agent.config.controller_exec_traj_path="${CTRL_EXEC_POST1515}" \
    ++agent.config.controller_condition_scope=global \
    ++agent.config.controller_style_pooling="${POOLING}" \
    ++agent.config.controller_condition_on_traj_feature=false \
    ++agent.config.controller_condition_on_bev_tokens=true \
    agent.config.controller_injection_mode="${INJ_MODE}" \
    agent.config.controller_injection_strength=0.1 \
    agent.config.controller_inject_every_step=false \
    "${CAP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

  # 3) aggressive tracker + no post
  python "${SCRIPT}" \
    agent=WoTE_agent \
    agent.checkpoint_path=\"${CKPT}\" \
    experiment_name="${EXP_ROOT}/${MODEL_TAG}/sim_aggressive_none" \
    split=test \
    scene_filter=navtest \
    simulator.tracker_style=aggressive \
    simulator.post_style=none \
    simulator.post_params.heading_scale=1.0 \
    simulator.post_params.heading_bias=0 \
    simulator.post_params.speed_scale=1.0 \
    simulator.post_params.speed_bias=0 \
    simulator.post_params.noise_std=0 \
    agent.config.controller_ref_traj_path="${CTRL_REF}" \
    agent.config.controller_exec_traj_path="${CTRL_EXEC_AGGRESSIVE}" \
    ++agent.config.controller_condition_scope=global \
    ++agent.config.controller_style_pooling="${POOLING}" \
    ++agent.config.controller_condition_on_traj_feature=false \
    ++agent.config.controller_condition_on_bev_tokens=true \
    agent.config.controller_injection_mode="${INJ_MODE}" \
    agent.config.controller_injection_strength=0.1 \
    agent.config.controller_inject_every_step=false \
    "${CAP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
}


#ADD
#ADD
#ADD 20260121  FILM_evaluate——02 1 EverySTEP融合

#都是add/film 注入强度都是0.1 都是仅仅第一步注入
#不同的：add mean //film mean //film mean im_target_bias_alpha 
#film attn 
# //film mean reward feat // film mean offset(未做 未训练)

echo "============================================================"
echo "[EVAL] add + mean pooling"
echo "============================================================"
run_eval_3styles \
  "global_mean_add_s0p1" \
  "/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260201_125438/epoch=19-step=26600.ckpt" \
  "add" \
  "mean"

echo "============================================================"
echo "[EVAL] film + mean pooling"
echo "============================================================"
run_eval_3styles \
  "global_mean_film_s0p1" \
  "/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260201_151150/epoch=19-step=26600.ckpt" \
  "film" \
  "mean"

echo "============================================================"
echo "[EVAL] film + mean pooling (imtargetbias ckpt)"
echo "============================================================"
run_eval_3styles \
  "global_mean_film_s0p1_imtargetbias_a0p05" \
  "/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260201_173505/epoch=19-step=26600.ckpt" \
  "film" \
  "mean"


echo "============================================================"
echo "[EVAL] film + attn pooling"
echo "============================================================"
run_eval_3styles \
  "global_attn_film_s0p1" \
  "/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260201_195922/epoch=19-step=26600.ckpt" \
  "film" \
  "attn"

echo "============================================================"
echo "[EVAL] film + mean pooling + reward_feature conditioning"
echo "============================================================"
run_eval_3styles \
  "global_mean_film_rewardfeat_s0p1" \
  "/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260201_222619/epoch=19-step=26600.ckpt" \
  "film" \
  "mean" \
  ++agent.config.controller_condition_on_reward_feature=true \
  ++agent.config.controller_reward_condition_strength=0.1

echo "All eval runs finished. Outputs under: ${NAVSIM_EXP_ROOT}/${EXP_ROOT}/"
