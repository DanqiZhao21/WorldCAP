#!/usr/bin/env bash
set -euo pipefail

# WoTE + HERM finetune with controller embedding injected into the latent world transition.
# This mirrors 0514train_wote_herm_exec_only_4gpu_32ep.sh, except:
# - WOTE_TRAIN_PROFILE=wm_reward_only so controller embedding/fusion layers are trainable
# - controller_condition_on_world_model=true

ROOT=${ROOT:-/home/zhaodanqi/clone/WoTE}
source "${ROOT}/tool/common/worldcap_newctrl_paths.sh"

PYTHON=${PYTHON:-/home/zhaodanqi/anaconda3/bin/python}

export CUDA_VISIBLE_DEVICES=${GPU_IDS:-0,1,2,3}
export PYTHONPATH="${ROOT}:${ROOT}/navsim:${ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-/mnt/data/navsim_workspace/exp}
export WOTE_CTRL_STYLE_SPLIT=${WOTE_CTRL_STYLE_SPLIT:-train}

export WOTE_TRAIN_PROFILE=wm_reward_only
export WOTE_INIT_CKPT=${WOTE_INIT_CKPT:-"${ROOT}/trainingResult/epoch=29-step=19950.ckpt"}

HERM_CKPT=${HERM_CKPT:-"${ROOT}/trainingResult/HERM/herm_support_1024_768_256_convattn_200ep.pt"}
PLANNER_ANCHORS=${PLANNER_ANCHORS:-"${ROOT}/extra_data/planning_vb/trajectory_anchors_256.npy"}
CTRL_REF=${CTRL_REF:-"${WORLDCAP_CTRL_REF_1024}"}
CTRL_EXEC=${CTRL_EXEC:-"${WORLDCAP_CTRL_EXEC_1024}"}
CACHE_PATH=${CACHE_PATH:-"${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114"}

RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}
RUN_NAME=${RUN_NAME:-"wote_herm_controller_embedding_4gpu_32ep_${RUN_TS}"}

export WOTE_WANDB_PROJECT=${WOTE_WANDB_PROJECT:-WOTE-training-2}
export WOTE_WANDB_GROUP=${WOTE_WANDB_GROUP:-"wote-herm-controller-embedding-${RUN_TS}"}
export WOTE_WANDB_RUN_NAME="${RUN_NAME}"
export WOTE_WANDB_TAGS=${WOTE_WANDB_TAGS:-wote,herm,controller-embedding,attn-film,4gpu,32ep}

"${PYTHON}" - <<PY
import os
paths = {
    "WOTE_INIT_CKPT": r"""${WOTE_INIT_CKPT}""",
    "HERM_CKPT": r"""${HERM_CKPT}""",
    "PLANNER_ANCHORS": r"""${PLANNER_ANCHORS}""",
    "CTRL_REF": r"""${CTRL_REF}""",
    "CTRL_EXEC": r"""${CTRL_EXEC}""",
    "CACHE_PATH": r"""${CACHE_PATH}""",
}
missing = [f"{k}={v}" for k, v in paths.items() if not os.path.exists(v)]
if missing:
    raise SystemExit("[ERR] Missing required path(s):\n" + "\n".join(missing))
print("[INFO] preflight OK")
PY

cd "${ROOT}"
"${PYTHON}" ./navsim/planning/script/run_training.py \
  agent=WoTE_agent \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  ++agent.config.cluster_file_path="${PLANNER_ANCHORS}" \
  ++agent.config.controller_ref_traj_path="${CTRL_REF}" \
  ++agent.config.controller_exec_traj_path="${CTRL_EXEC}" \
  use_cache_without_dataset=true \
  cache_path="${CACHE_PATH}" \
  dataloader.params.batch_size=16 \
  trainer.params.max_epochs=32 \
  +trainer.params.devices=4 \
  split=trainval \
  experiment_name="WoTE/herm_controller_embedding/herm_controller_embedding_4gpu_32ep_${RUN_TS}" \
  scene_filter=navtrain \
  agent.lr=1e-4 \
  agent.config.min_lr=1e-6 \
  agent.config.warmup_epochs=3 \
  ++agent.config.herm_enable=true \
  ++agent.config.herm_checkpoint_path="${HERM_CKPT}" \
  ++agent.config.herm_support_size=768 \
  ++agent.config.herm_support_seed=0 \
  ++agent.config.herm_query_chunk_size=256 \
  ++agent.config.herm_apply_in_train=true \
  ++agent.config.herm_apply_in_eval=true \
  ++agent.config.controller_condition_on_world_model=true \
  ++agent.config.controller_world_model_fusion=attn_film \
  ++agent.config.controller_world_model_inject_target=all \
  ++agent.config.controller_feature_mode=full \
  ++agent.config.controller_style_pooling=attn \
  ++agent.config.use_agent_loss=false \
  ++agent.config.use_map_loss=true \
  ++agent.config.bev_semantic_weight=0.5 \
  ++agent.config.fut_bev_semantic_weight=1.0 \
  ++agent.config.traj_offset_loss_weight=0.2 \
  ++agent.config.offset_im_reward_weight=0.1
