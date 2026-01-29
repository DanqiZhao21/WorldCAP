#!/bin/bash
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WOTE_STYLE_PROBS="0.4,0.4,0.2"
#'attn'| 'film' |  'concat' | 'add'
#自行修改111
# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 


# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \

#ADD
#film:  0.2 no yes ;1 no yes 

# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=false \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \


# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \



# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=false \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \


# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \

#ADD FILM END

#ADD ATTN 20260122 START
# film:  0.2 no yes ;1 no yes 

python ./navsim/planning/script/run_training.py \
  agent=WoTE_agent \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
  agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
  agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
  agent.config.controller_injection_mode=attn \
  agent.config.controller_injection_strength=0.2 \
  agent.config.controller_inject_every_step=false \
  use_cache_without_dataset=true \
  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
  dataloader.params.batch_size=16 \
  trainer.params.max_epochs=20 \
  split=trainval \
  experiment_name=WoTE/default \
  scene_filter=navtrain \
  agent.lr=1e-4 \
  agent.config.min_lr=1e-5 \
  agent.config.warmup_epochs=5 \


python ./navsim/planning/script/run_training.py \
  agent=WoTE_agent \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
  agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
  agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
  agent.config.controller_injection_mode=attn \
  agent.config.controller_injection_strength=0.2 \
  agent.config.controller_inject_every_step=true \
  use_cache_without_dataset=true \
  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
  dataloader.params.batch_size=16 \
  trainer.params.max_epochs=20 \
  split=trainval \
  experiment_name=WoTE/default \
  scene_filter=navtrain \
  agent.lr=1e-4 \
  agent.config.min_lr=1e-5 \
  agent.config.warmup_epochs=5 \



python ./navsim/planning/script/run_training.py \
  agent=WoTE_agent \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
  agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
  agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=1.0 \
  agent.config.controller_inject_every_step=false \
  use_cache_without_dataset=true \
  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
  dataloader.params.batch_size=16 \
  trainer.params.max_epochs=20 \
  split=trainval \
  experiment_name=WoTE/default \
  scene_filter=navtrain \
  agent.lr=1e-4 \
  agent.config.min_lr=1e-5 \
  agent.config.warmup_epochs=5 \


python ./navsim/planning/script/run_training.py \
  agent=WoTE_agent \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
  agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
  agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
  agent.config.controller_injection_mode=attn \
  agent.config.controller_injection_strength=1.0 \
  agent.config.controller_inject_every_step=true \
  use_cache_without_dataset=true \
  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
  dataloader.params.batch_size=16 \
  trainer.params.max_epochs=20 \
  split=trainval \
  experiment_name=WoTE/default \
  scene_filter=navtrain \
  agent.lr=1e-4 \
  agent.config.min_lr=1e-5 \
  agent.config.warmup_epochs=5 \





#ADD ATTN END






















# #####另外两种不是很好的；；；
# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=concat \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \


# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=add \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \

# ####0.2
# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \


# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \


# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
#   agent.config.controller_injection_mode=concat \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \


# python ./navsim/planning/script/run_training.py \
#   agent=WoTE_agent \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
#   agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
#   agent.config.controller_injection_mode=add \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=true \
#   use_cache_without_dataset=true \
#   cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
#   dataloader.params.batch_size=16 \
#   trainer.params.max_epochs=20 \
#   split=trainval \
#   experiment_name=WoTE/default \
#   scene_filter=navtrain \
#   agent.lr=1e-4 \
#   agent.config.min_lr=1e-5 \
#   agent.config.warmup_epochs=5 \



# #第6次训练 attn+noEveryStep+Aggressive
# # python ./navsim/planning/script/run_training.py \
# #  agent=WoTE_agent \
# #  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
# #  agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy \
# #  agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy \
# #  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy \
# #  agent.lr=1e-4 \
# #  agent.config.min_lr=1e-5 \
# #  agent.config.warmup_epochs=5 \
# #  use_cache_without_dataset=true \
# #  experiment_name=WoTE/default \
# #  scene_filter=navtrain \
# #  dataloader.params.batch_size=16 \
# #  trainer.params.max_epochs=20 \
# #  split=trainval \
# #  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_LQRstyleAggressive_0113 \
# #  agent.config.controller_injection_mode=attn \
# #  agent.config.controller_injection_strength=1.0 \
# #  agent.config.controller_inject_every_step=true


