#!/bin/bash
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1,2,3,4

# 第1次训练 attn+noEveryStep+original
python ./navsim/planning/script/run_training.py \
 agent=WoTE_agent \
 agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
  +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/LAB0_original/Anchor_NavsimSimulation_256_3.npy\
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
 agent.lr=1e-4 \
 agent.config.controller_injection_strength=0.0 \
 agent.config.min_lr=1e-5 \
 agent.config.warmup_epochs=5 \
 use_cache_without_dataset=true \
 experiment_name=WoTE/default \
 scene_filter=navtrain \
 dataloader.params.batch_size=16 \
 trainer.params.max_epochs=20 \
 split=trainval \
 agent.config.controller_injection_mode=attn \
 cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_ori_1223

#  # 第2次训练 attn+EveryStep+original
# python ./navsim/planning/script/run_training.py \
#  agent=WoTE_agent \
#  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/LAB0_original/Anchor_NavsimSimulation_256_3.npy \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#  agent.lr=1e-4 \
#  agent.config.min_lr=1e-5 \
#  agent.config.warmup_epochs=5 \
#  use_cache_without_dataset=true \
#  experiment_name=WoTE/default \
#  scene_filter=navtrain \
#  dataloader.params.batch_size=16 \
#  trainer.params.max_epochs=20 \
#  split=trainval \
#  agent.config.controller_injection_mode=attn \
#  agent.config.controller_inject_every_step=True \
#  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_ori_1223

# # 第3次训练 film+noEveryStep+original
# python ./navsim/planning/script/run_training.py \
#  agent=WoTE_agent \
#  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/LAB0_original/Anchor_NavsimSimulation_256_3.npy \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#  agent.lr=1e-4 \
#  agent.config.min_lr=1e-5 \
#  agent.config.warmup_epochs=5 \
#  use_cache_without_dataset=true \
#  experiment_name=WoTE/default \
#  scene_filter=navtrain \
#  dataloader.params.batch_size=16 \
#  trainer.params.max_epochs=20 \
#  split=trainval \
#  agent.config.controller_injection_mode=film \
#  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_ori_1223
# # 第4次训练 film+EveryStep+original
# python ./navsim/planning/script/run_training.py \
#  agent=WoTE_agent \
#  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/LAB0_original/Anchor_NavsimSimulation_256_3.npy \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#  agent.lr=1e-4 \
#  agent.config.min_lr=1e-5 \
#  agent.config.warmup_epochs=5 \
#  use_cache_without_dataset=true \
#  experiment_name=WoTE/default \
#  scene_filter=navtrain \
#  dataloader.params.batch_size=16 \
#  trainer.params.max_epochs=20 \
#  split=trainval \
#  agent.config.controller_injection_mode=film \
#  agent.config.controller_inject_every_step=True \
#  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_ori_1223




# #第5次训练
# python ./navsim/planning/script/run_training.py \
#  agent=WoTE_agent \
#  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab3_Agressive_default/Aggressive_yaw0_0_speed_extreme0_0__8.npy \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#  agent.lr=1e-4 \
#  agent.config.min_lr=1e-5 \
#  agent.config.warmup_epochs=5 \
#  use_cache_without_dataset=true \
#  experiment_name=WoTE/default \
#  scene_filter=navtrain \
#  dataloader.params.batch_size=16 \
#  trainer.params.max_epochs=30 \
#  split=trainval \
#  agent.config.controller_injection_mode=attn \
#  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget

# #第6次训练 attn+noEveryStep+Aggressive
# python ./navsim/planning/script/run_training.py \
#  agent=WoTE_agent \
#  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab3_Agressive_default/Aggressive_yaw0_0_speed_extreme0_0__8.npy \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#  agent.lr=1e-4 \
#  agent.config.min_lr=1e-5 \
#  agent.config.warmup_epochs=5 \
#  use_cache_without_dataset=true \
#  experiment_name=WoTE/default \
#  scene_filter=navtrain \
#  dataloader.params.batch_size=16 \
#  trainer.params.max_epochs=30 \
#  split=trainval \
#  agent.config.controller_injection_mode=attn \
#  agent.config.controller_inject_every_step=True \
#  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget

#  #第7次训练film+noEveryStep+Aggressive
#  python ./navsim/planning/script/run_training.py \
#  agent=WoTE_agent \
#  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab3_Agressive_default/Aggressive_yaw0_0_speed_extreme0_0__8.npy \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#  agent.lr=1e-4 \
#  agent.config.min_lr=1e-5 \
#  agent.config.warmup_epochs=5 \
#  use_cache_without_dataset=true \
#  experiment_name=WoTE/default \
#  scene_filter=navtrain \
#  dataloader.params.batch_size=16 \
#  trainer.params.max_epochs=30 \
#  split=trainval \
#  agent.config.controller_injection_mode=film \
#  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget

#  #第8次训练 film+noEveryStep+Aggressive
# python ./navsim/planning/script/run_training.py \
#  agent=WoTE_agent \
#  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab3_Agressive_default/Aggressive_yaw0_0_speed_extreme0_0__8.npy \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#  agent.lr=1e-4 \
#  agent.config.min_lr=1e-5 \
#  agent.config.warmup_epochs=5 \
#  use_cache_without_dataset=true \
#  experiment_name=WoTE/default \
#  scene_filter=navtrain \
#  dataloader.params.batch_size=16 \
#  trainer.params.max_epochs=30 \
#  split=trainval \
#  agent.config.controller_injection_mode=film \
#  agent.config.controller_inject_every_step=True \
#  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget

# # 第5次 测验 film+EveryStep
# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=7
# export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251221_212335_LAB_film_EveryStep/epoch=18-step=20216.ckpt"' \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab3_Agressive_default/Aggressive_yaw0_0_speed_extreme0_0__8.npy \
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   #修改simulator 6个参数 + 保存路径命名 + LQR风格
#   simulator.post_style=yaw_speed_extreme \
#   simulator.post_params.heading_scale=0 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=0\
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   simulator.tracker_style=aggressive \
#   +anchor_save_dir=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed \
#   +anchor_save_name=default_yaw15_0_speed_extreme15_0.npy \
#   +anchor_overwrite=False 