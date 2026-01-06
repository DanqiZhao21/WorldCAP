#!/bin/bash
# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=1,2,3,4,5
# /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt

  # +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

#第0_1 原ckpt的结果
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251229_052238_LAB0_02film_NoEveryStep/epoch=19-step=21280.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest1_10 \
  simulator.tracker_style=default 

  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/tool/epoch=29-step=19950.ckpt"' \


  

# #第0_2 重新加训练20epoch之后的结果,没有融入任何simulator的信息
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251229_140113_LAB0_00/epoch=19-step=26600.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default 



# 第1次 evaluate——1:1融合
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251223_205238_LAB0_attn_NoEveryStep/epoch=27-step=49672_best.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default 



python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251224_041121_LAB0_attn_EveryStep/epoch=27-step=49672_best.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default 



python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251224_093516_LAB0_film_NoEveryStep/epoch=29-step=39900_best.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default 

python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251224_134639_LAB0_film_EveryStep/epoch=29-step=39900_best.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default 
  #===================0.2融合=====================================
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251229_001549_LAB0_02attn_NoEveryStep/epoch=19-step=21280.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default 



python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251229_031522_LAB0_02attn_EveryStep/epoch=19-step=21280.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default 



python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251229_052238_LAB0_02film_NoEveryStep/epoch=19-step=21280.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default 

python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251229_071317_LAB0_02film_EveryStep/epoch=19-step=21280.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default 
# # # 第1次 evaluate
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251223_205238_LAB0_attn_NoEveryStep/epoch=27-step=49672_best.ckpt"' \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   agent.config.controller_injection_mode=none \
#   agent.config.controller_injection_strength=0.0 \
#   # scorer.anchor_save_dir=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed \
#   # scorer.anchor_save_name=default_yaw15_0_speed_extreme15_0.npy \
#   # scorer.anchor_overwrite=False 

#  # 第2次训练 attn+EveryStep+scale1515


# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251224_041121_LAB0_attn_EveryStep/epoch=27-step=49672_best.ckpt"' \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   agent.config.controller_injection_mode=none \
#   agent.config.controller_injection_strength=0.0 \
#   # scorer.anchor_save_dir=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed \
#   # scorer.anchor_save_name=default_yaw15_0_speed_extreme15_0.npy \
#   # scorer.anchor_overwrite=False

# # # 下一步
# # # 第3次训练 film+noEveryStep+scale1515
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251224_093516_LAB0_film_NoEveryStep/epoch=29-step=39900_best.ckpt"' \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   agent.config.controller_injection_mode=none \
#   agent.config.controller_injection_strength=0.0 \
#   # scorer.anchor_save_dir=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed \
#   # scorer.anchor_save_name=default_yaw15_0_speed_extreme15_0.npy \
#   # scorer.anchor_overwrite=False







# # # # 第4次训练 film+EveryStep+scale1515
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251224_134639_LAB0_film_EveryStep/epoch=29-step=39900_best.ckpt"' \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
#   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   agent.config.controller_injection_mode=none \
#   agent.config.controller_injection_strength=0.0 \
#   scorer.anchor_save_dir=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed \
#   scorer.anchor_save_name=default_yaw15_0_speed_extreme15_0.npy \
#   scorer.anchor_overwrite=False




