export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  
python /home/zhaodanqi/clone/WoTE/scripts/miscs/k_means_trajs.py    


# ##
# export HYDRA_FULL_ERROR=1
# export HYDRA_RUN_DIR="/home/zhaodanqi/clone/WoTE"
# export HYDRA_OUTPUT_SUBDIR="null"
# export PYTHONPATH=/home/zhaodanqi/clone/DiffusionDrive/WoTE/navsim:$PYTHONPATH
# bash /home/zhaodanqi/clone/WoTE/scripts/evaluation/eval_wote.sh

export PYTHONPATH=/home/zhaodanqi/clone/DiffusionDrive/navsim:$PYTHONPATH

export PYTHONPATH=/home/zhaodanqi/clone/DiffusionDrive/WoTE/navsim:/home/zhaodanqi/clone/DiffusionDrive/navsim:$PYTHONPATH
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
# export OPENSCENE_DATA_ROOT="/home/yingyan.li/repo/WoTE/dataset"

CONFIG_NAME=default

# evaluation, change the checkpoint_path
python /home/zhaodanqi/clone/DiffusionDrive/WoTE/navsim/planning/script/run_pdm_score.py \
agent=WoTE_agent \
'agent.checkpoint_path="/home/zhaodanqi/clone/DiffusionDrive/WoTE/epoch=29-step=19950.ckpt"' \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=eval/WoTE/${CONFIG_NAME}/ \
split=test \
scene_filter=navtest 

===================================√✅ ORIGINAL EVALUATE============================================================================
【原始模型】
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
# PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:/home/zhaodanqi/clone/nuplan-devkit \
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
CONFIG_NAME=default
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score.py \
agent=WoTE_agent \
'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/epoch=29-step=19950.ckpt"' \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=eval/WoTE/${CONFIG_NAME}/ \
split=test \
scene_filter=navtest 


【自己重头开始训练的模型】
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
# PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:/home/zhaodanqi/clone/nuplan-devkit \
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
CONFIG_NAME=default
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score.py \
agent=WoTE_agent \
'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251206_001102/epoch=epoch=14.ckpt"' \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=eval/WoTE/${CONFIG_NAME}/ \
split=test \
scene_filter=navtest 

/home/zhaodanqi/clone/WoTE/trainingResult/WOTE-training/sjfx64he/checkpoints/epoch=29-step=26610.ckpt

export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
# PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:/home/zhaodanqi/clone/nuplan-devkit \
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
CONFIG_NAME=default
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score.py \
agent=WoTE_agent \
'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251210_234427/epoch=15-step=17024.ckpt"' \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=eval/WoTE/${CONFIG_NAME}/ \
split=test \
scene_filter=navtest 
=========================✅ MULTI TRAJ=============================

export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
# PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:/home/zhaodanqi/clone/nuplan-devkit \
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
CONFIG_NAME=default
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
agent=WoTE_agent \
'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251218_213811/epoch=13-step=18620.ckpt"' \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=eval/WoTE/${CONFIG_NAME}/ \
split=test \
scene_filter=navtest 



python -m navsim.planning.script.run_pdm_score simulator.tracker_style=aggressive
===================================√ TRAIN============================================================================
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
CONFIG_NAME=default

# training
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
agent.lr=1e-4 \
+agent.config.min_lr=1e-5\
 +agent.config.warmup_epochs=5\
experiment_name=WoTE/${CONFIG_NAME} \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
trainer.params.max_epochs=60  \
split=trainval \
  agent.config.controller_injection_mode=attn
  
  

# =======================少量数据集进行过滤===============
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
CONFIG_NAME=default

# training
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=WoTE/${CONFIG_NAME} \
scene_filter=navtrain_threelog \
dataloader.params.batch_size=16 \
trainer.params.max_epochs=60  \
split=trainval 



# ===================================√ CACHE DATA FOR TRAIN============================================================================
# # ulimit -v $((16 * 1024 * 1024))
# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_dataset_caching.py agent=WoTE_agent  experiment_name=training_WOTE_agent  scene_filter=navtrain
# # output_dir: ${oc.env:NAVSIM_EXP_ROOT}/${experiment_name}

export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
python navsim/planning/script/run_dataset_caching.py \
  agent=WoTE_agent experiment_name=training_WOTE_agent scene_filter=navtrain \
  worker.threads_per_node=16
  
# '''✅这个是进行train的数据cache的 已经成功'''
# PYTHONPATH=~/clone/DiffusionDrive \
# python navsim/planning/script/run_dataset_caching.py \
# agent=diffusiondrive_agent experiment_name=training_diffusiondrive_agent train_test_split=navtrain

  
1️⃣1️⃣1️⃣1️⃣1️⃣====================Trainning训练代码=====================
##LAB3
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

python ./navsim/planning/script/run_training.py \
 agent=WoTE_agent \
 agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
  +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab3_Agressive_default/Aggressive_yaw0_0_speed_extreme0_0__8.npy \
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
 agent.lr=1e-4 \
 agent.config.min_lr=1e-5 \
 agent.config.warmup_epochs=5 \
 use_cache_without_dataset=true \
 experiment_name=WoTE/default \
 scene_filter=navtrain \
 dataloader.params.batch_size=16 \
 trainer.params.max_epochs=30 \
 split=trainval \
 agent.config.controller_injection_mode=film \
 agent.config.controller_inject_every_step=True 
#  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget
#  cache_path:= ${oc.env:NAVSIM_EXP_ROOT}/training_cache_allWoteTarget 

#LAB2
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

python ./navsim/planning/script/run_training.py \
 agent=WoTE_agent \
 agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
  +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed/default_yaw15_0_speed_extreme15_0__8.npy \
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
 agent.lr=1e-4 \
 agent.config.min_lr=1e-5 \
 agent.config.warmup_epochs=5 \
 use_cache_without_dataset=true \
 experiment_name=WoTE/default \
 scene_filter=navtrain \
 dataloader.params.batch_size=16 \
 trainer.params.max_epochs=30 \
 split=trainval \
 agent.config.controller_injection_mode=film \
 agent.config.controller_inject_every_step=True \
 cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_1515
#  cache_path:= ${oc.env:NAVSIM_EXP_ROOT}/training_cache_allWoteTarget1515 
 
   
python ./navsim/planning/script/run_training.py  agent=WoTE_agent  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig   +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy  +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab3_Agressive_default/Aggressive_yaw0_0_speed_extreme0_0__8.npy   +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy  agent.lr=1e-4  agent.config.min_lr=1e-5  agent.config.warmup_epochs=5  experiment_name=WoTE/default  scene_filter=navtrain  dataloader.params.batch_size=16  trainer.params.max_epochs=30  split=trainval  agent.config.controller_injection_mode=attn
 
#  ===============================
  attn film concat add四种不同的融合方式
#之前是epoch=30


2️⃣2️⃣2️⃣2️⃣2️⃣====================测评示例========================
  
 #======================================= 
# 将航向角变化放大 3 倍，并加 0.2 rad 偏置：
python -m navsim.planning.script.run_pdm_score \
  simulator.post_style=yaw_scale \
  simulator.post_params.heading_scale=3.0 \
  simulator.post_params.heading_bias=0.2
# 将速度缩小到原来 0.5 倍，并加 0.5 m/s 偏置
python -m navsim.planning.script.run_pdm_score \
  simulator.post_style=speed_scale \
  simulator.post_params.speed_scale=0.5 \
  simulator.post_params.speed_bias=0.5

# 同时极端放大航向与速度，并加噪声：
python -m navsim.planning.script.run_pdm_score \
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=2.5 \
  simulator.post_params.speed_scale=1.8 \
  simulator.post_params.noise_std=0.03
#==========================================================================
# 可与已有的 tracker 风格联用（先通过 LQR 产生风格，再后处理更激进）：
python -m navsim.planning.script.run_pdm_score \
  simulator.tracker_style=aggressive \
  simulator.post_style=aggressive_post \
  simulator.post_params.heading_scale=2.0 \
  simulator.post_params.speed_scale=1.5
  
# 覆盖写入到自定义目录文件：  
python -m navsim.planning.script.run_pdm_score_multiTraj \
  anchor_save_dir="/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251219" \
  anchor_save_name="anchors_simulated.npy" \
  anchor_overwrite=true
  
=========================================
=========================================

# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=7
# # PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:/home/zhaodanqi/clone/nuplan-devkit \
# export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
# CONFIG_NAME=default
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/epoch=29-step=19950.ckpt"' \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchor_NavsimSimulation_256_3.npy" \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest\
#   simulator.post_style=yaw_scale \
#   simulator.post_params.heading_scale=3.0 \
#   simulator.post_params.heading_bias=0.2\
#   simulator.tracker_style=aggressive \
#   anchor_save_dir="/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251219" \
#   anchor_save_name="anchors_simulated_aggressive_headsc3_headbia02.npy" \
#   anchor_overwrite=true
#####LAB3 Aggressive ==》有点问题，似乎得重新跑一下（不能scale设置成0）；训练的时候似乎也没解冻
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251221_161012_LAB3_attn_EveryStep/epoch=28-step=38570.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
  +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab3_Agressive_default/Aggressive_yaw0_0_speed_extreme0_0__8.npy \
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  #修改simulator 6个参数 + 保存路径命名 + LQR风格
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=0 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=0\
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  simulator.tracker_style=aggressive \
  +anchor_save_dir=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed \
  +anchor_save_name=default_yaw15_0_speed_extreme15_0.npy \
  +anchor_overwrite=False 
  
#==LAB2 Scale1515==================
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251222_233432_LAB2_film_EveryStep/epoch=25-step=27664.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default \
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=1.5 \
  simulator.post_params.heading_bias=0.0 \
  simulator.post_params.speed_scale=1.5 \
  simulator.post_params.speed_bias=0.0 \
  simulator.post_params.noise_std=0.0 \
  scorer.anchor_save_dir=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed \
  scorer.anchor_save_name=default_yaw15_0_speed_extreme15_0.npy \
  scorer.anchor_overwrite=False 
#======LAB0 original ==================
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251223_205238_LAB0_attn_NoEveryStep/epoch=27-step=49672_best.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default \
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=1 \
  simulator.post_params.heading_bias=0.0 \
  simulator.post_params.speed_scale=1 \
  simulator.post_params.speed_bias=0.0 \
  simulator.post_params.noise_std=0.0 \
  scorer.anchor_save_dir=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed \
  scorer.anchor_save_name=default_yaw15_0_speed_extreme15_0.npy \
  scorer.anchor_overwrite=False 


3️⃣3️⃣3️⃣3️⃣3️⃣==================CACHE示例==================== 只与初始anchor和提供的exc anchor有关
# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
# export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
# python navsim/planning/script/run_dataset_caching.py \
#   agent=WoTE_agent experiment_name=training_WOTE_agent scene_filter=navtrain \
#   worker.threads_per_node=16

export TMPDIR=/mnt/data/ray_tmp
mkdir -p $TMPDIR
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
python navsim/planning/script/run_dataset_caching.py \
  agent=WoTE_agent \
  +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
  +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/LAB0_original/Anchor_NavsimSimulation_256_3.npy\
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
  experiment_name=training_WOTE_agent \
  force_cache_computation=true \
  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_ori_1223 \
  scene_filter=navtrain \
  worker.threads_per_node=16
  








##杀死所有僵尸进程
##sudo kill -9 $(ps -eo pid,ppid,stat | awk '$3 ~ /^Z/ {print $2}' | sort -u)  

##杀死wandb
# import wandb
# init(id="aksjd6gu", project="WOTE-training-2")
# wandb.finish()




export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"

python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251221_212335_LAB3_film_EveryStep/epoch=17-step=19152.ckpt"' \
  agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
  +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
  +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab3_Agressive_default/Aggressive_yaw0_0_speed_extreme0_0__8.npy \
  +agent.config.cluster_file_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256.npy \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  #修改simulator 6个参数 + 保存路径命名 + LQR风格
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=0 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=0\
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  simulator.tracker_style=aggressive \
  +anchor_save_dir=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/lab2_15headingAnd15speed \
  +anchor_save_name=default_yaw15_0_speed_extreme15_0.npy \
  +anchor_overwrite=False 