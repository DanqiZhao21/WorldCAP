

##################
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=6
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
#===========================================================================================================

# 第1次 evaluate——1:1融合
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251222_084015_LAB2_attn_NoEveryStep/epoch=19-step=21280.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest_different \
  simulator.tracker_style=aggressive \
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=1.0 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.0\
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
  agent.config.controller_injection_mode=attn \
  agent.config.controller_injection_strength=1.0 \
  agent.config.controller_inject_every_step=false \
  worker=sequential




python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251222_113418_LAB2_attn_EveryStep/epoch=19-step=21280.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default \
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=1.5 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.5\
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
  agent.config.controller_injection_mode=attn \
  agent.config.controller_injection_strength=1.0 \
  agent.config.controller_inject_every_step=true 



python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251222_185954_LAB2_film_NoEveryStep/epoch=19-step=21280.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default \
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=1.5 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.5\
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=1.0 \
  agent.config.controller_inject_every_step=false



python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251222_233432_LAB2_film_EveryStep/epoch=19-step=21280.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default \
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=1.5 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.5\
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=1.0 \
  agent.config.controller_inject_every_step=true

#   #===================0.2融合=====================================
