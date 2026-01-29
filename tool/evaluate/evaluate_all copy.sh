

##################
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1,2,3
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
#===========================================================================================================
# #第0_1 原ckpt的结果
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/tool/epoch=29-step=19950.ckpt"' \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default 

# #   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/tool/epoch=29-step=19950.ckpt"' \

# # #第0_2 重新加训练20epoch之后的结果,没有融入任何simulator的信息
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251229_140113_LAB0_00/epoch=19-step=26600.ckpt"' \
#   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default 
#===========================================================================================================
#ADD第1次 attn_evaluate——02融合

# #NOTE 正常的   simulator.tracker_style=default \ simulator.post_style=none \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260115_070054_attn_02/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=none \
#   simulator.post_params.heading_scale=1.0 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.0 \
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy" \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=false

# #NOTE poststyle1.51.5    simulator.post_style=yaw_speed_extreme \   simulator.post_params.heading_scale=1.5 \   simulator.post_params.speed_scale=1.5 \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260115_070054_attn_02/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=yaw_speed_extreme \
#   simulator.post_params.heading_scale=1.5 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.5 \
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=false

# #NOTE LQRstyle aggressive   simulator.tracker_style=aggressive \   simulator.post_style=none \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260115_070054_attn_02/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=aggressive \
#   simulator.post_style=none \
#   simulator.post_params.heading_scale=1.0 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.0\
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy" \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=false

# #ADD第2次 attn_evaluate——1融合
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260115_070054_attn_02/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=none \
#   simulator.post_params.heading_scale=1.0 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.0 \
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy" \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=false

# #NOTE poststyle1.51.5    simulator.post_style=yaw_speed_extreme \   simulator.post_params.heading_scale=1.5 \   simulator.post_params.speed_scale=1.5 \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260115_070054_attn_02/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=yaw_speed_extreme \
#   simulator.post_params.heading_scale=1.5 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.5 \
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=false

# #NOTE LQRstyle aggressive   simulator.tracker_style=aggressive \   simulator.post_style=none \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260115_070054_attn_02/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=aggressive \
#   simulator.post_style=none \
#   simulator.post_params.heading_scale=1.0 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.0\
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy" \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=false


# #PRINT


# #ADD第3次 FILM_evaluate——02融合

# # 02 和 1 的no step
# #NOTE 正常的   simulator.tracker_style=default \ simulator.post_style=none \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260120_211657/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=none \
#   simulator.post_params.heading_scale=1.0 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.0 \
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy" \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=false

# #NOTE poststyle1.51.5    simulator.post_style=yaw_speed_extreme \   simulator.post_params.heading_scale=1.5 \   simulator.post_params.speed_scale=1.5 \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260120_211657/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=yaw_speed_extreme \
#   simulator.post_params.heading_scale=1.5 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.5 \
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=false

# #NOTE LQRstyle aggressive   simulator.tracker_style=aggressive \   simulator.post_style=none \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260120_211657/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=aggressive \
#   simulator.post_style=none \
#   simulator.post_params.heading_scale=1.0 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.0\
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy" \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=0.2 \
#   agent.config.controller_inject_every_step=false

# #ADD第4次 FILM_evaluate——1融合
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260121_024441/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=none \
#   simulator.post_params.heading_scale=1.0 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.0 \
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy" \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=false

# #NOTE poststyle1.51.5    simulator.post_style=yaw_speed_extreme \   simulator.post_params.heading_scale=1.5 \   simulator.post_params.speed_scale=1.5 \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260121_024441/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=yaw_speed_extreme \
#   simulator.post_params.heading_scale=1.5 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.5 \
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=false

# #NOTE LQRstyle aggressive   simulator.tracker_style=aggressive \   simulator.post_style=none \
# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260121_024441/epoch=19-step=26600.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=aggressive \
#   simulator.post_style=none \
#   simulator.post_params.heading_scale=1.0 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.0\
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy" \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=false



#ADD第3次 FILM_evaluate——02融合

# 02 和 1 的no step
#NOTE 正常的   simulator.tracker_style=default \ simulator.post_style=none \
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260120_211657/epoch=19-step=26600.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default \
  simulator.post_style=none \
  simulator.post_params.heading_scale=1.0 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.0 \
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy" \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=0.2 \
  agent.config.controller_inject_every_step=false

#NOTE poststyle1.51.5    simulator.post_style=yaw_speed_extreme \   simulator.post_params.heading_scale=1.5 \   simulator.post_params.speed_scale=1.5 \
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260120_211657/epoch=19-step=26600.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default \
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=1.5 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.5 \
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=0.2 \
  agent.config.controller_inject_every_step=false

#NOTE LQRstyle aggressive   simulator.tracker_style=aggressive \   simulator.post_style=none \
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260120_211657/epoch=19-step=26600.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=aggressive \
  simulator.post_style=none \
  simulator.post_params.heading_scale=1.0 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.0\
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy" \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=0.2 \
  agent.config.controller_inject_every_step=false

#ADD第4次 FILM_evaluate——1融合
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260121_024441/epoch=19-step=26600.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default \
  simulator.post_style=none \
  simulator.post_params.heading_scale=1.0 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.0 \
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB0_Original/Anchor_NavsimSimulation_256_3.npy" \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=1.0 \
  agent.config.controller_inject_every_step=false

#NOTE poststyle1.51.5    simulator.post_style=yaw_speed_extreme \   simulator.post_params.heading_scale=1.5 \   simulator.post_params.speed_scale=1.5 \
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260121_024441/epoch=19-step=26600.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=default \
  simulator.post_style=yaw_speed_extreme \
  simulator.post_params.heading_scale=1.5 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.5 \
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=1.0 \
  agent.config.controller_inject_every_step=false

#NOTE LQRstyle aggressive   simulator.tracker_style=aggressive \   simulator.post_style=none \
python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
  agent=WoTE_agent \
  'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260121_024441/epoch=19-step=26600.ckpt"' \
  experiment_name=eval/WoTE/default/ \
  split=test \
  scene_filter=navtest \
  simulator.tracker_style=aggressive \
  simulator.post_style=none \
  simulator.post_params.heading_scale=1.0 \
  simulator.post_params.heading_bias=0 \
  simulator.post_params.speed_scale=1.0\
  simulator.post_params.speed_bias=0 \
  simulator.post_params.noise_std=0\
  agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
  agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy" \
  agent.config.controller_injection_mode=film \
  agent.config.controller_injection_strength=1.0 \
  agent.config.controller_inject_every_step=false






# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251222_113418_LAB2_attn_EveryStep/epoch=19-step=21280.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=yaw_speed_extreme \
#   simulator.post_params.heading_scale=1.5 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.5\
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
#   agent.config.controller_injection_mode=attn \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=true 



# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251222_185954_LAB2_film_NoEveryStep/epoch=19-step=21280.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=yaw_speed_extreme \
#   simulator.post_params.heading_scale=1.5 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.5\
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=false



# python /home/zhaodanqi/clone/WoTE/navsim/planning/script/run_pdm_score_multiTraj.py \
#   agent=WoTE_agent \
#   'agent.checkpoint_path="/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20251222_233432_LAB2_film_EveryStep/epoch=19-step=21280.ckpt"' \
#   experiment_name=eval/WoTE/default/ \
#   split=test \
#   scene_filter=navtest \
#   simulator.tracker_style=default \
#   simulator.post_style=yaw_speed_extreme \
#   simulator.post_params.heading_scale=1.5 \
#   simulator.post_params.heading_bias=0 \
#   simulator.post_params.speed_scale=1.5\
#   simulator.post_params.speed_bias=0 \
#   simulator.post_params.noise_std=0\
#   agent.config.controller_ref_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256_centered.npy" \
#   agent.config.controller_exec_traj_path="/home/zhaodanqi/clone/WoTE/ControllerExp/LAB2_Poststyle_yawspeedextreme1515/default_yaw15_0_speed_extreme15_0__8.npy" \
#   agent.config.controller_injection_mode=film \
#   agent.config.controller_injection_strength=1.0 \
#   agent.config.controller_inject_every_step=true

# #   #===================0.2融合=====================================
