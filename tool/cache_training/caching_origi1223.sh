#caching origi  training_cache_allWoteTarget_ori_1223
export TMPDIR=/mnt/data/ray_tmp
mkdir -p $TMPDIR
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
python navsim/planning/script/run_dataset_caching.py \
  agent=WoTE_agent \
  +agent.config.controller_ref_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Anchors_Original_256_centered.npy\
  +agent.config.controller_exec_traj_path=/home/zhaodanqi/clone/WoTE/ControllerInTheLoop/step0_validationOfSimulation/Post20251220/LAB0_original/Anchor_NavsimSimulation_256_3.npy\
  experiment_name=training_WOTE_agent \
  force_cache_computation=true \
  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_ori_1223 \
  scene_filter=navtrain \
  worker.threads_per_node=16

