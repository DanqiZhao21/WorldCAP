#caching origi  training_cache_allWoteTarget_ori_1223
export TMPDIR=/mnt/data/ray_tmp
mkdir -p $TMPDIR
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/navsim:$PYTHONPATH
export PYTHONPATH=/home/zhaodanqi/clone/WoTE/nuplan-devkit:$PYTHONPATH
python navsim/planning/script/run_dataset_caching.py \
  agent=WoTE_agent \
  experiment_name=training_WOTE_agent \
  force_cache_computation=true \
  cache_path=${NAVSIM_EXP_ROOT}/training_cache_allWoteTarget_0114 \
  scene_filter=navtrain \
  worker.threads_per_node=16

