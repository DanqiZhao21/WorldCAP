SPLIT=test
# SPLIT=trainval

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path='/home/yingyan.li/repo/WoTE/exp/metric_cache' \
scene_filter.frame_interval=1 \
