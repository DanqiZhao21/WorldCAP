python scripts/misc/gen_multi_trajs_pdm_score.py \
agent=WoTE_agent \
'agent.checkpoint_path="/home/yingyan.li/repo/WoTE/exp/WoTE/default/lightning_logs/version_0/checkpoints/epoch=29-step=9990.ckpt"' \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
experiment_name=eval/gen_data \
metric_cache_path='/data2/yingyan_li/repo/WoTE/dataset/metric_cache/trainval' \
split=trainval \
scene_filter=navtrain \