# SPLIT=test
# CHECKPOINT="exp/training_ego_mlp_agent/ep50/lightning_logs/version_0/checkpoints/epoch=49-step=8050.ckpt"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=ego_status_mlp_agent \
experiment_name=ego_mlp_agent \
split=test \
scene_filter=navtest \
'agent.checkpoint_path="/data/yingyan.li/repo/navsim/exp/training_ego_mlp_agent/ep50/lightning_logs/version_0/checkpoints/ep50.ckpt"'