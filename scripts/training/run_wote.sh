# Define
export PYTHONPATH=/home/yingyan.li/repo/WoTE/
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/yingyan.li/repo/WoTE/dataset/maps"
export NAVSIM_EXP_ROOT="/home/yingyan.li/repo/WoTE/exp"
export NAVSIM_DEVKIT_ROOT="/home/yingyan.li/repo/WoTE/"
export OPENSCENE_DATA_ROOT="/home/yingyan.li/repo/WoTE/dataset"

CONFIG_NAME=default

# training
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=WoTE/${CONFIG_NAME} \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
trainer.params.max_epochs=30  \
split=trainval 

# evaluation
python ./navsim/planning/script/run_pdm_score.py \
agent=WoTE_agent \
'agent.checkpoint_path="/home/yingyan.li/repo/WoTE/exp/WoTE/default/lightning_logs/version_0/checkpoints/epoch=29-step=19950.ckpt"' \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=eval/WoTE/${CONFIG_NAME}/ \
split=test \
scene_filter=navtest \
