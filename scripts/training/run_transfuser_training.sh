# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
# agent=transfuser_agent \
# experiment_name=transfuser/default \
# dataloader.params.num_workers=8 \
# dataloader.params.batch_size=64 \
# scene_filter=navtrain \
# split=trainval \

# # 评估脚本
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=transfuser_agent \
'agent.checkpoint_path="/data2/yingyan_li/repo/WoTE//exp/transfuser/default/lightning_logs/version_2/checkpoints/epoch=19-step=6660.ckpt"' \
experiment_name=eval/transfuser/default/ \
split=test \
scene_filter=navtest \