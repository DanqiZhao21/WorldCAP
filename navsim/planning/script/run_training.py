from typing import Tuple
import hydra
from hydra.utils import instantiate
import logging, torch
from omegaconf import DictConfig
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.agents.abstract_agent import AbstractAgent
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb
from io import BytesIO
import tempfile
import os
from datetime import datetime
import threading
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
import re

logger = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_{timestamp}"
print(f"✅✅✅✅✅✅✅✅✅save_dir is {save_dir}✅✅✅✅✅✅✅✅✅")

ckpt_callback = ModelCheckpoint(
    dirpath=save_dir,
    filename="{epoch}-{step}",   # 文件名中不需要加时间，目录已有时间戳
    monitor=None,                 # 不监控指标，保存所有
    mode="min",
    save_top_k=-1,                # 保存所有 checkpoint
    save_last=True,               # 额外保留最后一个
    every_n_epochs=1,             # 每个 epoch 结束保存一次
)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


class WandbModelArtifactCallback(Callback):
    """在保存 checkpoint 后，将模型 ckpt 作为 artifact 上传到 Weights & Biases。
    避免重复上传：内部维护已上传路径集合；每次验证结束检查最新路径。
    """
    def __init__(self, wandb_logger: WandbLogger):
        super().__init__()
        self._wandb_logger = wandb_logger
        self._uploaded: set[str] = set()

    @staticmethod
    def _sanitize_artifact_name(name: str) -> str:
        # W&B artifact name限制：只允许字母数字、-、_、.
        return re.sub(r"[^0-9A-Za-z._-]+", "_", name)

    def on_validation_end(self, trainer, pl_module):
        try:
            self._on_validation_end_impl(trainer, pl_module)
        except Exception:
            # 任何 W&B 上传失败都不应中断训练/后续实验。
            logger.exception("W&B artifact upload failed in on_validation_end; continuing without stopping training.")

    def _on_validation_end_impl(self, trainer, pl_module):
        # 定位 ModelCheckpoint 回调
        mc: ModelCheckpoint = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                mc = cb
                break
        if mc is None:
            return

        # 使用 last_model_path（本 epoch 保存的最新 ckpt）
        ckpt_path = getattr(mc, "last_model_path", None)
        if not ckpt_path or not os.path.isfile(ckpt_path) or ckpt_path in self._uploaded:
            return
        # 仅在主进程上传，避免 DDP 多次上传
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        artifact_name = self._sanitize_artifact_name(
            f"{pl_module.__class__.__name__}-{Path(ckpt_path).stem}"
        )
        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_file(ckpt_path)
        self._wandb_logger.experiment.log_artifact(
            artifact,
            aliases=["last", f"epoch-{trainer.current_epoch}"]
        )
        self._uploaded.add(ckpt_path)

    def on_train_end(self, trainer, pl_module):
        try:
            self._on_train_end_impl(trainer, pl_module)
        except Exception:
            logger.exception("W&B artifact upload failed in on_train_end; continuing without stopping training.")

    def _on_train_end_impl(self, trainer, pl_module):
        # 训练结束时再上传一次最终/最佳权重，双重保险
        mc: ModelCheckpoint = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                mc = cb
                break
        if mc is None:
            return
        # 仅在主进程上传
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        candidates = []
        last_path = getattr(mc, "last_model_path", None)
        best_path = getattr(mc, "best_model_path", None)
        if last_path and os.path.isfile(last_path):
            candidates.append((last_path, ["final", f"epoch-{trainer.current_epoch}"]))
        if best_path and os.path.isfile(best_path) and best_path not in self._uploaded:
            candidates.append((best_path, ["best"]))

        for path, aliases in candidates:
            if path in self._uploaded:
                continue
            artifact_name = self._sanitize_artifact_name(
                f"{pl_module.__class__.__name__}-{Path(path).stem}"
            )
            artifact = wandb.Artifact(name=artifact_name, type="model")
            artifact.add_file(path)
            self._wandb_logger.experiment.log_artifact(artifact, aliases=aliases)
            self._uploaded.add(path)


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if train_scene_filter.log_names is not None:
        # train_scene_filter.log_names = [l for l in train_scene_filter.log_names if l in cfg.train_logs]
        train_scene_filter.log_names = list(set(train_scene_filter.log_names) & set(cfg.train_logs))
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if val_scene_filter.log_names is not None:
        # val_scene_filter.log_names = [l for l in val_scene_filter.log_names if l in cfg.val_logs]
        val_scene_filter.log_names = list(set(val_scene_filter.log_names) & set(cfg.val_logs))
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    train_debug = cfg.train_debug if hasattr(cfg, "train_debug") else False

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        train_debug=train_debug,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    use_fut_frames = agent.config.use_fut_frames if hasattr(agent.config, "use_fut_frames") else False
    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    return train_data, val_data





@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
        ckpt_path="/home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt"
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert cfg.force_cache_computation==False, "force_cache_computation must be False when using cached data without building SceneLoader"
        assert cfg.cache_path is not None, "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer_params = cfg.trainer.params
    # 避免 DDP 在部分模块未参与当前 loss 计算时报错
    trainer_params['strategy'] = "ddp_find_unused_parameters_true"
    
    logger_wandb = WandbLogger(
        project="WOTE-training-2",  # 项目名
        name="ControllerInTheLoop",
        save_dir="/home/zhaodanqi/clone/WoTE/trainingResult",    # 训练结果存储路径
    )

    # 可选：记录梯度/参数（视需求打开，可能有开销）
    # logger_wandb.watch(lightning_module, log="all", log_freq=100)
    
    
    trainer = pl.Trainer(
        **trainer_params,
        logger=logger_wandb,
        callbacks=[
            ckpt_callback,
            LearningRateMonitor(logging_interval='step'),
            # 默认不要上传 checkpoint 到 W&B（只记录指标/日志）。
            # 如需开启：export WOTE_WANDB_UPLOAD_CKPT=1
            *(
                [WandbModelArtifactCallback(logger_wandb)]
                if os.getenv("WOTE_WANDB_UPLOAD_CKPT", "0") == "1"
                else []
            ),
        ],
    )

    logger.info("Starting Training")
    # ckpt_path = '/home/zhaodanqi/clone/WoTE/epoch=29-step=19950.ckpt'
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # ckpt_path=ckpt_path,
        ckpt_path=None,   # 注意！此处不要再加载 ckpt！
    )

if __name__ == "__main__":
    main()