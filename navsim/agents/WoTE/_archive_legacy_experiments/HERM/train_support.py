from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from navsim.agents.WoTE.HERM.data import SupportQueryTrajectoryDataset
from navsim.agents.WoTE.HERM.geometry import project_to_plan_frenet, wrap_angle
from navsim.agents.WoTE.HERM.losses import herm_loss
from navsim.agents.WoTE.HERM.model import HERMConfig, SupportConditionalHERM


def _metrics_to_float(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in metrics.items()}


def _style_split(exec_path: str) -> tuple[list[int], list[int]]:
    data = np.load(exec_path, allow_pickle=True)
    num_styles = int(data["exec_trajs"].shape[0])
    if "train_style_indices" in data and "val_style_indices" in data:
        train_styles = np.asarray(data["train_style_indices"], dtype=np.int64).reshape(-1).tolist()
        val_styles = np.asarray(data["val_style_indices"], dtype=np.int64).reshape(-1).tolist()
        return train_styles, val_styles
    split = max(1, int(round(num_styles * 0.8)))
    return list(range(split)), list(range(split, num_styles))


def _init_wandb(args: argparse.Namespace, config: HERMConfig, train_len: int, val_len: int):
    if not bool(getattr(args, "wandb", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging requested but wandb is not installed") from exc

    return wandb.init(
        project=getattr(args, "wandb_project", "herm-support"),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_run_name", None),
        config={
            **asdict(config),
            "exec_path": args.exec_path,
            "output": args.output,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "support_size": int(args.support_size),
            "query_size": int(args.query_size),
            "style_emb_dim": int(args.style_emb_dim),
            "style_hidden_dim": int(args.style_hidden_dim),
            "w_residual": float(getattr(args, "w_residual", 0.0)),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": args.device,
            "train_style_count": int(train_len),
            "val_style_count": int(val_len),
        },
    )


def _support_loss(
    out,
    query_plan: torch.Tensor,
    target: torch.Tensor,
    w_residual: float,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loss, metrics = herm_loss(out.exec_traj, target, out.residual, out.params)#模型预测的执行轨迹；真实的执行轨迹；模型预测的residual 残差；模型预测的frenet residual
    if float(w_residual) <= 0.0:#不启用residual supervision
        return loss, metrics

    batch, query, horizon, dim = query_plan.shape
    flat_plan = query_plan.reshape(batch * query, horizon, dim)
    flat_target = target.reshape(batch * query, horizon, dim)
    target_res = project_to_plan_frenet(flat_plan, flat_target)
    target_residual = torch.stack(
        [target_res["delta_s"], target_res["delta_d"], target_res["delta_theta"]],
        dim=-1,
    ).reshape(batch, query, horizon, 3)
    residual_l1 = torch.mean(torch.abs(out.residual - target_residual))
    residual_theta_l1 = torch.mean(torch.abs(wrap_angle(out.residual[..., 2] - target_residual[..., 2])))
    total = loss + float(w_residual) * residual_l1
    metrics = {
        **metrics,
        "loss": total.detach(),
        "trajectory_loss": loss.detach(),
        "residual_l1": residual_l1.detach(),
        "residual_theta_l1": residual_theta_l1.detach(),
    }
    return total, metrics


def _evaluate(
    model: SupportConditionalHERM,
    loader: DataLoader,
    device: torch.device,
    w_residual: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            support_plan = batch["support_plan_traj"].to(device)
            support_exec = batch["support_exec_traj"].to(device)
            query_plan = batch["query_plan_traj"].to(device)
            target = batch["query_exec_traj"].to(device)
            out = model(support_plan, support_exec, query_plan)
            loss, metrics = _support_loss(out, query_plan, target, float(w_residual))
            batch_size = int(query_plan.shape[0])
            count += batch_size
            for key, value in _metrics_to_float(metrics).items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    train_styles, val_styles = _style_split(args.exec_path)
    train_set = SupportQueryTrajectoryDataset(
        args.exec_path,
        style_indices=train_styles,
        support_size=int(args.support_size),
        query_size=int(args.query_size),
        seed=int(args.seed),
    )
    val_set = SupportQueryTrajectoryDataset(
        args.exec_path,
        style_indices=val_styles,
        support_size=int(args.support_size),
        query_size=int(args.query_size),
        seed=int(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    config = HERMConfig(
        num_poses=int(args.num_poses),
        dt=float(args.dt),
        controller_emb_dim=int(args.style_emb_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    )
    model = SupportConditionalHERM(
        config,
        style_emb_dim=int(args.style_emb_dim),
        style_hidden_dim=int(args.style_hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    wandb_run = _init_wandb(args, config, train_len=len(train_styles), val_len=len(val_styles))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf")
    best_metrics: Dict[str, float] = {}

    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            train_set.set_epoch(epoch)
            train_totals: Dict[str, float] = {}
            train_count = 0
            for batch in train_loader:
                support_plan = batch["support_plan_traj"].to(device)
                support_exec = batch["support_exec_traj"].to(device)
                query_plan = batch["query_plan_traj"].to(device)
                target = batch["query_exec_traj"].to(device)
                out = model(support_plan, support_exec, query_plan)
                loss, metrics = _support_loss(out, query_plan, target, float(getattr(args, "w_residual", 0.0)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = int(query_plan.shape[0])
                train_count += batch_size
                for key, value in _metrics_to_float(metrics).items():
                    train_totals[key] = train_totals.get(key, 0.0) + value * batch_size

            train_metrics = {key: value / max(train_count, 1) for key, value in train_totals.items()}
            val_set.set_epoch(epoch)
            val_metrics = _evaluate(model, val_loader, device, w_residual=float(getattr(args, "w_residual", 0.0)))
            val_key = val_metrics.get("pos_l1", val_metrics.get("loss", float("inf")))

            print(
                f"epoch={epoch:04d} "
                f"train_loss={train_metrics.get('loss', 0.0):.6f} "
                f"val_loss={val_metrics.get('loss', 0.0):.6f} "
                f"val_pos_l1={val_metrics.get('pos_l1', 0.0):.6f}"
            )

            if val_key < best_metric:
                best_metric = val_key
                best_metrics = val_metrics
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(config),
                        "style_emb_dim": int(args.style_emb_dim),
                        "style_hidden_dim": int(args.style_hidden_dim),
                        "support_size": int(args.support_size),
                        "query_size": int(args.query_size),
                        "w_residual": float(getattr(args, "w_residual", 0.0)),
                        "epoch": epoch,
                        "metrics": val_metrics,
                        "exec_path": args.exec_path,
                        "train_style_indices": train_styles,
                        "val_style_indices": val_styles,
                    },
                    output,
                )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"train/{key}": value for key, value in train_metrics.items()},
                        **{f"val/{key}": value for key, value in val_metrics.items()},
                        "val/best_pos_l1": best_metric,
                        "epoch": epoch,
                    },
                    step=epoch,
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return best_metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train support-conditioned Frenet Error-Dynamics HERM.")
    parser.add_argument("--exec-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--support-size", type=int, default=192)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--num-poses", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--style-emb-dim", type=int, default=64)
    parser.add_argument("--style-hidden-dim", type=int, default=128)
    parser.add_argument("--w-residual", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="herm-support")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()

'''
PYTHONPATH=/home/zhaodanqi/clone/WoTE \
python -m navsim.agents.WoTE.HERM.train_support \
  --exec-path /home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz \
  --output /home/zhaodanqi/clone/WoTE/trainingResult/HERM/herm_support_256.pt \
  --epochs 100 \
  --batch-size 8 \
  --support-size 192 \
  --query-size 64 \
  --device cuda:0 \
  --wandb \
  --wandb-project WoTE-HERM \
  --wandb-run-name herm-support-256

更新版本：
PYTHONPATH=/home/zhaodanqi/clone/WoTE \
python -m navsim.agents.WoTE.HERM.train_support \
  --exec-path /home/zhaodanqi/clone/WoTE/ControllerExp/generated/1024/controller_styles_1024.npz \
  --output /home/zhaodanqi/clone/WoTE/trainingResult/HERM/herm_support_1024_768_256_convattn_200ep.pt \
  --epochs 200 \
  --batch-size 4 \
  --support-size 768 \
  --query-size 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dt 0.5 \
  --num-poses 8 \
  --hidden-dim 256 \
  --num-layers 2 \
  --dropout 0.1 \
  --style-emb-dim 64 \
  --style-hidden-dim 128 \
  --seed 0 \
  --device cuda:0 \
  --wandb \
  --wandb-project WoTE-HERM \
  --wandb-run-name herm-support-1024-s768-q256-convattn-200ep



'''
