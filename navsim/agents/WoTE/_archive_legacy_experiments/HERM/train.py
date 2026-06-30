from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from navsim.agents.WoTE.HERM.data import ControllerTrajectoryDataset
from navsim.agents.WoTE.HERM.losses import herm_loss
from navsim.agents.WoTE.HERM.model import HERMConfig, FrenetErrorDynamicsHERM


def _metrics_to_float(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in metrics.items()}


def _evaluate(model: FrenetErrorDynamicsHERM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            plan = batch["plan_traj"].to(device)
            target = batch["exec_traj"].to(device)
            out = model(plan)
            _, metrics = herm_loss(out.exec_traj, target, out.residual, out.params)
            batch_size = int(plan.shape[0])
            count += batch_size
            for key, value in _metrics_to_float(metrics).items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def _npz_style_split(exec_path: str) -> Tuple[list[int], list[int]] | None:
    if not str(exec_path).endswith(".npz"):
        return None

    data = np.load(exec_path, allow_pickle=True)
    if "train_style_indices" not in data or "val_style_indices" not in data:
        return None

    train_styles = np.asarray(data["train_style_indices"], dtype=np.int64).reshape(-1).tolist()
    val_styles = np.asarray(data["val_style_indices"], dtype=np.int64).reshape(-1).tolist()
    if not train_styles or not val_styles:
        return None
    return train_styles, val_styles


def _build_datasets(args: argparse.Namespace) -> Tuple[Dataset, Dataset, Dict[str, int | str]]:
    style_split = _npz_style_split(args.exec_path)
    if style_split is not None:
        train_styles, val_styles = style_split
        train_set = ControllerTrajectoryDataset(args.ref_path, args.exec_path, style_indices=train_styles)
        val_set = ControllerTrajectoryDataset(args.ref_path, args.exec_path, style_indices=val_styles)
        return train_set, val_set, {
            "split_mode": "style",
            "train_style_count": len(train_styles),
            "val_style_count": len(val_styles),
        }

    dataset = ControllerTrajectoryDataset(args.ref_path, args.exec_path)
    val_len = int(round(len(dataset) * float(args.val_ratio)))
    val_len = min(max(val_len, 1), max(len(dataset) - 1, 1)) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len

    if val_len > 0:
        generator = torch.Generator().manual_seed(int(args.seed))
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    else:
        train_set = dataset
        val_set = dataset
    return train_set, val_set, {"split_mode": "random", "train_style_count": 0, "val_style_count": 0}


def _init_wandb(
    args: argparse.Namespace,
    config: HERMConfig,
    train_len: int,
    val_len: int,
    split_info: Dict[str, int | str],
):
    if not bool(getattr(args, "wandb", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging requested but wandb is not installed") from exc

    return wandb.init(
        project=getattr(args, "wandb_project", "herm"),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_run_name", None),
        config={
            **asdict(config),
            "ref_path": args.ref_path,
            "exec_path": args.exec_path,
            "output": args.output,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "device": args.device,
            "train_len": int(train_len),
            "val_len": int(val_len),
            **split_info,
        },
    )


def train(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    train_set, val_set, split_info = _build_datasets(args)
    train_len = len(train_set)
    val_len = len(val_set)

    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    config = HERMConfig(
        num_poses=int(args.num_poses),
        dt=float(args.dt),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
    )
    model = FrenetErrorDynamicsHERM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    wandb_run = _init_wandb(args, config, train_len=train_len, val_len=val_len, split_info=split_info)

    best_metric = float("inf")
    best_metrics: Dict[str, float] = {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            train_totals: Dict[str, float] = {}
            train_count = 0
            for batch in train_loader:
                plan = batch["plan_traj"].to(device)
                target = batch["exec_traj"].to(device)
                out = model(plan)
                loss, metrics = herm_loss(out.exec_traj, target, out.residual, out.params)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = int(plan.shape[0])
                train_count += batch_size
                for key, value in _metrics_to_float(metrics).items():
                    train_totals[key] = train_totals.get(key, 0.0) + value * batch_size

            train_metrics = {key: value / max(train_count, 1) for key, value in train_totals.items()}
            val_metrics = _evaluate(model, val_loader, device)
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
                        "epoch": epoch,
                        "metrics": val_metrics,
                        "ref_path": args.ref_path,
                        "exec_path": args.exec_path,
                        "split_info": split_info,
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
    parser = argparse.ArgumentParser(description="Train standalone Frenet Error-Dynamics HERM.")
    parser.add_argument("--ref-path", default=None)
    parser.add_argument("--exec-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--num-poses", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="herm")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
