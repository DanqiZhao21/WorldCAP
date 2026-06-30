from __future__ import annotations

import argparse
import math
from typing import Dict

import torch
from torch.utils.data import DataLoader

from navsim.agents.WoTE.HERM.data import ControllerTrajectoryDataset
from navsim.agents.WoTE.HERM.geometry import wrap_angle
from navsim.agents.WoTE.HERM.inference import load_herm_checkpoint


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    model = load_herm_checkpoint(args.checkpoint, device=device)
    dataset = ControllerTrajectoryDataset(args.ref_path, args.exec_path)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    totals = {
        "xy_l1": 0.0,
        "xy_rmse_sum": 0.0,
        "fde": 0.0,
        "yaw_abs": 0.0,
        "final_yaw_abs": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            plan = batch["plan_traj"].to(device)
            target = batch["exec_traj"].to(device)
            pred = model(plan).exec_traj
            batch_size = int(plan.shape[0])
            count += batch_size

            xy_err = pred[..., :2] - target[..., :2]
            yaw_err = wrap_angle(pred[..., 2] - target[..., 2])
            totals["xy_l1"] += torch.mean(torch.abs(xy_err)).item() * batch_size
            totals["xy_rmse_sum"] += torch.mean(torch.sum(xy_err * xy_err, dim=-1)).item() * batch_size
            totals["fde"] += torch.mean(torch.linalg.norm(xy_err[:, -1, :], dim=-1)).item() * batch_size
            totals["yaw_abs"] += torch.mean(torch.abs(yaw_err)).item() * batch_size
            totals["final_yaw_abs"] += torch.mean(torch.abs(yaw_err[:, -1])).item() * batch_size

    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["xy_rmse"] = math.sqrt(max(metrics.pop("xy_rmse_sum"), 0.0))
    for key in ["xy_l1", "xy_rmse", "fde", "yaw_abs", "final_yaw_abs"]:
        print(f"{key}: {metrics[key]:.6f}")
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate standalone Frenet Error-Dynamics HERM.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ref-path", default=None)
    parser.add_argument("--exec-path", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
