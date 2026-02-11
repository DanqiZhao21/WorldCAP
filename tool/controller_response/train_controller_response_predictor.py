#!/usr/bin/env python
import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ControllerInTheLoop.step2_Embedding.ControllerEmbedding import ControllerEmbedding
from navsim.agents.WoTE.controller_response_predictor import ControllerResponsePredictor


class StyleTrajDataset(Dataset):
    """Dataset over (style_id, traj_id)."""

    def __init__(self, exec_trajs: np.ndarray, ref_traj: np.ndarray):
        # exec_trajs: [S, N, T, 3], ref_traj: [N, T, 3]
        assert exec_trajs.ndim == 4, exec_trajs.shape
        assert ref_traj.ndim == 3, ref_traj.shape
        self.exec_trajs = exec_trajs
        self.ref_traj = ref_traj
        self.num_styles = exec_trajs.shape[0]
        self.num_trajs = exec_trajs.shape[1]

    def __len__(self) -> int:
        return int(self.num_styles * self.num_trajs)

    def __getitem__(self, idx: int):
        s = idx // self.num_trajs
        t = idx % self.num_trajs
        ref = self.ref_traj[t]  # [T,3]
        exe = self.exec_trajs[s, t]  # [T,3]
        return int(s), int(t), ref.astype(np.float32), exe.astype(np.float32)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, required=True, help=".npz with exec_trajs/ref_traj")
    ap.add_argument("--out", type=str, required=True, help="output .pt checkpoint")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--phi-bank", type=int, default=32, help="num trajs sampled to compute style embedding phi")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean"], help="pooling for style embedding")
    ap.add_argument("--feature-mode", type=str, default="full", choices=["full", "lateral_only"], help="ControllerEmbedding feature_mode")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.bundle, allow_pickle=True)
    exec_trajs = data.get("exec_trajs", None)
    ref_traj = data.get("ref_traj", None)
    if exec_trajs is None or ref_traj is None:
        raise RuntimeError("bundle must contain exec_trajs and ref_traj")

    # Shapes: exec_trajs [S, N, T, 3], ref_traj [N, T, 3]
    S, N, T, D = exec_trajs.shape
    assert ref_traj.shape == (N, T, D), (ref_traj.shape, (N, T, D))

    device = torch.device(args.device)

    controller_encoder = ControllerEmbedding(emb_dim=64, feature_mode=args.feature_mode).to(device)
    response_predictor = ControllerResponsePredictor(num_poses=T, traj_dim=D, controller_emb_dim=64).to(device)

    params = list(controller_encoder.parameters()) + list(response_predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    dataset = StyleTrajDataset(exec_trajs=exec_trajs, ref_traj=ref_traj)

    def sample_batch(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (ref_target, exec_target, phi) each shaped [B, T, 3]/[B,64]."""
        ref_target_list = []
        exec_target_list = []
        phi_list = []

        for _ in range(batch_size):
            style_id = random.randrange(S)
            target_id = random.randrange(N)

            # sample a small bank to compute style embedding
            bank_ids = np.random.choice(N, size=min(args.phi_bank, N), replace=False)

            ref_bank = torch.from_numpy(ref_traj[bank_ids]).to(device)
            exec_bank = torch.from_numpy(exec_trajs[style_id, bank_ids]).to(device)

            bank_emb = controller_encoder(ref_bank, exec_bank)  # [M, 64]
            phi = bank_emb.mean(dim=0, keepdim=False)  # [64]

            ref_t = torch.from_numpy(ref_traj[target_id]).to(device)
            exe_t = torch.from_numpy(exec_trajs[style_id, target_id]).to(device)

            ref_target_list.append(ref_t)
            exec_target_list.append(exe_t)
            phi_list.append(phi)

        return (
            torch.stack(ref_target_list, dim=0),
            torch.stack(exec_target_list, dim=0),
            torch.stack(phi_list, dim=0),
        )

    for epoch in range(args.epochs):
        controller_encoder.train()
        response_predictor.train()

        losses = []
        for step in range(args.steps_per_epoch):
            ref_b, exec_b, phi_b = sample_batch(args.batch_size)
            target_residual = exec_b - ref_b

            pred_residual = response_predictor(ref_b, phi_b)
            loss = F.mse_loss(pred_residual, target_residual)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            losses.append(float(loss.item()))

        avg_loss = sum(losses) / max(1, len(losses))
        print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.6f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ckpt = {
        "controller_encoder": controller_encoder.state_dict(),
        "response_predictor": response_predictor.state_dict(),
        "meta": {
            "bundle": args.bundle,
            "T": T,
            "D": D,
            "feature_mode": args.feature_mode,
        },
    }
    torch.save(ckpt, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
