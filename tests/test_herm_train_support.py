import argparse

import numpy as np

from navsim.agents.WoTE.HERM.train_support import train


def test_support_training_saves_checkpoint(tmp_path):
    ref = np.zeros((6, 8, 3), dtype=np.float32)
    ref[:, :, 0] = np.arange(8, dtype=np.float32)
    exe = np.broadcast_to(ref[None, ...], (2, 6, 8, 3)).copy()
    exe[0, :, :, 1] = 0.1
    exe[1, :, :, 1] = 0.2
    bundle = tmp_path / "bundle.npz"
    output = tmp_path / "support.pt"
    np.savez(
        bundle,
        ref_traj=ref,
        exec_trajs=exe,
        train_style_indices=np.array([0], dtype=np.int64),
        val_style_indices=np.array([1], dtype=np.int64),
    )

    args = argparse.Namespace(
        exec_path=str(bundle),
        output=str(output),
        epochs=1,
        batch_size=1,
        support_size=4,
        query_size=2,
        lr=1e-3,
        weight_decay=1e-4,
        dt=0.5,
        num_poses=8,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        style_emb_dim=8,
        style_hidden_dim=16,
        w_residual=0.5,
        seed=0,
        device="cpu",
        wandb=False,
        wandb_project="herm-support-test",
        wandb_entity=None,
        wandb_run_name=None,
    )

    metrics = train(args)

    assert output.exists()
    assert "loss" in metrics
    assert "residual_l1" in metrics
