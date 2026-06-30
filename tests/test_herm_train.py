import argparse
import sys
import types

import numpy as np

from navsim.agents.WoTE.HERM.train import _build_datasets, train


def test_build_datasets_uses_npz_style_split(tmp_path):
    ref = np.zeros((3, 8, 3), dtype=np.float32)
    exe = np.zeros((4, 3, 8, 3), dtype=np.float32)
    bundle_path = tmp_path / "bundle.npz"
    np.savez(
        bundle_path,
        ref_traj=ref,
        exec_trajs=exe,
        train_style_indices=np.array([0, 2], dtype=np.int64),
        val_style_indices=np.array([1, 3], dtype=np.int64),
    )

    args = argparse.Namespace(exec_path=str(bundle_path), ref_path=None, val_ratio=0.25, seed=0)

    train_set, val_set, split_info = _build_datasets(args)

    assert len(train_set) == 6
    assert len(val_set) == 6
    assert split_info["split_mode"] == "style"
    assert split_info["train_style_count"] == 2
    assert split_info["val_style_count"] == 2
    assert {train_set[i]["style_idx"].item() for i in range(len(train_set))} == {0, 2}
    assert {val_set[i]["style_idx"].item() for i in range(len(val_set))} == {1, 3}


def test_train_logs_epoch_metrics_to_wandb(tmp_path, monkeypatch):
    ref = np.zeros((4, 8, 3), dtype=np.float32)
    exe = ref.copy()
    exe[:, :, 1] = 0.1
    ref_path = tmp_path / "ref.npy"
    exec_path = tmp_path / "exec.npy"
    output = tmp_path / "herm.pt"
    np.save(ref_path, ref)
    np.save(exec_path, exe)

    calls = {"init": [], "log": [], "finish": 0}

    class FakeRun:
        def log(self, values, step=None):
            calls["log"].append((values, step))

        def finish(self):
            calls["finish"] += 1

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: calls["init"].append(kwargs) or FakeRun(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    args = argparse.Namespace(
        ref_path=str(ref_path),
        exec_path=str(exec_path),
        output=str(output),
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-4,
        dt=0.5,
        num_poses=8,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        val_ratio=0.25,
        seed=0,
        device="cpu",
        wandb=True,
        wandb_project="herm-test",
        wandb_entity=None,
        wandb_run_name="unit-test",
    )

    train(args)

    assert calls["init"]
    assert calls["init"][0]["project"] == "herm-test"
    assert calls["log"]
    logged, step = calls["log"][0]
    assert step == 1
    assert "train/loss" in logged
    assert "val/loss" in logged
    assert "val/best_pos_l1" in logged
    assert calls["finish"] == 1
