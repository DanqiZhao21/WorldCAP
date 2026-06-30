import numpy as np
import torch

from navsim.agents.WoTE.HERM.data import (
    ControllerTrajectoryDataset,
    SupportQueryTrajectoryDataset,
    load_controller_pairs,
)


def test_load_controller_pairs_from_npy(tmp_path):
    ref = np.zeros((3, 8, 3), dtype=np.float32)
    exe = np.ones((3, 8, 3), dtype=np.float32)
    ref_path = tmp_path / "ref.npy"
    exec_path = tmp_path / "exec.npy"
    np.save(ref_path, ref)
    np.save(exec_path, exe)

    pairs = load_controller_pairs(str(ref_path), str(exec_path))

    assert pairs["plan_traj"].shape == (3, 8, 3)
    assert pairs["exec_traj"].shape == (3, 8, 3)
    assert pairs["style_idx"].shape == (3,)
    assert pairs["style_idx"].tolist() == [0, 0, 0]


def test_load_controller_pairs_from_npz_bundle(tmp_path):
    ref = np.zeros((4, 8, 3), dtype=np.float32)
    exe = np.ones((2, 4, 8, 3), dtype=np.float32)
    bundle_path = tmp_path / "bundle.npz"
    np.savez(bundle_path, ref_traj=ref, exec_trajs=exe, style_names=np.array(["a", "b"]))

    pairs = load_controller_pairs(None, str(bundle_path))

    assert pairs["plan_traj"].shape == (8, 8, 3)
    assert pairs["exec_traj"].shape == (8, 8, 3)
    assert pairs["style_idx"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert pairs["style_name"].tolist() == ["a", "a", "a", "a", "b", "b", "b", "b"]


def test_controller_trajectory_dataset_returns_tensors(tmp_path):
    ref = np.zeros((2, 8, 3), dtype=np.float32)
    exe = np.ones((2, 8, 3), dtype=np.float32)
    ref_path = tmp_path / "ref.npy"
    exec_path = tmp_path / "exec.npy"
    np.save(ref_path, ref)
    np.save(exec_path, exe)

    dataset = ControllerTrajectoryDataset(str(ref_path), str(exec_path))
    sample = dataset[0]

    assert len(dataset) == 2
    assert isinstance(sample["plan_traj"], torch.Tensor)
    assert isinstance(sample["exec_traj"], torch.Tensor)
    assert sample["plan_traj"].shape == (8, 3)
    assert sample["style_idx"].item() == 0


def test_controller_trajectory_dataset_filters_styles(tmp_path):
    ref = np.zeros((3, 8, 3), dtype=np.float32)
    exe = np.ones((2, 3, 8, 3), dtype=np.float32)
    bundle_path = tmp_path / "bundle.npz"
    np.savez(bundle_path, ref_traj=ref, exec_trajs=exe)

    dataset = ControllerTrajectoryDataset(None, str(bundle_path), style_indices=[1])

    assert len(dataset) == 3
    assert all(dataset[i]["style_idx"].item() == 1 for i in range(len(dataset)))


def test_support_query_dataset_returns_style_episode(tmp_path):
    ref = np.zeros((6, 8, 3), dtype=np.float32)
    exe = np.zeros((3, 6, 8, 3), dtype=np.float32)
    for style in range(3):
        exe[style, :, :, 1] = float(style)
    bundle_path = tmp_path / "bundle.npz"
    np.savez(bundle_path, ref_traj=ref, exec_trajs=exe)

    dataset = SupportQueryTrajectoryDataset(
        str(bundle_path),
        style_indices=[2],
        support_size=4,
        query_size=2,
        seed=0,
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["support_plan_traj"].shape == (4, 8, 3)
    assert sample["support_exec_traj"].shape == (4, 8, 3)
    assert sample["query_plan_traj"].shape == (2, 8, 3)
    assert sample["query_exec_traj"].shape == (2, 8, 3)
    assert sample["style_idx"].item() == 2
    assert torch.allclose(sample["support_exec_traj"][..., 1], torch.full((4, 8), 2.0))


def test_support_query_dataset_resamples_by_reproducible_epoch(tmp_path):
    ref = np.zeros((10, 8, 3), dtype=np.float32)
    ref[:, :, 0] = np.arange(10, dtype=np.float32)[:, None]
    exe = np.broadcast_to(ref[None, ...], (1, 10, 8, 3)).copy()
    bundle_path = tmp_path / "bundle.npz"
    np.savez(bundle_path, ref_traj=ref, exec_trajs=exe)

    first = SupportQueryTrajectoryDataset(str(bundle_path), support_size=6, query_size=4, seed=7)
    second = SupportQueryTrajectoryDataset(str(bundle_path), support_size=6, query_size=4, seed=7)

    first.set_epoch(3)
    second.set_epoch(3)
    first_epoch3 = first[0]["support_plan_traj"][:, 0, 0].clone()
    second_epoch3 = second[0]["support_plan_traj"][:, 0, 0].clone()

    first.set_epoch(4)
    first_epoch4 = first[0]["support_plan_traj"][:, 0, 0].clone()

    assert torch.equal(first_epoch3, second_epoch3)
    assert not torch.equal(first_epoch3, first_epoch4)
