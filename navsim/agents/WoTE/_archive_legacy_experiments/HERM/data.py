from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _validate_traj_array(name: str, value: np.ndarray, ndim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != ndim or arr.shape[-1] < 3:
        raise ValueError(f"{name} must have shape {'[N,T,3]' if ndim == 3 else '[S,N,T,3]'}, got {arr.shape}")
    return arr[..., :3]


def load_controller_pairs(ref_path: Optional[str], exec_path: str) -> Dict[str, np.ndarray]:
    """Load controller trajectory pairs from .npy banks or .npz style bundles."""
    if exec_path is None:
        raise ValueError("exec_path is required")

    if str(exec_path).endswith(".npz"):
        data = np.load(exec_path, allow_pickle=True)
        if "ref_traj" not in data or "exec_trajs" not in data:
            raise ValueError(f"{exec_path} must contain ref_traj and exec_trajs")
        ref = _validate_traj_array("ref_traj", data["ref_traj"], ndim=3)
        exe = _validate_traj_array("exec_trajs", data["exec_trajs"], ndim=4)
        if exe.shape[1:] != ref.shape:
            raise ValueError(f"bundle shape mismatch: ref={ref.shape}, exec={exe.shape}")

        num_styles, num_traj, horizon, dim = exe.shape
        plan = np.broadcast_to(ref[None, ...], (num_styles, num_traj, horizon, dim)).reshape(-1, horizon, dim)
        executed = exe.reshape(-1, horizon, dim)
        style_idx = np.repeat(np.arange(num_styles, dtype=np.int64), num_traj)

        pairs: Dict[str, np.ndarray] = {
            "plan_traj": np.asarray(plan, dtype=np.float32),
            "exec_traj": np.asarray(executed, dtype=np.float32),
            "style_idx": style_idx,
        }
        if "style_names" in data:
            style_names = np.asarray(data["style_names"]).astype(str)
            if style_names.shape[0] == num_styles:
                pairs["style_name"] = np.repeat(style_names, num_traj)
        return pairs

    if ref_path is None:
        raise ValueError("ref_path is required when exec_path is not a .npz bundle")

    ref = _validate_traj_array("ref_traj", np.load(ref_path), ndim=3)
    exe = _validate_traj_array("exec_traj", np.load(exec_path), ndim=3)
    if ref.shape != exe.shape:
        raise ValueError(f"shape mismatch: ref={ref.shape}, exec={exe.shape}")

    return {
        "plan_traj": ref.astype(np.float32, copy=False),
        "exec_traj": exe.astype(np.float32, copy=False),
        "style_idx": np.zeros((ref.shape[0],), dtype=np.int64),
    }


class ControllerTrajectoryDataset(Dataset):
    """Dataset of planned/executed trajectory pairs for standalone HERM training."""

    def __init__(
        self,
        ref_path: Optional[str],
        exec_path: str,
        normalize: bool = False,
        style_indices: Optional[Iterable[int]] = None,
    ) -> None:
        if normalize:
            raise ValueError("normalize=True is not implemented for HERM trajectory pairs")

        pairs = load_controller_pairs(ref_path, exec_path)
        if style_indices is not None:
            allowed = np.asarray(list(style_indices), dtype=np.int64)
            mask = np.isin(pairs["style_idx"], allowed)
            pairs = {
                key: value[mask] if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0] else value
                for key, value in pairs.items()
            }

        self.plan_traj = torch.as_tensor(pairs["plan_traj"], dtype=torch.float32)
        self.exec_traj = torch.as_tensor(pairs["exec_traj"], dtype=torch.float32)
        self.style_idx = torch.as_tensor(pairs["style_idx"], dtype=torch.long)
        self.style_name = pairs.get("style_name", None)

        if self.plan_traj.shape != self.exec_traj.shape:
            raise ValueError(
                f"plan and exec tensors must have same shape, got {self.plan_traj.shape} and {self.exec_traj.shape}"
            )

    def __len__(self) -> int:
        return int(self.plan_traj.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = {
            "plan_traj": self.plan_traj[index],
            "exec_traj": self.exec_traj[index],
            "style_idx": self.style_idx[index],
        }
        if self.style_name is not None:
            sample["style_name"] = self.style_name[index]
        return sample


class SupportQueryTrajectoryDataset(Dataset):
    """Style episodes with support pairs and query trajectories."""

    def __init__(
        self,
        exec_path: str,
        support_size: int = 192,
        query_size: int = 64,
        style_indices: Optional[Iterable[int]] = None,
        seed: int = 0,
    ) -> None:
        if not str(exec_path).endswith(".npz"):
            raise ValueError("SupportQueryTrajectoryDataset requires a .npz controller bundle")
        data = np.load(exec_path, allow_pickle=True)
        if "ref_traj" not in data or "exec_trajs" not in data:
            raise ValueError(f"{exec_path} must contain ref_traj and exec_trajs")

        self.ref_traj = torch.as_tensor(_validate_traj_array("ref_traj", data["ref_traj"], ndim=3))
        self.exec_trajs = torch.as_tensor(_validate_traj_array("exec_trajs", data["exec_trajs"], ndim=4))
        if tuple(self.exec_trajs.shape[1:]) != tuple(self.ref_traj.shape):
            raise ValueError(f"bundle shape mismatch: ref={tuple(self.ref_traj.shape)} exec={tuple(self.exec_trajs.shape)}")

        num_styles = int(self.exec_trajs.shape[0])
        num_traj = int(self.ref_traj.shape[0])
        total = int(support_size) + int(query_size)
        if total > num_traj:
            raise ValueError(f"support_size + query_size must be <= {num_traj}, got {total}")
        self.support_size = int(support_size)
        self.query_size = int(query_size)
        self.seed = int(seed)
        self.num_traj = num_traj

        if style_indices is None:
            self.style_indices = list(range(num_styles))
        else:
            self.style_indices = [int(idx) for idx in style_indices]
            if any(idx < 0 or idx >= num_styles for idx in self.style_indices):
                raise ValueError(f"style_indices out of range for {num_styles} styles")

        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + int(epoch))
        perm = rng.permutation(self.num_traj)
        total = self.support_size + self.query_size
        self.support_indices = torch.as_tensor(perm[: self.support_size], dtype=torch.long)
        self.query_indices = torch.as_tensor(perm[self.support_size : total], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.style_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        style_idx = self.style_indices[index]
        support_plan = self.ref_traj.index_select(0, self.support_indices)
        query_plan = self.ref_traj.index_select(0, self.query_indices)
        support_exec = self.exec_trajs[style_idx].index_select(0, self.support_indices)
        query_exec = self.exec_trajs[style_idx].index_select(0, self.query_indices)
        return {
            "support_plan_traj": support_plan,
            "support_exec_traj": support_exec,
            "query_plan_traj": query_plan,
            "query_exec_traj": query_exec,
            "style_idx": torch.tensor(style_idx, dtype=torch.long),
        }
