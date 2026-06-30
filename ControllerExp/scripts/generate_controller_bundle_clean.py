#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = ROOT / "ControllerExp/scripts/genTest.py"


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("worldcap_legacy_genTest", LEGACY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy_module()


StyleSpec = LEGACY.StyleSpec
TARGET_LEN = LEGACY.TARGET_LEN
TRACKER_SUBSTYLES = LEGACY.TRACKER_SUBSTYLES


def resample_anchors_to_len(anchors: np.ndarray, target_len: int) -> np.ndarray:
    batch, _, dim = anchors.shape
    if dim < 3:
        raise ValueError(f"Expected anchors last dim >= 3, got {anchors.shape}")
    out = np.zeros((batch, target_len, 3), dtype=anchors.dtype)
    for i in range(batch):
        out[i] = LEGACY._resample_traj_xyz(anchors[i, :, :3], target_len)
    return out


def downsample_anchors(anchors: np.ndarray, target_len: int) -> np.ndarray:
    num_steps = anchors.shape[1]
    idx = np.linspace(0, num_steps - 1, target_len, dtype=int)
    return anchors[:, idx, :3].astype(np.float32)


def compute_ref_payload(
    source_ref_8: np.ndarray,
    ref_41: np.ndarray,
    target_len: int,
    ref_payload_mode: str,
) -> np.ndarray:
    mode = (ref_payload_mode or "source_ref").strip().lower()
    if mode == "source_ref":
        return source_ref_8[:, :, :3].astype(np.float32)
    if mode == "roundtrip_ref":
        return downsample_anchors(ref_41, target_len)
    raise ValueError(f"Unknown ref_payload_mode={ref_payload_mode!r}")


def summarize_bundle_stats(
    source_ref: np.ndarray,
    bundle_ref: np.ndarray,
    exec_trajs: np.ndarray,
    train_style_indices: np.ndarray,
    val_style_indices: np.ndarray,
) -> Dict[str, Any]:
    return {
        "source_ref_shape": tuple(source_ref.shape),
        "bundle_ref_shape": tuple(bundle_ref.shape),
        "exec_trajs_shape": tuple(exec_trajs.shape),
        "num_styles": int(exec_trajs.shape[0]),
        "train_styles": int(len(train_style_indices)),
        "val_styles": int(len(val_style_indices)),
        "ref_max_abs_diff_vs_source": float(np.max(np.abs(bundle_ref - source_ref))),
        "ref_mean_abs_diff_vs_source": float(np.mean(np.abs(bundle_ref - source_ref))),
    }


def build_bundle(
    anchors_path: Path,
    out_path: Path,
    ref_payload_mode: str,
    style_seed: int,
) -> Dict[str, Any]:
    np.random.seed(style_seed)
    import random

    random.seed(style_seed)

    source_ref = np.load(str(anchors_path)).astype(np.float32)
    if source_ref.ndim != 3 or source_ref.shape[-1] < 3:
        raise ValueError(f"Expected anchors [N,T,>=3], got {source_ref.shape}")

    if source_ref.shape[1] != 41:
        ref_41 = resample_anchors_to_len(source_ref, 41)
    else:
        ref_41 = source_ref[:, :, :3].astype(np.float32)

    proposal_sampling = LEGACY.TrajectorySampling(time_horizon=4.0, interval_length=0.1)
    initial_state = LEGACY.ego_from_anchor_pair(ref_41[0, 0, :3], ref_41[0, 1, :3])
    proposal_states = LEGACY._build_global_proposal_states(ref_41[:, :, :3].astype(np.float32), initial_state)

    specs: List[StyleSpec] = LEGACY.build_style_specs()
    train_style_indices, val_style_indices = LEGACY.split_train_val(specs)

    all_execs: List[np.ndarray] = []
    style_meta: List[Dict[str, Any]] = []
    for kind_id, spec in enumerate(specs):
        simulator = LEGACY.PDMSimulator(
            proposal_sampling,
            tracker_style="default",
            post_style=spec.post_style,
            post_params=spec.post_params,
            tracker_params=spec.tracker_params,
        )
        exec_41 = simulator.simulate_proposals(proposal_states, initial_state)
        exec_ego = LEGACY._global_to_ego_xyyaw_all(exec_41, initial_state)
        exec_8 = downsample_anchors(exec_ego, TARGET_LEN)
        all_execs.append(exec_8)
        style_meta.append(
            {
                "kind_id": int(kind_id),
                "style": str(spec.style_name),
                "lqr": (spec.tracker_params or dict(TRACKER_SUBSTYLES["default"]["base"])),
                "post": spec.post_params,
                "post_style": str(spec.post_style),
            }
        )

    exec_trajs = np.stack(all_execs).astype(np.float32)
    bundle_ref = compute_ref_payload(
        source_ref_8=source_ref[:, :, :3].astype(np.float32),
        ref_41=ref_41.astype(np.float32),
        target_len=TARGET_LEN,
        ref_payload_mode=ref_payload_mode,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        exec_trajs=exec_trajs,
        ref_traj=bundle_ref,
        style_names=np.array([m["style"] for m in style_meta], dtype=object),
        lqr_params=np.array([m["lqr"] for m in style_meta], dtype=object),
        post_params=np.array([m["post"] for m in style_meta], dtype=object),
        style_kind=np.array([s.kind for s in specs], dtype=object),
        style_group_big=np.array([s.group_big for s in specs], dtype=object),
        style_group_sub=np.array([s.group_sub for s in specs], dtype=object),
        train_style_indices=train_style_indices,
        val_style_indices=val_style_indices,
        seed=np.array([style_seed], dtype=np.int64),
        version=np.array(["clean_bundle_generator_v1"], dtype=object),
        source_anchors_path=np.array([str(anchors_path)], dtype=object),
        ref_payload_mode=np.array([ref_payload_mode], dtype=object),
    )

    return summarize_bundle_stats(
        source_ref=source_ref[:, :, :3].astype(np.float32),
        bundle_ref=bundle_ref,
        exec_trajs=exec_trajs,
        train_style_indices=train_style_indices,
        val_style_indices=val_style_indices,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean controller bundle for WorldCAP rebuttal experiments.")
    parser.add_argument("--anchors-path", required=True, help="Path to source centered reference trajectories (.npy).")
    parser.add_argument("--out-path", required=True, help="Path to output bundle (.npz).")
    parser.add_argument(
        "--ref-payload-mode",
        default="source_ref",
        choices=["source_ref", "roundtrip_ref"],
        help="How to store bundle ref_traj: original source_ref or simulator roundtrip_ref.",
    )
    parser.add_argument("--style-seed", type=int, default=42, help="Seed for controller style sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = build_bundle(
        anchors_path=Path(args.anchors_path).expanduser().resolve(),
        out_path=Path(args.out_path).expanduser().resolve(),
        ref_payload_mode=args.ref_payload_mode,
        style_seed=int(args.style_seed),
    )

    print("[OK] bundle generated")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
