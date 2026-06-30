import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path("/home/zhaodanqi/clone/WoTE/ControllerExp/scripts/generate_controller_bundle_clean.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_controller_bundle_clean", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_ref_payload_source_preserves_original_ref():
    mod = _load_module()

    src_ref = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.5, 0.1], [2.0, 1.0, 0.2], [3.0, 1.5, 0.3]],
            [[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [2.8, 0.4, -0.1], [4.1, 0.6, -0.2]],
        ],
        dtype=np.float32,
    )

    payload = mod.compute_ref_payload(
        source_ref_8=src_ref,
        ref_41=src_ref,
        target_len=4,
        ref_payload_mode="source_ref",
    )

    assert payload.shape == src_ref.shape
    assert np.array_equal(payload, src_ref)


def test_compute_ref_payload_roundtrip_matches_resample_then_downsample():
    mod = _load_module()

    src_ref = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [2.2, 0.1, 0.2], [3.1, 0.2, 0.3]],
            [[0.0, 0.0, 0.0], [0.8, -0.2, -0.1], [1.6, -0.3, -0.2], [2.5, -0.4, -0.3]],
        ],
        dtype=np.float32,
    )

    ref_41 = mod.resample_anchors_to_len(src_ref, 7)
    payload = mod.compute_ref_payload(
        source_ref_8=src_ref,
        ref_41=ref_41,
        target_len=4,
        ref_payload_mode="roundtrip_ref",
    )
    expected = mod.downsample_anchors(ref_41, 4)

    assert payload.shape == src_ref.shape
    assert np.allclose(payload, expected)


def test_summarize_bundle_stats_reports_ref_delta():
    mod = _load_module()

    source_ref = np.zeros((2, 4, 3), dtype=np.float32)
    bundle_ref = np.ones((2, 4, 3), dtype=np.float32)
    exec_trajs = np.zeros((3, 2, 4, 3), dtype=np.float32)
    train_idx = np.array([0, 1], dtype=np.int64)
    val_idx = np.array([2], dtype=np.int64)

    stats = mod.summarize_bundle_stats(
        source_ref=source_ref,
        bundle_ref=bundle_ref,
        exec_trajs=exec_trajs,
        train_style_indices=train_idx,
        val_style_indices=val_idx,
    )

    assert stats["source_ref_shape"] == (2, 4, 3)
    assert stats["bundle_ref_shape"] == (2, 4, 3)
    assert stats["exec_trajs_shape"] == (3, 2, 4, 3)
    assert stats["num_styles"] == 3
    assert stats["train_styles"] == 2
    assert stats["val_styles"] == 1
    assert stats["ref_max_abs_diff_vs_source"] == 1.0

