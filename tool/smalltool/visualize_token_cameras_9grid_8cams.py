#!/usr/bin/env python3
"""Save an 8-camera mosaic in a 3x3 grid with the center left blank.

Layout (3 rows x 3 cols), center is empty:
  Row 1: cam front-left (cam_l0), cam front (cam_f0), cam front-right (cam_r0)
  Row 2: cam left (cam_l1),       EMPTY,              cam right (cam_r1)
  Row 3: cam back-left (cam_l2),  cam back (cam_b0),   cam back-right (cam_r2)

Note:
- Back row (cam_l2/cam_b0/cam_r2) is mirrored horizontally for left-right symmetry.
- This script follows the dataset/path handling patterns used in
  tool/smalltool/visualize_token_cameras_6view.py.

Typical usage:
  python tool/smalltool/visualize_token_cameras_9grid_8cams.py \
    --token 1e9a42ef8f4057a6 \
    --split val \
    --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/9gridScenarios_transparent

Environment variables (same conventions as other tools):
- OPENSCENE_DATA_ROOT: contains navsim_logs/ and sensor_blobs/
- NAVSIM_EXP_ROOT:     used for default output + metric_cache (optional)
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Optional


def _ensure_local_devkit_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    navsim_pkg = repo_root / "navsim"
    nuplan_pkg = repo_root / "nuplan-devkit"

    for p in [repo_root, navsim_pkg, nuplan_pkg]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")


# Headless backend (servers/ssh)
import matplotlib

matplotlib.use("Agg")


def _default_navsim_log_path(split: str) -> Optional[str]:
    root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
    if not root:
        return None
    base = Path(root) / "navsim_logs"
    candidate = base / split
    if candidate.exists():
        return str(candidate)

    fallback_map = {"val": "trainval", "train": "trainval"}
    alt = fallback_map.get(split)
    if alt:
        alt_candidate = base / alt
        if alt_candidate.exists():
            print(
                f"[WARN] navsim_logs split '{split}' not found; falling back to '{alt}'.",
                file=sys.stderr,
            )
            return str(alt_candidate)

    return str(candidate)


def _default_sensor_blobs_path(split: str) -> Optional[str]:
    root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
    if not root:
        return None
    base = Path(root) / "sensor_blobs"
    candidate = base / split
    if candidate.exists():
        return str(candidate)

    fallback_map = {"val": "trainval", "train": "trainval"}
    alt = fallback_map.get(split)
    if alt:
        alt_candidate = base / alt
        if alt_candidate.exists():
            print(
                f"[WARN] sensor_blobs split '{split}' not found; falling back to '{alt}'.",
                file=sys.stderr,
            )
            return str(alt_candidate)

    return str(candidate)


def _default_metric_cache_path() -> Optional[str]:
    root = (os.environ.get("NAVSIM_EXP_ROOT") or "").strip()
    if not root:
        return None
    return str(Path(root) / "metric_cache")


def _get_metric_cache_file(metric_cache_path: Path, token: str) -> Path:
    from navsim.common.dataloader import MetricCacheLoader

    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    if token not in metric_cache_loader.metric_cache_paths:
        raise KeyError(f"token not found in metric cache: {token}")
    return Path(metric_cache_loader.metric_cache_paths[token])


def _infer_log_name_from_metric_cache_file(metric_cache_file: Path) -> Optional[str]:
    parts = list(metric_cache_file.parts)
    try:
        idx = parts.index("metric_cache")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except Exception:
        pass
    return None


def _detect_split_for_log(*, openscene_root: Path, preferred_split: str, log_name: str) -> Optional[str]:
    logs_root = openscene_root / "navsim_logs"
    if not logs_root.exists():
        return None

    candidates = []
    if preferred_split:
        candidates.append(preferred_split)
    for s in ["test", "trainval", "mini", "val", "train"]:
        if s not in candidates:
            candidates.append(s)
    for d in sorted([p.name for p in logs_root.iterdir() if p.is_dir()]):
        if d not in candidates:
            candidates.append(d)

    for split in candidates:
        p = logs_root / split / f"{log_name}.pkl"
        if p.exists():
            return split
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True, help="scenario token")
    p.add_argument(
        "--split",
        default="test",
        help="dataset split (test/trainval/mini/val). If token belongs to a different split, script may auto-detect when metric_cache is available.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="output root dir (default: $NAVSIM_EXP_ROOT/token_cam9grid)",
    )
    p.add_argument(
        "--navsim-log-path",
        default=None,
        help="override navsim logs path (default: $OPENSCENE_DATA_ROOT/navsim_logs/<split>)",
    )
    p.add_argument(
        "--sensor-blobs-path",
        default=None,
        help="override sensor blobs path (default: $OPENSCENE_DATA_ROOT/sensor_blobs/<split>)",
    )
    p.add_argument(
        "--metric-cache-path",
        default=None,
        help="optional metric_cache dir (used only to infer log_name/split faster)",
    )
    return p.parse_args()


def _camera_to_uint8_rgb(cam) -> "object":
    """Return a HxWx3 uint8 RGB numpy array."""

    import numpy as np

    img = None
    if cam is not None:
        img = getattr(cam, "image", None)
    if img is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return arr


def _flip_lr(img) -> "object":
    import numpy as np

    arr = np.asarray(img)
    return np.ascontiguousarray(arr[:, ::-1])


def _draw_label(pil_img, text: str, *, font_size: int) -> None:
    """Draw white text with pink outline at top-left."""

    from PIL import ImageDraw, ImageFont

    # Keep consistent with the user's current 6-view settings.
    x, y = 45, 30
    outline_color = (255, 105, 180)  # hot pink
    text_color = (255, 255, 255)
    outline_px = 2

    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for dx in range(-outline_px, outline_px + 1):
        for dy in range(-outline_px, outline_px + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    draw.text((x, y), text, font=font, fill=text_color)


def _save_cam9grid_mosaic(*, frame, out_path: Path) -> None:
    """Create a seamless 3x3 mosaic (center empty) and save to disk."""

    import numpy as np
    from PIL import Image

    # 8 cameras, center is intentionally left blank (transparent).
    tiles = [
        (frame.cameras.cam_l0, "CAM_FRONT_LEFT", False),
        (frame.cameras.cam_f0, "CAM_FRONT", False),
        (frame.cameras.cam_r0, "CAM_FRONT_RIGHT", False),
        (frame.cameras.cam_l1, "CAM_LEFT", False),
        (None, "", False),
        (frame.cameras.cam_r1, "CAM_RIGHT", False),
        (frame.cameras.cam_l2, "CAM_BACK_LEFT", True),
        (frame.cameras.cam_b0, "CAM_BACK", True),
        (frame.cameras.cam_r2, "CAM_BACK_RIGHT", True),
    ]

    imgs = []
    labels = []
    for cam, label, flip_back in tiles:
        arr = _camera_to_uint8_rgb(cam)
        if flip_back:
            arr = _flip_lr(arr)
        imgs.append(arr)
        labels.append(label)

    h0, w0 = imgs[0].shape[:2]
    pil_tiles = []
    for arr, label in zip(imgs, labels):
        if label == "":
            im = Image.new("RGBA", (w0, h0), (0, 0, 0, 0))
        else:
            im = Image.fromarray(arr)
            if im.size != (w0, h0):
                im = im.resize((w0, h0), resample=Image.BILINEAR)
            im = im.convert("RGBA")
            _draw_label(im, label, font_size=80)
        pil_tiles.append(im)

    canvas = Image.new("RGBA", (w0 * 3, h0 * 3), (0, 0, 0, 0))

    for idx, im in enumerate(pil_tiles):
        r = idx // 3
        c = idx % 3
        canvas.paste(im, (c * w0, r * h0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    args = _parse_args()
    _ensure_local_devkit_on_path()

    token = args.token.strip()
    split = (args.split or "").strip()
    if split.lower() == "text":
        print("[WARN] split='text' looks like a typo; using 'test'.", file=sys.stderr)
        split = "test"

    navsim_log_path_str = args.navsim_log_path or _default_navsim_log_path(split)
    sensor_blobs_path_str = args.sensor_blobs_path or _default_sensor_blobs_path(split)

    if not navsim_log_path_str:
        print("ERROR: --navsim-log-path is required if OPENSCENE_DATA_ROOT is not set", file=sys.stderr)
        return 2
    if not sensor_blobs_path_str:
        print(
            "ERROR: --sensor-blobs-path is required if OPENSCENE_DATA_ROOT is not set",
            file=sys.stderr,
        )
        return 2

    navsim_log_path = Path(navsim_log_path_str).expanduser().resolve()
    sensor_blobs_path = Path(sensor_blobs_path_str).expanduser().resolve()

    log_name = None
    metric_cache_path_str = args.metric_cache_path or _default_metric_cache_path()
    if metric_cache_path_str:
        metric_cache_path = Path(metric_cache_path_str).expanduser().resolve()
        if metric_cache_path.exists():
            try:
                mc_file = _get_metric_cache_file(metric_cache_path, token)
                log_name = _infer_log_name_from_metric_cache_file(mc_file)
                openscene_root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
                if log_name and openscene_root:
                    detected = _detect_split_for_log(
                        openscene_root=Path(openscene_root),
                        preferred_split=split,
                        log_name=log_name,
                    )
                    if detected and detected != split:
                        print(
                            f"[WARN] token appears to belong to split '{detected}' (log={log_name}); overriding split '{split}' -> '{detected}'.",
                            file=sys.stderr,
                        )
                        split = detected
                        if args.navsim_log_path is None:
                            navsim_log_path = Path(_default_navsim_log_path(split)).expanduser().resolve()
                        if args.sensor_blobs_path is None:
                            sensor_blobs_path = Path(_default_sensor_blobs_path(split)).expanduser().resolve()
            except Exception as e:
                print(f"[WARN] split/log inference skipped: {e}", file=sys.stderr)

    if not navsim_log_path.exists():
        print(f"ERROR: navsim_log_path not found: {navsim_log_path}", file=sys.stderr)
        return 2
    if not sensor_blobs_path.exists():
        print(f"ERROR: sensor_blobs_path not found: {sensor_blobs_path}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_root is None:
        navsim_exp_root = (os.environ.get("NAVSIM_EXP_ROOT") or "").strip()
        if not navsim_exp_root:
            print("ERROR: --out-dir is required if NAVSIM_EXP_ROOT is not set", file=sys.stderr)
            return 2
        out_root = (Path(navsim_exp_root) / "token_cam9grid").resolve()

    out_dir = (out_root / token).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{token}_{ts}_cam9grid.png"

    # Load scene with only 1 frame (history=1, future=0) and all 8 cameras.
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig

    scene_filter = SceneFilter(
        num_history_frames=1,
        num_future_frames=0,
        frame_interval=1,
        has_route=False,
        max_scenes=1,
        log_names=[log_name] if log_name else None,
        tokens=[token],
    )
    sensor_config = SensorConfig(
        cam_f0=True,
        cam_l0=True,
        cam_l1=True,
        cam_l2=True,
        cam_r0=True,
        cam_r1=True,
        cam_r2=True,
        cam_b0=True,
        lidar_pc=False,
    )

    scene_loader = SceneLoader(
        data_path=navsim_log_path,
        sensor_blobs_path=sensor_blobs_path,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    if token not in scene_loader.tokens:
        print(f"ERROR: token not found after filtering: {token}", file=sys.stderr)
        return 2

    scene = scene_loader.get_scene_from_token(token)
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]

    _save_cam9grid_mosaic(frame=frame, out_path=out_path)

    print("=== token_cam9grid done ===")
    print(f"token: {token}")
    print(f"split: {split}")
    print(f"out:   {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
