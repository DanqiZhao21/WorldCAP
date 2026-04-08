#!/usr/bin/env python3
"""Save a 6-camera mosaic (2x3) for a given scenario token.

Layout (2 rows x 3 cols):
  Row 1: cam front-left (cam_l0), cam front (cam_f0), cam front-right (cam_r0)
  Row 2: cam back-left  (cam_l2), cam back  (cam_b0), cam back-right  (cam_r2)

This script intentionally mirrors the dataset/path handling patterns used in
tool/smalltool/visualize_token_compare.py, but does NOT require model checkpoints.

2026-03 update:
- Output SVG only (PNG is embedded as base64 inside an SVG container).
- Output directory prefers an existing subdir whose name starts with the token
    (e.g. token--good); otherwise creates <out_root>/<token>/.
- Output filename is deterministic (<token>_cam6.svg) and will overwrite if exists.

Typical usage:
  python tool/smalltool/visualize_token_cameras_6view.py \
    --token 9b16e4fea2c25446 \
    --split val \
    --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/6camScenarios
#FIXME: COLLISION
  python tool/smalltool/visualize_token_cameras_6view.py \
        --tokens 25226305bdff5efd 7b8d8487706f501e bc91cb648d525c6c 055c41d3c8e75bdc 9b16e4fea2c25446 ae06592110305073 98c7a48dd75052ab 4c22cdcc527e5a36 432598c0bda65445 198bc5f3280e52cd 3d4616d64a4c5f53 431869f33ace51a0 5a80299213875068 7ae00644dbef537f 885f450f0b875861 8fb11d5808355072 b77b4b6eb149553f b87ed2985e545397 c5ff90667143574a f1dabe118a6955d6 b66db24cee7957a2 9164913b818a58b1 667afe1f010351c5 79a151c333745253 d1bae9e7d9785598 \
        --split val \
        --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault
#FIXME: drivable
  python tool/smalltool/visualize_token_cameras_6view.py \
    --tokens 12f5053055935463 2be2e48e80985bee 75f5cc1f425c501d 8164612a623156ae 891fe2fc30c95109 9c9b62965da75764 a9c368a110585b1a b50b8f11d75a5cb0 f4814e7eb01252b6 fb42b4cbf6b95b6c 32a19a2ab50f563d 0defa8939c9851ea ea21c4b17b865a4d 2751f9eb641455c9 0558c7a64ef157b1 4909b88b347c5764 13fa82d6564e5bac 6950076b024c51db aaa11cdbc8d35178 bb9c441a4c2b5791 1e9a42ef8f4057a6 d37e71f395695c6f 778219f3cac65d35 f99992756cbb5adb 8364af67153a5193 62462203db6b5ba5 1039e136e6605cfb e03da8beb33a5e06 b113e988ede45a4f 87b32a1aeeb85613 87d3c1135ac85583 c87fe1d7a3bd57cb 5c5d006eb7b854c3 bc2314763cfc545d ea34282dc63d5a9a cb5022a3bef557e3 6c28c001109f5718 91fc00df56ae5aca c2ed826b31065c66 602e1bc4f8575d4e ce3d0bc0b2d55876 1da6196444e35b0f fc9b5914f47e58fe c93a302d2fb2508f 406f8b299de35ce2 6d0a7f0bb4e7584b 0c91824ce1e65b6e be85b447a33b59f8 d5fc95fa66025d7a 012432bd62b85f80 e13b89a8813159df 8183cdc6ff5a5726 f9c8ec6aefc05be4 14b6e7ce317d531f 759af2e479de5bbb \
    --split val \
    --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_drivable
    
  python /home/zhaodanqi/clone/WoTE/tool/smalltool/visualize_token_cameras_6view.py \
    --tokens c3ace87d2f985eaa \
    --split val \
    --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault


Environment variables (same conventions as other tools):
- OPENSCENE_DATA_ROOT: contains navsim_logs/ and sensor_blobs/
- NAVSIM_EXP_ROOT:     used for default output + metric_cache (optional)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import base64
import io

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    p.add_argument(
        "--token",
        nargs="+",
        default=None,
        help="scenario token(s) (one or more). For batches prefer --tokens or --token-file.",
    )
    p.add_argument(
        "--tokens",
        nargs="+",
        default=None,
        help="batch tokens to visualize (space-separated)",
    )
    p.add_argument(
        "--token-file",
        default=None,
        help="text file containing tokens (one per line; supports # comments)",
    )
    p.add_argument(
        "--split",
        default="test",
        help="dataset split (test/trainval/mini/val). If token belongs to a different split, script may auto-detect when metric_cache is available.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="output root dir (default: $NAVSIM_EXP_ROOT/token_cam6)",
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
        # Best-effort conversion
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

    # You can manually tweak these in code.
    # x, y = 12, 10
    x, y = 45, 30
    # outline_color = (255, 105, 180)  # hot pink
    # outline_color = "#6666F8"  # hot pink
    outline_color = "#F273A3"  # hot pink
    text_color = (255, 255, 255)
    outline_px = 3

    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Manual outline (works across Pillow versions)
    for dx in range(-outline_px, outline_px + 1):
        for dy in range(-outline_px, outline_px + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    draw.text((x, y), text, font=font, fill=text_color)


def _build_cam6_mosaic(*, frame):
    """Create a seamless 2x3 mosaic and return a PIL Image."""

    import numpy as np
    from PIL import Image

    # Required camera order (as requested)
    cams = [
        (frame.cameras.cam_l0, "CAM_FRONT_LEFT", False),
        (frame.cameras.cam_f0, "CAM_FRONT", False),
        (frame.cameras.cam_r0, "CAM_FRONT_RIGHT", False),
        (frame.cameras.cam_l2, "CAM_BACK_LEFT", True),
        (frame.cameras.cam_b0, "CAM_BACK", True),
        (frame.cameras.cam_r2, "CAM_BACK_RIGHT", True),
    ]

    imgs = []
    labels = []
    for cam, label, flip_back in cams:
        arr = _camera_to_uint8_rgb(cam)
        if flip_back:
            arr = _flip_lr(arr)
        imgs.append(arr)
        labels.append(label)

    # Normalize sizes (seamless paste requires identical tile size)
    h0, w0 = imgs[0].shape[:2]
    pil_tiles = []
    for arr in imgs:
        im = Image.fromarray(arr)
        if im.size != (w0, h0):
            im = im.resize((w0, h0), resample=Image.BILINEAR)
        pil_tiles.append(im)

    # Font size (tweakable). Use relative size to remain readable for papers.
    # font_size = max(18, int(min(w0, h0) * 0.055))
    # font_size = max(36, int(min(w0, h0) * 0.055))
    font_size = 80

    # Draw labels on each tile
    for im, label in zip(pil_tiles, labels):
        _draw_label(im, label, font_size=font_size)

    canvas = Image.new("RGB", (w0 * 3, h0 * 2))

    # Paste: row 0
    canvas.paste(pil_tiles[0], (0 * w0, 0 * h0))
    canvas.paste(pil_tiles[1], (1 * w0, 0 * h0))
    canvas.paste(pil_tiles[2], (2 * w0, 0 * h0))
    # Paste: row 1
    canvas.paste(pil_tiles[3], (0 * w0, 1 * h0))
    canvas.paste(pil_tiles[4], (1 * w0, 1 * h0))
    canvas.paste(pil_tiles[5], (2 * w0, 1 * h0))

    return canvas


def _save_pil_as_embedded_svg(*, pil_img, out_path: Path) -> None:
    """Save a PIL image as an SVG file by embedding a PNG data URI."""

    from PIL import Image

    if not isinstance(pil_img, Image.Image):
        raise TypeError(f"expected PIL.Image.Image, got: {type(pil_img)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = pil_img.size

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")

    svg = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n"
        f"  <image width=\"{width}\" height=\"{height}\" href=\"data:image/png;base64,{data}\"/>\n"
        "</svg>\n"
    )
    out_path.write_text(svg, encoding="utf-8")


def _resolve_token_out_dir(out_root: Path, token: str) -> Path:
    """Prefer an existing out_root/<token*> directory; otherwise use out_root/<token>."""

    try:
        if out_root.exists():
            candidates = [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith(token)]
            if candidates:
                candidates = sorted(candidates, key=lambda p: p.name)
                return candidates[0].resolve()
    except Exception:
        pass
    return (out_root / token).resolve()


def main() -> int:
    args = _parse_args()
    _ensure_local_devkit_on_path()

    # Collect tokens (single/list/file).
    tokens_raw = []
    if args.token:
        tokens_raw.extend([str(t) for t in args.token])
    if args.tokens:
        # Convenience: allow `--tokens /path/to/tokens.txt`.
        if len(args.tokens) == 1:
            maybe_path = Path(str(args.tokens[0])).expanduser()
            if maybe_path.suffix.lower() in {".txt", ".list"} and maybe_path.is_file():
                try:
                    for line in maybe_path.read_text().splitlines():
                        s = line.strip()
                        if not s or s.startswith("#"):
                            continue
                        tokens_raw.append(s)
                except Exception as e:
                    print(f"ERROR: failed to read --tokens file: {maybe_path} ({e})", file=sys.stderr)
                    return 2
            else:
                tokens_raw.extend([str(t) for t in args.tokens])
        else:
            tokens_raw.extend([str(t) for t in args.tokens])
    if args.token_file:
        p = Path(args.token_file).expanduser().resolve()
        if not p.exists():
            print(f"ERROR: --token-file not found: {p}", file=sys.stderr)
            return 2
        try:
            for line in p.read_text().splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                tokens_raw.append(s)
        except Exception as e:
            print(f"ERROR: failed to read --token-file: {p} ({e})", file=sys.stderr)
            return 2

    tokens: list[str] = []
    seen = set()
    for t in tokens_raw:
        s = (t or "").strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            tokens.append(s)

    if not tokens:
        print(
            "ERROR: no tokens provided. Use --token <TOK...>, or --tokens <TOK1> <TOK2>..., or --token-file <path>.",
            file=sys.stderr,
        )
        return 2

    # Output root
    out_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_root is None:
        navsim_exp_root = (os.environ.get("NAVSIM_EXP_ROOT") or "").strip()
        if not navsim_exp_root:
            print("ERROR: --out-dir is required if NAVSIM_EXP_ROOT is not set", file=sys.stderr)
            return 2
        out_root = (Path(navsim_exp_root) / "token_cam6").resolve()

    failures = 0
    for token in tokens:
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

        # Optional: infer log_name and adjust split to avoid scanning all splits.
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
            print(f"[ERR] navsim_log_path not found: {navsim_log_path}", file=sys.stderr)
            failures += 1
            continue
        if not sensor_blobs_path.exists():
            print(f"[ERR] sensor_blobs_path not found: {sensor_blobs_path}", file=sys.stderr)
            failures += 1
            continue

        out_dir = _resolve_token_out_dir(out_root, token)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{token}_cam6.svg"

        try:
            # Load scene with only 1 frame (history=1, future=0) and only required cameras.
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
                cam_l1=False,
                cam_l2=True,
                cam_r0=True,
                cam_r1=False,
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
                raise RuntimeError(f"token not found after filtering: {token}")

            scene = scene_loader.get_scene_from_token(token)
            frame = scene.frames[scene.scene_metadata.num_history_frames - 1]

            canvas = _build_cam6_mosaic(frame=frame)
            _save_pil_as_embedded_svg(pil_img=canvas, out_path=out_path)

            print("=== token_cam6 done ===")
            print(f"token: {token}")
            print(f"split: {split}")
            print(f"out:   {out_path}")
        except Exception as e:
            failures += 1
            print(f"[ERR] token failed: {token} ({e})", file=sys.stderr)
            continue

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
