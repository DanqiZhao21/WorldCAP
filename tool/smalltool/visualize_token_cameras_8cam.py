#!/usr/bin/env python3
"""Save 8 camera images for one or more scenario tokens.

This is a simplified version of the old 9-grid mosaic script:
- No mosaic stitching.
- For each token, saves 8 individual camera frames into a token-prefixed folder.
    If <out_root> already contains a subfolder whose name starts with the token
    (e.g. token--note), that folder is reused; otherwise <out_root>/<token>/ is created.

Outputs (per token):
    <token>_camF0.svg <token>_camL0.svg <token>_camR0.svg <token>_camL1.svg
    <token>_camR1.svg <token>_camL2.svg <token>_camB0.svg <token>_camR2.svg

Note:
- Camera images are raster; SVG output is an SVG container embedding a PNG via base64.

Environment variables (same conventions as other tools):
- OPENSCENE_DATA_ROOT: contains navsim_logs/ and sensor_blobs/
- NAVSIM_EXP_ROOT:     used for default output + metric_cache (optional)


#FIXME: COLLISION
  python /home/zhaodanqi/clone/WoTE/tool/smalltool/visualize_token_cameras_8cam.py \
        --tokens 25226305bdff5efd 7b8d8487706f501e bc91cb648d525c6c 055c41d3c8e75bdc 9b16e4fea2c25446 ae06592110305073 98c7a48dd75052ab 4c22cdcc527e5a36 432598c0bda65445 198bc5f3280e52cd 3d4616d64a4c5f53 431869f33ace51a0 5a80299213875068 7ae00644dbef537f 885f450f0b875861 8fb11d5808355072 b77b4b6eb149553f b87ed2985e545397 c5ff90667143574a f1dabe118a6955d6 b66db24cee7957a2 9164913b818a58b1 667afe1f010351c5 79a151c333745253 d1bae9e7d9785598 \
        --split val \
        --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault
        
#FIXME: drivable
  python /home/zhaodanqi/clone/WoTE/tool/smalltool/visualize_token_cameras_8cam.py \
    --tokens 12f5053055935463 2be2e48e80985bee 75f5cc1f425c501d 8164612a623156ae 891fe2fc30c95109 9c9b62965da75764 a9c368a110585b1a b50b8f11d75a5cb0 f4814e7eb01252b6 fb42b4cbf6b95b6c 32a19a2ab50f563d 0defa8939c9851ea ea21c4b17b865a4d 2751f9eb641455c9 0558c7a64ef157b1 4909b88b347c5764 13fa82d6564e5bac 6950076b024c51db aaa11cdbc8d35178 bb9c441a4c2b5791 1e9a42ef8f4057a6 d37e71f395695c6f 778219f3cac65d35 f99992756cbb5adb 8364af67153a5193 62462203db6b5ba5 1039e136e6605cfb e03da8beb33a5e06 b113e988ede45a4f 87b32a1aeeb85613 87d3c1135ac85583 c87fe1d7a3bd57cb 5c5d006eb7b854c3 bc2314763cfc545d ea34282dc63d5a9a cb5022a3bef557e3 6c28c001109f5718 91fc00df56ae5aca c2ed826b31065c66 602e1bc4f8575d4e ce3d0bc0b2d55876 1da6196444e35b0f fc9b5914f47e58fe c93a302d2fb2508f 406f8b299de35ce2 6d0a7f0bb4e7584b 0c91824ce1e65b6e be85b447a33b59f8 d5fc95fa66025d7a 012432bd62b85f80 e13b89a8813159df 8183cdc6ff5a5726 f9c8ec6aefc05be4 14b6e7ce317d531f 759af2e479de5bbb \
    --split val \
    --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_drivable


  python /home/zhaodanqi/clone/WoTE/tool/smalltool/visualize_token_cameras_8cam.py \
        --tokens c3ace87d2f985eaa \
        --split val \
        --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault


  python /home/zhaodanqi/clone/WoTE/tool/smalltool/visualize_token_compare.py \
        --token c3ace87d2f985eaa \
        --split val \
        --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
        --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
        --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault


"""

from __future__ import annotations

import argparse
import os
import sys
import base64
import io
from pathlib import Path
from typing import Iterable, Optional


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
        help="output root dir (default: $NAVSIM_EXP_ROOT/token_cam8)",
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


def _save_pil_as_embedded_svg(pil_img: "object", out_path: Path) -> None:
    """Save a PIL image into an SVG container embedding PNG bytes via base64."""

    from PIL import Image

    if not isinstance(pil_img, Image.Image):
        raise TypeError("pil_img must be a PIL.Image.Image")

    img = pil_img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    w, h = img.size
    svg = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" "
        f"width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n"
        f"  <image x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" "
        f"xlink:href=\"data:image/png;base64,{png_b64}\" />\n"
        "</svg>\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


def _read_tokens_from_file(path: Path) -> list[str]:
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        tokens.append(s)
    return tokens


def _collect_tokens(args: argparse.Namespace) -> list[str]:
    tokens: list[str] = []
    if args.token:
        tokens.extend([t.strip() for t in args.token if t.strip()])
    if args.tokens:
        # Convenience: allow `--tokens /path/to/tokens.txt`.
        toks = [t.strip() for t in args.tokens if t.strip()]
        if len(toks) == 1:
            maybe_path = Path(toks[0]).expanduser()
            if maybe_path.suffix.lower() in {".txt", ".list"} and maybe_path.is_file():
                tokens.extend(_read_tokens_from_file(maybe_path.resolve()))
            else:
                tokens.extend(toks)
        else:
            tokens.extend(toks)
    if args.token_file:
        tokens.extend(_read_tokens_from_file(Path(args.token_file).expanduser().resolve()))

    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _resolve_token_out_dir(out_root: Path, token: str) -> Path:
    if out_root.exists():
        try:
            for child in sorted(out_root.iterdir()):
                if child.is_dir() and child.name.startswith(token):
                    return child
        except Exception:
            pass
    return out_root / token


def _save_8_cameras(*, frame, token: str, out_dir: Path) -> list[Path]:
    """Save 8 camera images for a single frame and return written paths."""

    from PIL import Image

    cams: list[tuple[object, str]] = [
        (frame.cameras.cam_f0, "camF0"),
        (frame.cameras.cam_l0, "camL0"),
        (frame.cameras.cam_r0, "camR0"),
        (frame.cameras.cam_l1, "camL1"),
        (frame.cameras.cam_r1, "camR1"),
        (frame.cameras.cam_l2, "camL2"),
        (frame.cameras.cam_b0, "camB0"),
        (frame.cameras.cam_r2, "camR2"),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cam, suffix in cams:
        arr = _camera_to_uint8_rgb(cam)
        im = Image.fromarray(arr).convert("RGBA")
        out_path = out_dir / f"{token}_{suffix}.svg"
        _save_pil_as_embedded_svg(im, out_path)
        written.append(out_path)
    return written


def main() -> int:
    args = _parse_args()
    _ensure_local_devkit_on_path()

    tokens = _collect_tokens(args)
    if not tokens:
        print("ERROR: no tokens provided. Use --token/--tokens/--token-file", file=sys.stderr)
        return 2

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

    metric_cache_path_str = args.metric_cache_path or _default_metric_cache_path()
    metric_cache_path = None
    if metric_cache_path_str:
        metric_cache_path = Path(metric_cache_path_str).expanduser().resolve()
        if not metric_cache_path.exists():
            metric_cache_path = None

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
        out_root = (Path(navsim_exp_root) / "token_cam8").resolve()

    # Load scene with only 1 frame (history=1, future=0) and all 8 cameras.
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig

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

    n_ok = 0
    for token in tokens:
        log_name = None
        split_for_token = split
        navsim_log_path_for_token = navsim_log_path
        sensor_blobs_path_for_token = sensor_blobs_path

        if metric_cache_path is not None:
            try:
                mc_file = _get_metric_cache_file(metric_cache_path, token)
                log_name = _infer_log_name_from_metric_cache_file(mc_file)
                openscene_root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
                if log_name and openscene_root:
                    detected = _detect_split_for_log(
                        openscene_root=Path(openscene_root),
                        preferred_split=split_for_token,
                        log_name=log_name,
                    )
                    if detected and detected != split_for_token:
                        print(
                            f"[WARN] token {token} appears to belong to split '{detected}' (log={log_name}); overriding split '{split_for_token}' -> '{detected}'.",
                            file=sys.stderr,
                        )
                        split_for_token = detected
                        if args.navsim_log_path is None:
                            navsim_log_path_for_token = Path(_default_navsim_log_path(split_for_token)).expanduser().resolve()
                        if args.sensor_blobs_path is None:
                            sensor_blobs_path_for_token = Path(_default_sensor_blobs_path(split_for_token)).expanduser().resolve()
            except Exception as e:
                print(f"[WARN] token {token} split/log inference skipped: {e}", file=sys.stderr)

        if not navsim_log_path_for_token.exists():
            print(f"[ERR] navsim_log_path not found for token {token}: {navsim_log_path_for_token}", file=sys.stderr)
            continue
        if not sensor_blobs_path_for_token.exists():
            print(f"[ERR] sensor_blobs_path not found for token {token}: {sensor_blobs_path_for_token}", file=sys.stderr)
            continue

        scene_filter = SceneFilter(
            num_history_frames=1,
            num_future_frames=0,
            frame_interval=1,
            has_route=False,
            max_scenes=1,
            log_names=[log_name] if log_name else None,
            tokens=[token],
        )

        scene_loader = SceneLoader(
            data_path=navsim_log_path_for_token,
            sensor_blobs_path=sensor_blobs_path_for_token,
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )

        if token not in scene_loader.tokens:
            print(f"[ERR] token not found after filtering: {token}", file=sys.stderr)
            continue

        scene = scene_loader.get_scene_from_token(token)
        frame = scene.frames[scene.scene_metadata.num_history_frames - 1]

        token_out_dir = _resolve_token_out_dir(out_root, token).resolve()
        written = _save_8_cameras(frame=frame, token=token, out_dir=token_out_dir)

        print("=== token_cam8 done ===")
        print(f"token: {token}")
        print(f"split: {split_for_token}")
        print(f"out:   {token_out_dir}")
        if written:
            print(f"files: {len(written)}")
        n_ok += 1

    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
