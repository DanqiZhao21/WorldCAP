#!/usr/bin/env python3
"""Visualize GT + WoTE + WorldCAP trajectories on the FRONT camera (cam_f0).

This script:
1) Loads scene for a given token
2) Runs two checkpoints (baseline vs worldcap) to get the model-selected trajectory
3) Projects GT / baseline / worldcap trajectories into cam_f0 image
4) Saves a single SVG (PNG embedded as base64 inside an SVG container)

Output layout:
    <out-dir>/<token*/>/<token>_frontcam.svg

2026-03 update:
- Support batch tokens via --token/--tokens/--token-file
- Output SVG only
- Output directory prefers an existing subdir whose name starts with the token
        (e.g. token--good); otherwise creates <out_root>/<token>/.
- Output filename is deterministic (<token>_frontcam.svg) and will overwrite if exists.

Typical usage:
  python tool/smalltool/visualize_token_frontcam_traj_compare.py \
    --token 12f5053055935463 \
    --split val \
    --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
    --worldcap-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt

Defaults:
  out-dir = /home/zhaodanqi/clone/WoTE/WorldCAP_pic/frontCAM

Notes:
- This follows the dataset/path handling patterns used in the other token tools.
- It intentionally keeps the visualization minimal (just trajectories on the front cam).


#FIXME: COLLISION
  python /home/zhaodanqi/clone/WoTE/tool/smalltool/visualize_token_frontcam_traj_compare.py \
        --tokens 25226305bdff5efd 7b8d8487706f501e bc91cb648d525c6c 055c41d3c8e75bdc 9b16e4fea2c25446 ae06592110305073 98c7a48dd75052ab 4c22cdcc527e5a36 432598c0bda65445 198bc5f3280e52cd 3d4616d64a4c5f53 431869f33ace51a0 5a80299213875068 7ae00644dbef537f 885f450f0b875861 8fb11d5808355072 b77b4b6eb149553f b87ed2985e545397 c5ff90667143574a f1dabe118a6955d6 b66db24cee7957a2 9164913b818a58b1 667afe1f010351c5 79a151c333745253 d1bae9e7d9785598 \
        --split val \
        --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
        --worldcap-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
        --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault
        
#FIXME: drivable
  python /home/zhaodanqi/clone/WoTE/tool/smalltool/visualize_token_frontcam_traj_compare.py \
    --tokens 12f5053055935463 2be2e48e80985bee 75f5cc1f425c501d 8164612a623156ae 891fe2fc30c95109 9c9b62965da75764 a9c368a110585b1a b50b8f11d75a5cb0 f4814e7eb01252b6 fb42b4cbf6b95b6c 32a19a2ab50f563d 0defa8939c9851ea ea21c4b17b865a4d 2751f9eb641455c9 0558c7a64ef157b1 4909b88b347c5764 13fa82d6564e5bac 6950076b024c51db aaa11cdbc8d35178 bb9c441a4c2b5791 1e9a42ef8f4057a6 d37e71f395695c6f 778219f3cac65d35 f99992756cbb5adb 8364af67153a5193 62462203db6b5ba5 1039e136e6605cfb e03da8beb33a5e06 b113e988ede45a4f 87b32a1aeeb85613 87d3c1135ac85583 c87fe1d7a3bd57cb 5c5d006eb7b854c3 bc2314763cfc545d ea34282dc63d5a9a cb5022a3bef557e3 6c28c001109f5718 91fc00df56ae5aca c2ed826b31065c66 602e1bc4f8575d4e ce3d0bc0b2d55876 1da6196444e35b0f fc9b5914f47e58fe c93a302d2fb2508f 406f8b299de35ce2 6d0a7f0bb4e7584b 0c91824ce1e65b6e be85b447a33b59f8 d5fc95fa66025d7a 012432bd62b85f80 e13b89a8813159df 8183cdc6ff5a5726 f9c8ec6aefc05be4 14b6e7ce317d531f 759af2e479de5bbb \
    --split val \
    --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
    --worldcap-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
    --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_drivable

c3ace87d2f985eaa

  python /home/zhaodanqi/clone/WoTE/tool/smalltool/visualize_token_frontcam_traj_compare.py \
        --token c3ace87d2f985eaa \
        --split val \
        --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
        --worldcap-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
        --out-dir /home/zhaodanqi/clone/WoTE/WorldCAP_pic/scenario/scenarios_collisonatfault
        
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import base64
import io


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

    p.add_argument("--baseline-ckpt", required=True, help="path to baseline checkpoint")
    p.add_argument("--worldcap-ckpt", required=True, help="path to worldcap checkpoint")

    p.add_argument(
        "--out-dir",
        default="/home/zhaodanqi/clone/WoTE/WorldCAP_pic/frontCAM",
        help="output root dir (default: /home/zhaodanqi/clone/WoTE/WorldCAP_pic/frontCAM)",
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

    # Keep defaults aligned with batch eval conventions
    p.add_argument(
        "--baseline-ctrl-wm",
        type=int,
        default=0,
        choices=[0, 1],
        help="Override controller_condition_on_world_model for baseline run.",
    )
    p.add_argument(
        "--worldcap-ctrl-wm",
        type=int,
        default=1,
        choices=[0, 1],
        help="Override controller_condition_on_world_model for worldcap run.",
    )
    p.add_argument(
        "--worldcap-wm-fusion",
        default="film03",
        help="Override controller_world_model_fusion for worldcap run (default: film03).",
    )
    p.add_argument(
        "--worldcap-wm-strength",
        type=float,
        default=0.3,
        help="Override controller_world_model_strength for worldcap run (default: 0.3).",
    )

    return p.parse_args()


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


def _save_pil_as_embedded_svg(*, pil_img: Any, out_path: Path) -> None:
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


def _camera_to_uint8_rgb(cam) -> "object":
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

    if np.issubdtype(arr.dtype, np.floating):
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if amax <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    return arr


def _hex_to_rgba(hex_color: str, alpha: float) -> Tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    a = int(max(0.0, min(1.0, alpha)) * 255)
    return (r, g, b, a)


def _project_lidar_points_to_pixels(
    *,
    points_lidar_xyz: "object",
    cam: Any,
    image_shape: Tuple[int, int],
    eps: float = 1e-3,
) -> Tuple["object", "object"]:
    """Project Nx3 lidar-frame points to image pixels using cam extrinsics + intrinsics."""

    import numpy as np

    # IMPORTANT:
    # Match navsim.visualization.camera._transform_pcs_to_images() exactly.
    # That code uses a row-vector convention and stores translation in the last ROW,
    # then applies transform via transpose. Re-using the same math avoids systematic
    # pixel shifts/rotations.

    sensor2lidar_rotation = cam.sensor2lidar_rotation
    sensor2lidar_translation = cam.sensor2lidar_translation
    intrinsic = cam.intrinsics

    pts = np.asarray(points_lidar_xyz, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(0, 2), np.zeros((0,), dtype=bool)

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    proj = (lidar2img_rt @ pts_h.T).T

    valid = proj[:, 2] > eps
    pix = proj[:, 0:2] / np.maximum(proj[:, 2:3], np.ones_like(proj[:, 2:3]) * eps)

    img_h, img_w = image_shape
    valid = (
        valid
        & (pix[:, 0] < (img_w - 1))
        & (pix[:, 0] > 0)
        & (pix[:, 1] < (img_h - 1))
        & (pix[:, 1] > 0)
    )

    return pix, valid


def _draw_polyline_segments_rgba(
    *,
    overlay,
    pixels_xy: "object",
    valid_mask: "object",
    color_rgba: Tuple[int, int, int, int],
    width: int,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(overlay)
    pts = pixels_xy
    m = valid_mask
    n = len(pts)
    for i in range(n - 1):
        if bool(m[i]) and bool(m[i + 1]):
            x0, y0 = float(pts[i][0]), float(pts[i][1])
            x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
            draw.line([(x0, y0), (x1, y1)], fill=color_rgba, width=width)


def _trajectory_to_lidar_points(traj: Any) -> "object":
    """Convert a navsim Trajectory into Nx3 points in (approx) lidar frame on ground plane."""

    import numpy as np

    poses = np.asarray(traj.poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 2:
        return np.zeros((0, 3), dtype=np.float32)

    # Include t0 at origin (0,0)
    xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), poses[:, 0:2]], axis=0)
    z = np.zeros((xy.shape[0], 1), dtype=np.float32)
    return np.concatenate([xy, z], axis=1)


def _compose_cfg(
    *,
    config_dir: Path,
    split: str,
    output_dir: Path,
    navsim_log_path: Path,
    sensor_blobs_path: Path,
    metric_cache_path: Optional[Path],
    ckpt_path: Path,
    ctrl_enable_world_model: Optional[bool],
    ctrl_world_model_fusion: Optional[str],
    ctrl_world_model_strength: Optional[float],
) -> Any:
    from hydra import compose

    def _q(v: str) -> str:
        # Quote values so Hydra override parser doesn't choke on special chars like '=' in paths.
        s = str(v)
        s = s.replace("\\", "\\\\")
        s = s.replace('"', "\\\"")
        return f'"{s}"'

    overrides = [
        "agent=WoTE_agent",
        f"split={_q(split)}",
        "experiment_name=token_frontcam",
        f"output_dir={_q(str(output_dir))}",
        f"navsim_log_path={_q(str(navsim_log_path))}",
        f"sensor_blobs_path={_q(str(sensor_blobs_path))}",
        "worker=sequential",
        f"agent.checkpoint_path={_q(str(ckpt_path))}",
    ]
    if metric_cache_path is not None:
        overrides.append(f"metric_cache_path={_q(str(metric_cache_path))}")

    if ctrl_enable_world_model is not None:
        overrides.append(
            f"++agent.config.controller_condition_on_world_model={int(bool(ctrl_enable_world_model))}"
        )
    if ctrl_world_model_fusion is not None:
        overrides.append(
            f"++agent.config.controller_world_model_fusion={_q(ctrl_world_model_fusion)}"
        )
    if ctrl_world_model_strength is not None:
        overrides.append(
            f"++agent.config.controller_world_model_strength={float(ctrl_world_model_strength)}"
        )

    cfg = compose(config_name="default_run_pdm_score", overrides=overrides)

    # Ensure single token lookup
    try:
        cfg.scene_filter.frame_interval = 1
        cfg.scene_filter.tokens = []
        cfg.scene_filter.log_names = None
    except Exception:
        pass

    return cfg


def _run_agent_one(
    *,
    cfg: Any,
    token: str,
    log_name: Optional[str],
    navsim_log_path: Path,
    sensor_blobs_path: Path,
) -> Tuple[Any, Any, Any]:
    """Return (scene, agentout, chosen_traj)."""

    from hydra.utils import instantiate
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.is_eval = True

    # Build a SceneLoader aligned with agent sensors
    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    scene_filter.frame_interval = 1
    scene_filter.tokens = [token]
    if log_name:
        scene_filter.log_names = [log_name]

    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    if token not in scene_loader.tokens:
        raise KeyError(f"token not found after filtering: {token}")

    scene = scene_loader.get_scene_from_token(token)

    use_future_frames = bool(getattr(getattr(agent, "config", None), "use_fut_frames", False))
    agent_input = scene_loader.get_agent_input_from_token(token, use_fut_frames=use_future_frames)

    if agent.requires_scene:
        agentout = agent.compute_trajectory(agent_input, scene)
    else:
        agentout = agent.compute_trajectory(agent_input)

    import numpy as np

    chosen_idx = int(np.argmax(agentout["trajectory_scores"]))
    chosen_traj = agentout["trajectories"][chosen_idx]

    return scene, agentout, chosen_traj


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

    baseline_ckpt = Path(args.baseline_ckpt).expanduser().resolve()
    worldcap_ckpt = Path(args.worldcap_ckpt).expanduser().resolve()
    if not baseline_ckpt.exists():
        print(f"ERROR: baseline ckpt not found: {baseline_ckpt}", file=sys.stderr)
        return 2
    if not worldcap_ckpt.exists():
        print(f"ERROR: worldcap ckpt not found: {worldcap_ckpt}", file=sys.stderr)
        return 2

    # Hydra compose
    from hydra import initialize_config_dir

    repo_root = Path(__file__).resolve().parents[2]
    config_dir = (repo_root / "navsim" / "planning" / "script" / "config" / "pdm_scoring").resolve()
    if not config_dir.exists():
        print(f"ERROR: hydra config dir not found: {config_dir}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir).expanduser().resolve()

    # Compose + run agents (Hydra API differs across versions)
    try:
        ctx = initialize_config_dir(version_base=None, config_dir=str(config_dir))
    except TypeError:
        ctx = initialize_config_dir(config_dir=str(config_dir))

    failures = 0
    split_base = (args.split or "").strip()
    if split_base.lower() == "text":
        print("[WARN] split='text' looks like a typo; using 'test'.", file=sys.stderr)
        split_base = "test"

    with ctx:
        for token in tokens:
            split = split_base

            navsim_log_path_str = args.navsim_log_path or _default_navsim_log_path(split)
            sensor_blobs_path_str = args.sensor_blobs_path or _default_sensor_blobs_path(split)
            if not navsim_log_path_str:
                print("ERROR: --navsim-log-path is required if OPENSCENE_DATA_ROOT is not set", file=sys.stderr)
                return 2
            if not sensor_blobs_path_str:
                print("ERROR: --sensor-blobs-path is required if OPENSCENE_DATA_ROOT is not set", file=sys.stderr)
                return 2

            navsim_log_path = Path(navsim_log_path_str).expanduser().resolve()
            sensor_blobs_path = Path(sensor_blobs_path_str).expanduser().resolve()

            # Optional: infer log_name and adjust split.
            log_name = None
            metric_cache_path = None
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
            out_path = out_dir / f"{token}_frontcam.svg"

            try:
                cfg_base = _compose_cfg(
                    config_dir=config_dir,
                    split=split,
                    output_dir=out_dir,
                    navsim_log_path=navsim_log_path,
                    sensor_blobs_path=sensor_blobs_path,
                    metric_cache_path=metric_cache_path,
                    ckpt_path=baseline_ckpt,
                    ctrl_enable_world_model=bool(args.baseline_ctrl_wm),
                    ctrl_world_model_fusion=None,
                    ctrl_world_model_strength=None,
                )
                cfg_wc = _compose_cfg(
                    config_dir=config_dir,
                    split=split,
                    output_dir=out_dir,
                    navsim_log_path=navsim_log_path,
                    sensor_blobs_path=sensor_blobs_path,
                    metric_cache_path=metric_cache_path,
                    ckpt_path=worldcap_ckpt,
                    ctrl_enable_world_model=bool(args.worldcap_ctrl_wm),
                    ctrl_world_model_fusion=str(args.worldcap_wm_fusion),
                    ctrl_world_model_strength=float(args.worldcap_wm_strength),
                )

                scene, _agentout_b, traj_b = _run_agent_one(
                    cfg=cfg_base,
                    token=token,
                    log_name=log_name,
                    navsim_log_path=navsim_log_path,
                    sensor_blobs_path=sensor_blobs_path,
                )
                _scene2, _agentout_wc, traj_wc = _run_agent_one(
                    cfg=cfg_wc,
                    token=token,
                    log_name=log_name,
                    navsim_log_path=navsim_log_path,
                    sensor_blobs_path=sensor_blobs_path,
                )

                # Build GT trajectory
                human_traj = scene.get_future_trajectory()

                frame_idx = scene.scene_metadata.num_history_frames - 1
                frame = scene.frames[frame_idx]
                cam = frame.cameras.cam_f0

                from PIL import Image

                base_rgb = _camera_to_uint8_rgb(cam)
                base = Image.fromarray(base_rgb).convert("RGBA")
                overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

                # Colors: keep consistent with BEV focused plot script.
                COLOR_GT = "#80c5f0"
                COLOR_WOTE = "#27cfbe"
                COLOR_WORLDCAP = "#fa7f6f"

                # Alpha + width
                alpha_gt = 0.85
                alpha_wote = 0.95
                alpha_wc = 0.98
                width_gt = 6
                width_pred = 6

                img_h, img_w = base_rgb.shape[0], base_rgb.shape[1]

                def _draw(traj: Any, color_hex: str, alpha: float, width: int) -> None:
                    pts_lidar = _trajectory_to_lidar_points(traj)
                    if len(pts_lidar) < 2:
                        return
                    pix, valid = _project_lidar_points_to_pixels(
                        points_lidar_xyz=pts_lidar,
                        cam=cam,
                        image_shape=(img_h, img_w),
                    )
                    _draw_polyline_segments_rgba(
                        overlay=overlay,
                        pixels_xy=pix,
                        valid_mask=valid,
                        color_rgba=_hex_to_rgba(color_hex, alpha),
                        width=width,
                    )

                _draw(human_traj, COLOR_GT, alpha_gt, width_gt)
                _draw(traj_b, COLOR_WOTE, alpha_wote, width_pred)
                _draw(traj_wc, COLOR_WORLDCAP, alpha_wc, width_pred)

                out_img = Image.alpha_composite(base, overlay)
                _save_pil_as_embedded_svg(pil_img=out_img, out_path=out_path)

                print("=== frontcam trajectory compare done ===")
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
