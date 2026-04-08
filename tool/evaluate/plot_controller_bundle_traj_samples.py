#!/usr/bin/env python3
"""Sample controller styles from one or more controller bundle .npz and plot traj overlays.

Each bundle (.npz) is expected to contain:
- ref_traj:   (N_ref, T, 3) float32
- exec_trajs: (N_style, N_ref, T, 3) float32
- style_kind: (N_style,) object/str, typically {'trk','pd'}
- style_names/style_group_big/style_group_sub: metadata

For each bundle, this script randomly samples:
- 5 tracker-level styles  (style_kind == 'trk')
- 5 post-dynamics styles  (style_kind == 'pd')

And writes one SVG per sampled style:
- red: reference trajectories
- blue: executed trajectories for that style

Note: Large bundles (e.g. N_ref=1024) are downsampled for plotting via --max-ref.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sanitize_filename(s: str, *, max_len: int = 120) -> str:
    s = (s or "").strip()
    if not s:
        return "unnamed"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    if not s:
        s = "unnamed"
    return s[:max_len]


def _maybe_fix_path(p: Path) -> Path:
    if p.exists():
        return p
    # Common typo seen in the conversation: .../generated/128/controller_styles_64.npz
    ps = str(p)
    if "/generated/128/" in ps and ps.endswith("controller_styles_64.npz"):
        alt = Path(ps.replace("/generated/128/", "/generated/64/"))
        if alt.exists():
            print(f"[WARN] bundle not found: {p} -> using {alt}", file=sys.stderr)
            return alt
    return p


def _iter_style_indices(style_kind: np.ndarray, kind: str) -> np.ndarray:
    # style_kind comes from npz as object array of strings.
    kinds = np.asarray(style_kind).astype(str)
    return np.flatnonzero(kinds == kind)


def _plot_overlay(
    ref_xy: np.ndarray,
    exec_xy: np.ndarray,
    title: str,
    out_svg: Path,
    *,
    alpha_ref: float,
    alpha_exec: float,
    lw_ref: float,
    lw_exec: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=160)

    for i in range(ref_xy.shape[0]):
        ax.plot(ref_xy[i, :, 0], ref_xy[i, :, 1], color=(1.0, 0.2, 0.2), alpha=alpha_ref, linewidth=lw_ref)
        ax.plot(exec_xy[i, :, 0], exec_xy[i, :, 1], color=(0.15, 0.35, 1.0), alpha=alpha_exec, linewidth=lw_exec)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def _sample_without_replacement(rng: np.random.Generator, indices: np.ndarray, k: int) -> np.ndarray:
    if indices.size == 0:
        return indices
    k = min(int(k), int(indices.size))
    return rng.choice(indices, size=k, replace=False)


def _bundle_tag(bundle_path: Path) -> str:
    # Prefer the numeric folder name (64/128/1024) when present.
    parts = list(bundle_path.parts)
    for seg in reversed(parts):
        if seg.isdigit():
            return seg
    return bundle_path.stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bundles",
        nargs="+",
        required=True,
        help="One or more controller bundle .npz paths (e.g. ControllerExp/generated/64/controller_styles_64.npz)",
    )
    p.add_argument("--out-dir", default="ControllerExp/generated/plots_bundle_traj_samples", help="Output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=5, help="How many styles to sample per kind (trk/pd)")
    p.add_argument(
        "--style-idx-trk",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit tracker style indices to plot (space-separated). If set, overrides random sampling.",
    )
    p.add_argument(
        "--style-idx-pd",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit post-dynamics style indices to plot (space-separated). If set, overrides random sampling.",
    )
    p.add_argument(
        "--max-ref",
        type=int,
        default=128,
        help="Max number of reference trajectories to plot per figure (downsample for large bundles)",
    )
    p.add_argument("--alpha-ref", type=float, default=0.25)
    p.add_argument("--alpha-exec", type=float, default=0.35)
    p.add_argument("--lw-ref", type=float, default=0.9)
    p.add_argument("--lw-exec", type=float, default=1.0)
    return p.parse_args()


def _name_to_index_map(style_names: Optional[np.ndarray]) -> dict[str, int]:
    if style_names is None:
        return {}
    out: dict[str, int] = {}
    for i, n in enumerate(style_names.tolist()):
        key = str(n)
        if key not in out:
            out[key] = int(i)
    return out


def _resolve_style_indices_for_bundle(
    *,
    target_style_kind: np.ndarray,
    target_style_names: Optional[np.ndarray],
    ref_selected_indices: np.ndarray,
    ref_selected_names: Optional[list[str]],
    expected_kind: str,
) -> np.ndarray:
    """Map reference-selected styles to indices in a target bundle.

    Primary behavior: use the same integer indices.
    Fallback: if style_name exists in target bundle, use name->index mapping.
    """

    target_kinds = np.asarray(target_style_kind).astype(str)
    n_style = int(target_kinds.shape[0])

    name_map = _name_to_index_map(target_style_names)

    resolved: list[int] = []
    for j, ref_idx in enumerate(ref_selected_indices.tolist()):
        idx = int(ref_idx)

        # Name-based mapping (more robust if ordering ever changes)
        if ref_selected_names is not None and j < len(ref_selected_names):
            nm = ref_selected_names[j]
            if nm in name_map:
                idx = int(name_map[nm])

        if idx < 0 or idx >= n_style:
            continue
        if target_kinds[idx] != expected_kind:
            continue
        resolved.append(idx)

    return np.asarray(resolved, dtype=int)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    out_root = Path(args.out_dir).expanduser().resolve()

    bundles = [Path(b).expanduser().resolve() for b in args.bundles]
    bundles = [_maybe_fix_path(b) for b in bundles]

    # Load all bundles first so we can pick a consistent set of style indices.
    loaded = []
    for bundle_path in bundles:
        if not bundle_path.exists():
            print(f"[ERR] missing bundle: {bundle_path}", file=sys.stderr)
            continue
        d = np.load(bundle_path, allow_pickle=True)
        needed = ["ref_traj", "exec_trajs", "style_kind"]
        if any(k not in d for k in needed):
            missing = [k for k in needed if k not in d]
            print(f"[ERR] bundle {bundle_path} missing keys: {missing}", file=sys.stderr)
            continue

        ref_traj = np.asarray(d["ref_traj"], dtype=np.float32)  # (N_ref,T,3)
        exec_trajs = np.asarray(d["exec_trajs"], dtype=np.float32)  # (N_style,N_ref,T,3)
        if exec_trajs.shape[1] != ref_traj.shape[0]:
            print(
                f"[ERR] shape mismatch in {bundle_path}: exec_trajs.shape={exec_trajs.shape} ref_traj.shape={ref_traj.shape}",
                file=sys.stderr,
            )
            continue

        loaded.append(
            {
                "path": bundle_path,
                "ref_traj": ref_traj,
                "exec_trajs": exec_trajs,
                "style_kind": d["style_kind"],
                "style_names": d["style_names"] if "style_names" in d else None,
                "group_big": d["style_group_big"] if "style_group_big" in d else None,
                "group_sub": d["style_group_sub"] if "style_group_sub" in d else None,
            }
        )

    if not loaded:
        return 2

    ref_bundle = loaded[0]
    ref_kind = ref_bundle["style_kind"]
    ref_names_arr = ref_bundle["style_names"]

    if args.style_idx_trk is not None:
        picked_trk_ref = np.asarray(args.style_idx_trk, dtype=int)
    else:
        idxs = _iter_style_indices(ref_kind, "trk")
        picked_trk_ref = _sample_without_replacement(rng, idxs, int(args.n_samples))

    if args.style_idx_pd is not None:
        picked_pd_ref = np.asarray(args.style_idx_pd, dtype=int)
    else:
        idxs = _iter_style_indices(ref_kind, "pd")
        picked_pd_ref = _sample_without_replacement(rng, idxs, int(args.n_samples))

    picked_trk_names = None
    picked_pd_names = None
    if ref_names_arr is not None:
        ref_names_list = [str(x) for x in ref_names_arr.tolist()]
        picked_trk_names = [ref_names_list[i] for i in picked_trk_ref.tolist() if 0 <= int(i) < len(ref_names_list)]
        picked_pd_names = [ref_names_list[i] for i in picked_pd_ref.tolist() if 0 <= int(i) < len(ref_names_list)]

    print(f"[INFO] picked trk style_idx (reference): {picked_trk_ref.tolist()}")
    print(f"[INFO] picked pd  style_idx (reference): {picked_pd_ref.tolist()}")

    n_ok = 0
    for bundle in loaded:
        bundle_path: Path = bundle["path"]
        ref_traj: np.ndarray = bundle["ref_traj"]
        exec_trajs: np.ndarray = bundle["exec_trajs"]
        style_kind = bundle["style_kind"]
        style_names = bundle["style_names"]
        group_big = bundle["group_big"]
        group_sub = bundle["group_sub"]

        # Choose reference traj subset for plotting (downsample large bundles).
        n_ref = int(ref_traj.shape[0])
        if args.max_ref > 0 and n_ref > int(args.max_ref):
            ref_idx = rng.choice(n_ref, size=int(args.max_ref), replace=False)
            ref_idx = np.sort(ref_idx)
        else:
            ref_idx = np.arange(n_ref)

        tag = _bundle_tag(bundle_path)
        bundle_out = out_root / f"bundle_{tag}"

        # Resolve reference-picked styles to this bundle.
        picked_trk = _resolve_style_indices_for_bundle(
            target_style_kind=style_kind,
            target_style_names=style_names,
            ref_selected_indices=picked_trk_ref,
            ref_selected_names=picked_trk_names,
            expected_kind="trk",
        )
        picked_pd = _resolve_style_indices_for_bundle(
            target_style_kind=style_kind,
            target_style_names=style_names,
            ref_selected_indices=picked_pd_ref,
            ref_selected_names=picked_pd_names,
            expected_kind="pd",
        )

        for kind, pick in (("trk", picked_trk), ("pd", picked_pd)):
            if pick.size == 0:
                print(f"[WARN] bundle {bundle_path} resolved no styles for kind='{kind}'", file=sys.stderr)
                continue
            for style_idx in pick.tolist():
                name = str(style_names[style_idx]) if style_names is not None else f"style_{style_idx}"
                gb = str(group_big[style_idx]) if group_big is not None else ""
                gs = str(group_sub[style_idx]) if group_sub is not None else ""

                title = f"bundle={tag} kind={kind} idx={style_idx} name={name} {gb}/{gs}".strip()
                ref_xy = ref_traj[ref_idx, :, :2]
                exec_xy = exec_trajs[style_idx, ref_idx, :, :2]

                out_name = _sanitize_filename(f"{style_idx:03d}__{name}__{gb}__{gs}.svg")
                out_svg = bundle_out / kind / out_name

                _plot_overlay(
                    ref_xy,
                    exec_xy,
                    title,
                    out_svg,
                    alpha_ref=float(args.alpha_ref),
                    alpha_exec=float(args.alpha_exec),
                    lw_ref=float(args.lw_ref),
                    lw_exec=float(args.lw_exec),
                )

        print(f"[OK] {bundle_path} -> {bundle_out}")
        n_ok += 1

    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
