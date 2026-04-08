#!/usr/bin/env python3
"""Visualize a single token for two WoTE checkpoints (baseline vs film03).

This script:
1) Loads the scene + metric cache for a given token
2) Runs WoTE to generate multi-modal candidate trajectories
3) Runs PDM scoring in *evaluate_all_trajectories* mode (for visualization)
4) Saves two BEV images (one per checkpoint)

It reuses the repo's existing implementations:
- navsim.evaluate.pdm_score.pdm_score_multiTraj
- navsim.visualization.plots.plot_bev_with_agent_and_simulation

Typical usage:
  python tool/smalltool/visualize_token_compare.py \
    --token <TOKEN> \
    --split val \
    --baseline-ckpt /path/to/baseline.ckpt \
    --film-ckpt /path/to/film03.ckpt \
    --out-dir /path/to/out
    
    
    
python tool/smalltool/visualize_token_compare.py \
    --token 055c41d3c8e75bdc \
    --split val \
    --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
    --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
    --out-dir /home/zhaodanqi/clone/WoTE/pic/scenarios
    

Paths can be provided explicitly, otherwise env vars are used:
- OPENSCENE_DATA_ROOT (for navsim logs + sensor blobs)
- NAVSIM_EXP_ROOT     (for metric cache)
"""

from __future__ import annotations

import argparse
import contextlib
import lzma
import os
import pickle
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _ensure_local_devkit_on_path() -> None:
    """Mimic the repo's eval scripts that export PYTHONPATH for local packages."""

    repo_root = Path(__file__).resolve().parents[2]
    navsim_pkg = repo_root / "navsim"
    nuplan_pkg = repo_root / "nuplan-devkit"

    for p in [repo_root, navsim_pkg, nuplan_pkg]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    # Consistent map version default (same as eval scripts)
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")


# Headless backend for servers/ssh.
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


# Backward compatible alias (older drafts used the plural form).
def _default_navsim_logs_path(split: str) -> Optional[str]:
    return _default_navsim_log_path(split)


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


def _list_subdirs(p: Path) -> str:
    try:
        if not p.exists():
            return "<missing>"
        names = [c.name for c in p.iterdir() if c.is_dir()]
        names = sorted(names)
        return ", ".join(names) if names else "<none>"
    except Exception:
        return "<unavailable>"


def _default_metric_cache_path() -> Optional[str]:
    root = (os.environ.get("NAVSIM_EXP_ROOT") or "").strip()
    if not root:
        return None
    return str(Path(root) / "metric_cache")


def _load_metric_cache(metric_cache_path: Path, token: str) -> MetricCache:
    from navsim.common.dataloader import MetricCacheLoader

    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    if token not in metric_cache_loader.metric_cache_paths:
        raise KeyError(f"token not found in metric cache: {token}")

    metric_cache_file = metric_cache_loader.metric_cache_paths[token]
    with lzma.open(metric_cache_file, "rb") as f:
        metric_cache = pickle.load(f)
    return metric_cache


def _get_metric_cache_file(metric_cache_path: Path, token: str) -> Path:
    from navsim.common.dataloader import MetricCacheLoader

    metric_cache_loader = MetricCacheLoader(metric_cache_path)
    if token not in metric_cache_loader.metric_cache_paths:
        raise KeyError(f"token not found in metric cache: {token}")
    return Path(metric_cache_loader.metric_cache_paths[token])


def _infer_log_name_from_metric_cache_file(metric_cache_file: Path) -> Optional[str]:
    """Infer log_name from metric_cache path.

    Typical layout:
      <metric_cache_root>/<log_name>/unknown/<token>/metric_cache.pkl
    """

    parts = list(metric_cache_file.parts)
    try:
        idx = parts.index("metric_cache")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except Exception:
        pass
    return None


def _detect_split_for_log(
    *,
    openscene_root: Path,
    preferred_split: str,
    log_name: str,
) -> Optional[str]:
    """Pick the dataset split that contains <log_name>.pkl under navsim_logs."""

    logs_root = openscene_root / "navsim_logs"
    if not logs_root.exists():
        return None

    candidates = []
    if preferred_split:
        candidates.append(preferred_split)
    # common splits in this repo
    for s in ["test", "trainval", "mini", "val", "train"]:
        if s not in candidates:
            candidates.append(s)
    # also scan any other dirs
    for d in sorted([p.name for p in logs_root.iterdir() if p.is_dir()]):
        if d not in candidates:
            candidates.append(d)

    for split in candidates:
        p = logs_root / split / f"{log_name}.pkl"
        if p.exists():
            return split
    return None


def _build_scene_loader(
    *,
    cfg: Any,
    agent: AbstractAgent,
    token: str,
    log_name: Optional[str],
    navsim_log_path: Path,
    sensor_blobs_path: Path,
) -> SceneLoader:
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter

    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    # Critical for single-token lookup:
    # Some default scene filters leave frame_interval=None, which becomes num_frames (non-overlapping).
    # That means only every-kth token is kept, and arbitrary tokens may be missing.
    # Eval scripts often use frame_interval=1 (e.g., navtest), so we enforce it here.
    scene_filter.frame_interval = 1
    scene_filter.tokens = [token]
    if log_name:
        scene_filter.log_names = [log_name]

    return SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )


def _instantiate_sim_and_scorer(cfg: Any) -> Tuple[PDMSimulator, PDMScorer]:
    from hydra.utils import instantiate
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    simulator: PDMSimulator = instantiate(cfg.simulator)

    # Sanitize scorer config to avoid unexpected root-level keys being passed in.
    scorer_cfg: Dict[str, Any] = {
        "_target_": cfg.scorer._target_,
        "_convert_": getattr(cfg.scorer, "_convert_", "all"),
        "proposal_sampling": cfg.scorer.proposal_sampling,
        "config": cfg.scorer.config,
    }
    scorer: PDMScorer = instantiate(scorer_cfg)

    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    return simulator, scorer


def _compose_cfg(
    *,
    config_dir: Path,
    split: str,
    token: str,
    output_dir: Path,
    navsim_log_path: Path,
    sensor_blobs_path: Path,
    metric_cache_path: Path,
    ckpt_path: Path,
    cluster_file_path: Optional[Path],
    sim_reward_dict_path: Optional[Path],
    controller_ref_traj_path: Optional[Path],
    controller_exec_traj_path: Optional[Path],
    controller_response_predictor_path: Optional[Path],
    simulator_post_apply_mode: str,
    ctrl_enable_world_model: Optional[bool],
    ctrl_world_model_fusion: Optional[str],
    ctrl_world_model_strength: Optional[float],
) -> Any:
    from hydra import compose

    def _q(v: str) -> str:
        """Quote a string for Hydra override grammar.

        Needed for paths containing special chars like '=' (e.g. epoch=29-step=...).
        """

        v = str(v)
        return "'" + v.replace("'", "\\'") + "'"

    overrides = [
        "agent=WoTE_agent",
        f"split={_q(split)}",
        "experiment_name=token_viz",
        f"output_dir={_q(output_dir)}",
        f"navsim_log_path={_q(navsim_log_path)}",
        f"sensor_blobs_path={_q(sensor_blobs_path)}",
        f"metric_cache_path={_q(metric_cache_path)}",
        "evaluate_all_trajectories=true",
        f"agent.checkpoint_path={_q(ckpt_path)}",
        # Match eval script behavior: allow online post transforms to affect rollout dynamics.
        f"+simulator.post_params.apply_mode={_q(simulator_post_apply_mode)}",
    ]

    # Match eval script controller overrides (only WM toggles here; other controller flags remain as agent defaults).
    if ctrl_enable_world_model is not None:
        overrides.append(f"++agent.config.controller_condition_on_world_model={str(bool(ctrl_enable_world_model)).lower()}")
    if ctrl_world_model_fusion is not None:
        overrides.append(f"++agent.config.controller_world_model_fusion={_q(ctrl_world_model_fusion)}")
    if ctrl_world_model_strength is not None:
        overrides.append(f"++agent.config.controller_world_model_strength={float(ctrl_world_model_strength)}")

    # Optional: force anchors + sim reward dict.
    if cluster_file_path is not None:
        overrides.append(f"++agent.config.cluster_file_path={_q(cluster_file_path)}")
    if sim_reward_dict_path is not None:
        overrides.append(f"++agent.config.sim_reward_dict_path={_q(sim_reward_dict_path)}")
    if controller_ref_traj_path is not None:
        overrides.append(f"++agent.config.controller_ref_traj_path={_q(controller_ref_traj_path)}")
    if controller_exec_traj_path is not None:
        overrides.append(f"++agent.config.controller_exec_traj_path={_q(controller_exec_traj_path)}")
    if controller_response_predictor_path is not None:
        overrides.append(
            f"++agent.config.controller_response_predictor_path={_q(controller_response_predictor_path)}"
        )

    # For a single-token visualization script we don't need Ray workers.
    overrides.append("worker=sequential")

    cfg = compose(config_name="default_run_pdm_score", overrides=overrides)

    # Narrow down to single token as a guardrail (even if scene_filter defaults change).
    try:
        cfg.scene_filter.tokens = [token]
    except Exception:
        pass

    return cfg


def _run_one(
    *,
    cfg: Any,
    token: str,
    out_dir: Path,
    out_name: str,
    log_name: Optional[str],
    compute_isolated_scores: bool = False,
) -> Dict[str, Any]:
    from hydra.utils import instantiate
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.evaluate.pdm_score import pdm_score_multiTraj
    from navsim.visualization.plots import plot_bev_with_agent_and_simulation

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.is_eval = True

    simulator, scorer = _instantiate_sim_and_scorer(cfg)

    metric_cache = _load_metric_cache(Path(cfg.metric_cache_path), token)

    scene_loader = _build_scene_loader(
        cfg=cfg,
        agent=agent,
        token=token,
        log_name=log_name,
        navsim_log_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
    )

    use_future_frames = bool(getattr(getattr(agent, "config", None), "use_fut_frames", False))
    agent_input = scene_loader.get_agent_input_from_token(token, use_fut_frames=use_future_frames)

    if agent.requires_scene:
        scene = scene_loader.get_scene_from_token(token)
        agentout = agent.compute_trajectory(agent_input, scene)
    else:
        scene = scene_loader.get_scene_from_token(token)
        agentout = agent.compute_trajectory(agent_input)

    # NOTE:
    # Batch eval CSVs are produced with evaluate_all_trajectories=false (fast mode).
    # Visualization needs evaluate_all_trajectories=true (full mode) to get all simulated proposals.
    # These can differ in practice, so compute BOTH.
    (
        pdm_pred_fast,
        _pdm_best_fast,
        _sim_fast,
        _pred_fast,
        _best_pdm_idx_fast,
        pred_idx_fast,
    ) = pdm_score_multiTraj(
        metric_cache=metric_cache,
        model_trajectories=agentout["trajectories"],
        trajectory_Anchor=agentout.get("trajectoryAnchor"),
        model_scores=agentout["trajectory_scores"],
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        anchor_save_dir=None,
        anchor_save_name=None,
        anchor_overwrite=False,
        evaluate_all_trajectories=False,
    )

    (
        pdm_pred_full,
        pdm_best_full,
        simulated_states_all_egoframe,
        _pred_states_all_egoframe,
        best_pdm_idx,
        pred_idx_full,
    ) = pdm_score_multiTraj(
        metric_cache=metric_cache,
        model_trajectories=agentout["trajectories"],
        trajectory_Anchor=agentout.get("trajectoryAnchor"),
        model_scores=agentout["trajectory_scores"],
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        anchor_save_dir=None,
        anchor_save_name=None,
        anchor_overwrite=False,
        evaluate_all_trajectories=True,
    )

    isolated_scores_path: Optional[str] = None
    if compute_isolated_scores:
        import numpy as np

        model_trajectories = agentout["trajectories"]
        num_traj = len(model_trajectories)
        isolated_scores = np.full((num_traj,), np.nan, dtype=np.float32)

        # Isolated fast-mode scoring: each candidate is evaluated alone (baseline + that candidate).
        # This matches the batch-eval CSV's scoring path (evaluate_all_trajectories=false), but for all candidates.
        failures = 0
        for i in range(num_traj):
            try:
                pdm_pred_i, *_rest = pdm_score_multiTraj(
                    metric_cache=metric_cache,
                    model_trajectories=[model_trajectories[i]],
                    trajectory_Anchor=agentout.get("trajectoryAnchor"),
                    model_scores=[0.0],
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                    anchor_save_dir=None,
                    anchor_save_name=None,
                    anchor_overwrite=False,
                    evaluate_all_trajectories=False,
                )
                isolated_scores[i] = float(getattr(pdm_pred_i, "score"))
            except Exception:
                failures += 1
                isolated_scores[i] = np.nan

        out_dir.mkdir(parents=True, exist_ok=True)
        isolated_name = f"{Path(out_name).stem}_isolated_scores.npy"
        isolated_path = out_dir / isolated_name
        np.save(isolated_path, isolated_scores)
        isolated_scores_path = str(isolated_path)
        if failures:
            print(
                f"[WARN] isolated scoring had {failures}/{num_traj} failures (saved as NaN).",
                file=sys.stderr,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    human_traj = scene.get_future_trajectory()
    plot_bev_with_agent_and_simulation(
        scene,
        agent,
        best_pdm_idx=best_pdm_idx,
        best_pred_idx=pred_idx_full,
        human_trajectory=human_traj,
        agent_trajectory=agentout,
        simulation_state=simulated_states_all_egoframe,
        pdm_result_pred=pdm_pred_fast,
        pdm_result_best=pdm_best_full,
        save_path=str(out_dir),
        file_name=out_name,
    )

    return {
        "token": token,
        "out": str(out_dir / out_name),
        # CSV-aligned (fast mode)
        "pdm_pred": asdict(pdm_pred_fast),
        "pred_idx": int(pred_idx_fast),
        # Debug (full mode)
        "pdm_pred_full": asdict(pdm_pred_full),
        "pdm_best": asdict(pdm_best_full),
        "pred_idx_full": int(pred_idx_full),
        "best_pdm_idx": int(best_pdm_idx),
        "isolated_scores_path": isolated_scores_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--token", required=True, help="scenario token to visualize")
    parser.add_argument(
        "--split",
        default="test",
        help="dataset split name (common: test/trainval/mini). If you pass val/train and only trainval exists, the script auto-falls-back. Also tolerates typo: text->test.",
    )

    parser.add_argument("--baseline-ckpt", required=True, help="path to baseline checkpoint")
    parser.add_argument("--film-ckpt", required=True, help="path to film03 checkpoint")

    # Controller bundle alignment (matches tool/evaluate/eval_ckpts_20260225_valstyles.sh)
    parser.add_argument(
        "--controller-ref-traj-path",
        default=None,
        help="(optional) override agent.config.controller_ref_traj_path",
    )
    parser.add_argument(
        "--controller-exec-traj-path",
        default=None,
        help="(optional) override agent.config.controller_exec_traj_path",
    )
    parser.add_argument(
        "--controller-response-predictor-path",
        default=None,
        help="(optional) override agent.config.controller_response_predictor_path",
    )

    # --- Config alignment knobs (to match eval scripts) ---
    parser.add_argument(
        "--sim-post-apply-mode",
        default="online",
        choices=["auto", "online", "offline"],
        help="PDMSimulator post apply mode. Eval scripts often use 'online'.",
    )
    parser.add_argument(
        "--baseline-ctrl-wm",
        type=int,
        default=0,
        choices=[0, 1],
        help="Override agent.config.controller_condition_on_world_model for baseline run (default: 0 like eval script).",
    )
    parser.add_argument(
        "--film-ctrl-wm",
        type=int,
        default=1,
        choices=[0, 1],
        help="Override agent.config.controller_condition_on_world_model for film run (default: 1 like eval script).",
    )
    parser.add_argument(
        "--film-wm-fusion",
        default="film03",
        help="Override agent.config.controller_world_model_fusion for film run (default: film03).",
    )
    parser.add_argument(
        "--film-wm-strength",
        type=float,
        default=0.3,
        help="Override agent.config.controller_world_model_strength for film run (default: 0.3).",
    )
    parser.add_argument(
        "--style-idx",
        type=int,
        default=None,
        help="If set, fixes controller+sim style idx (sets WOTE_CTRL_STYLE_IDX and PDM_SIM_STYLE_IDX).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for python/random/numpy/torch (best-effort).",
    )

    parser.add_argument(
        "--compute-isolated-scores",
        action="store_true",
        help=(
            "If set, computes isolated fast-mode PDM score for every candidate trajectory "
            "(evaluate_all_trajectories=false; baseline + that candidate) and saves a .npy next to the PNG."
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory to save images (default: $NAVSIM_EXP_ROOT/token_viz)",
    )

    parser.add_argument(
        "--navsim-log-path",
        default=None,
        help="path to navsim logs (default: $OPENSCENE_DATA_ROOT/navsim_logs/<split>)",
    )
    parser.add_argument(
        "--sensor-blobs-path",
        default=None,
        help="path to sensor blobs (default: $OPENSCENE_DATA_ROOT/sensor_blobs/<split>)",
    )
    parser.add_argument(
        "--metric-cache-path",
        default=None,
        help="path to metric_cache dir (default: $NAVSIM_EXP_ROOT/metric_cache)",
    )

    parser.add_argument(
        "--cluster-file-path",
        default=None,
        help="(optional) override agent.config.cluster_file_path",
    )
    parser.add_argument(
        "--sim-reward-dict-path",
        default=None,
        help="(optional) override agent.config.sim_reward_dict_path",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Must happen before importing navsim/nuplan modules.
    _ensure_local_devkit_on_path()

    token = args.token.strip()
    split = (args.split or "").strip()
    if split.lower() == "text":
        print("[WARN] split='text' looks like a typo; using 'test'.", file=sys.stderr)
        split = "test"

    baseline_ckpt = Path(args.baseline_ckpt).expanduser().resolve()
    film_ckpt = Path(args.film_ckpt).expanduser().resolve()

    if not baseline_ckpt.exists():
        print(f"ERROR: baseline ckpt not found: {baseline_ckpt}", file=sys.stderr)
        return 2
    if not film_ckpt.exists():
        print(f"ERROR: film ckpt not found: {film_ckpt}", file=sys.stderr)
        return 2

    navsim_log_path_str = args.navsim_log_path or _default_navsim_log_path(split)
    sensor_blobs_path_str = args.sensor_blobs_path or _default_sensor_blobs_path(split)
    metric_cache_path_str = args.metric_cache_path or _default_metric_cache_path()

    if not navsim_log_path_str:
        print(
            "ERROR: --navsim-log-path is required if OPENSCENE_DATA_ROOT is not set",
            file=sys.stderr,
        )
        return 2
    if not sensor_blobs_path_str:
        print(
            "ERROR: --sensor-blobs-path is required if OPENSCENE_DATA_ROOT is not set",
            file=sys.stderr,
        )
        return 2
    if not metric_cache_path_str:
        print(
            "ERROR: --metric-cache-path is required if NAVSIM_EXP_ROOT is not set",
            file=sys.stderr,
        )
        return 2

    navsim_log_path = Path(navsim_log_path_str).expanduser().resolve()
    sensor_blobs_path = Path(sensor_blobs_path_str).expanduser().resolve()
    metric_cache_path = Path(metric_cache_path_str).expanduser().resolve()

    if not navsim_log_path.exists():
        root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
        hint = ""
        if root:
            hint = f" (available splits under navsim_logs: {_list_subdirs(Path(root) / 'navsim_logs')})"
        print(f"ERROR: navsim_log_path not found: {navsim_log_path}{hint}", file=sys.stderr)
        return 2
    if not sensor_blobs_path.exists():
        root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
        hint = ""
        if root:
            hint = f" (available splits under sensor_blobs: {_list_subdirs(Path(root) / 'sensor_blobs')})"
        print(f"ERROR: sensor_blobs_path not found: {sensor_blobs_path}{hint}", file=sys.stderr)
        return 2
    if not metric_cache_path.exists():
        print(f"ERROR: metric_cache_path not found: {metric_cache_path}", file=sys.stderr)
        return 2

    # Use metric_cache metadata to infer which log (and split) this token belongs to.
    log_name = None
    openscene_root = (os.environ.get("OPENSCENE_DATA_ROOT") or "").strip()
    if openscene_root:
        try:
            mc_file = _get_metric_cache_file(metric_cache_path, token)
            log_name = _infer_log_name_from_metric_cache_file(mc_file)
            if log_name:
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
                    # Recompute default paths for the detected split (only if user didn't pass explicit paths).
                    if args.navsim_log_path is None:
                        navsim_log_path = Path(_default_navsim_log_path(split)).expanduser().resolve()
                    if args.sensor_blobs_path is None:
                        sensor_blobs_path = Path(_default_sensor_blobs_path(split)).expanduser().resolve()
        except Exception as e:
            print(f"[WARN] failed to infer log/split from metric_cache: {e}", file=sys.stderr)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir is None:
        navsim_exp_root = (os.environ.get("NAVSIM_EXP_ROOT") or "").strip()
        if not navsim_exp_root:
            print(
                "ERROR: --out-dir is required if NAVSIM_EXP_ROOT is not set",
                file=sys.stderr,
            )
            return 2
        out_dir = (Path(navsim_exp_root) / "token_viz").resolve()

    cluster_file_path = (
        Path(args.cluster_file_path).expanduser().resolve() if args.cluster_file_path else None
    )
    sim_reward_dict_path = (
        Path(args.sim_reward_dict_path).expanduser().resolve() if args.sim_reward_dict_path else None
    )

    # Controller paths: default to the same artifacts used by the batch eval script.
    repo_root = Path(__file__).resolve().parents[2]
    default_ctrl_ref = (repo_root / "ControllerExp" / "Anchors_Original_256_centered.npy").resolve()
    default_ctrl_exec = (repo_root / "ControllerExp" / "generated" / "controller_styles.npz").resolve()
    default_ctrl_rp = (repo_root / "ControllerExp" / "generated" / "controller_response_predictor.pt").resolve()

    controller_ref_traj_path = (
        Path(args.controller_ref_traj_path).expanduser().resolve()
        if args.controller_ref_traj_path
        else (default_ctrl_ref if default_ctrl_ref.exists() else None)
    )
    controller_exec_traj_path = (
        Path(args.controller_exec_traj_path).expanduser().resolve()
        if args.controller_exec_traj_path
        else (default_ctrl_exec if default_ctrl_exec.exists() else None)
    )
    controller_response_predictor_path = (
        Path(args.controller_response_predictor_path).expanduser().resolve()
        if args.controller_response_predictor_path
        else (default_ctrl_rp if default_ctrl_rp.exists() else None)
    )

    config_dir = (
        Path(__file__).resolve().parents[2]
        / "navsim"
        / "planning"
        / "script"
        / "config"
        / "pdm_scoring"
    )

    if not config_dir.exists():
        print(f"ERROR: hydra config dir not found: {config_dir}", file=sys.stderr)
        return 2

    # Ensure Hydra is in a clean state (important for scripts invoked multiple times).
    try:
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
    except Exception:
        pass

    try:
        from hydra import initialize_config_dir
    except Exception as e:
        print(f"ERROR: failed to import hydra: {e}", file=sys.stderr)
        return 2

    # Fail early with a more actionable error if the current env is missing deps.
    try:
        import torch  # noqa: F401
        import pytorch_lightning  # noqa: F401
    except Exception as e:
        print(
            "ERROR: python env is missing required ML deps (torch/pytorch_lightning/etc).\n"
            "Hint: run this script with the same conda env you use for training/eval (e.g. wotenewnew),\n"
            "and/or install the missing pip/conda packages.\n"
            f"Root error: {e}",
            file=sys.stderr,
        )
        return 2

    # Determinism (best-effort). Note: simulator/controller may still have non-determinism depending on GPU/cuDNN.
    try:
        import numpy as np

        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    except Exception:
        pass

    # Optional: fix controller + simulator style index to match eval logs.
    if args.style_idx is not None:
        idx = int(args.style_idx)
        os.environ.setdefault("WOTE_CTRL_STYLE_SPLIT", "val")
        os.environ.setdefault("WOTE_CTRL_EVAL_SAMPLE", "0")
        os.environ["WOTE_CTRL_STYLE_IDX"] = str(idx)
        # Simulator bundle env (PDMSimulator reads these)
        # If you want a different bundle, set PDM_SIM_BUNDLE_PATH in your shell.
        os.environ.setdefault(
            "PDM_SIM_BUNDLE_PATH",
            str(controller_exec_traj_path) if controller_exec_traj_path is not None else str(default_ctrl_exec),
        )
        os.environ.setdefault("PDM_SIM_BUNDLE_APPLY", "all")
        os.environ.setdefault("PDM_SIM_EVAL_SAMPLE", "0")
        os.environ["PDM_SIM_STYLE_IDX"] = str(idx)

    print("\n=== Run config summary (for score alignment) ===")
    print(f"seed: {args.seed}")
    print(f"sim.post_params.apply_mode: {args.sim_post_apply_mode}")
    print(f"baseline: ctrl_wm={int(args.baseline_ctrl_wm)}")
    print(
        "film:     "
        f"ctrl_wm={int(args.film_ctrl_wm)}, "
        f"wm_fusion={args.film_wm_fusion}, "
        f"wm_strength={args.film_wm_strength}"
    )
    if args.style_idx is not None:
        print(
            "style_idx fixed via env: "
            f"WOTE_CTRL_STYLE_IDX={os.environ.get('WOTE_CTRL_STYLE_IDX')}, "
            f"PDM_SIM_STYLE_IDX={os.environ.get('PDM_SIM_STYLE_IDX')}, "
            f"PDM_SIM_BUNDLE_PATH={os.environ.get('PDM_SIM_BUNDLE_PATH')}"
        )
    else:
        print(
            "style_idx env not forced (scores may differ vs CSV if eval used fixed style): "
            f"WOTE_CTRL_STYLE_IDX={os.environ.get('WOTE_CTRL_STYLE_IDX')}, "
            f"PDM_SIM_STYLE_IDX={os.environ.get('PDM_SIM_STYLE_IDX')}"
        )
    if cluster_file_path is not None:
        print(f"cluster_file_path: {cluster_file_path}")
    if sim_reward_dict_path is not None:
        print(f"sim_reward_dict_path: {sim_reward_dict_path}")
    if controller_ref_traj_path is not None:
        print(f"controller_ref_traj_path: {controller_ref_traj_path}")
    if controller_exec_traj_path is not None:
        print(f"controller_exec_traj_path: {controller_exec_traj_path}")
    if controller_response_predictor_path is not None:
        print(f"controller_response_predictor_path: {controller_response_predictor_path}")
    print("=== End summary ===\n")

    def _hydra_init_ctx(p: Path):
        """Hydra API compatibility shim.

        Some Hydra versions don't support `version_base` for initialize_config_dir.
        """

        try:
            return initialize_config_dir(config_dir=str(p), version_base=None)
        except TypeError:
            return initialize_config_dir(config_dir=str(p))

    with _hydra_init_ctx(config_dir):
        cfg_baseline = _compose_cfg(
            config_dir=config_dir,
            split=split,
            token=token,
            output_dir=out_dir,
            navsim_log_path=navsim_log_path,
            sensor_blobs_path=sensor_blobs_path,
            metric_cache_path=metric_cache_path,
            ckpt_path=baseline_ckpt,
            cluster_file_path=cluster_file_path,
            sim_reward_dict_path=sim_reward_dict_path,
            controller_ref_traj_path=controller_ref_traj_path,
            controller_exec_traj_path=controller_exec_traj_path,
            controller_response_predictor_path=controller_response_predictor_path,
            simulator_post_apply_mode=args.sim_post_apply_mode,
            ctrl_enable_world_model=bool(args.baseline_ctrl_wm),
            ctrl_world_model_fusion=None,
            ctrl_world_model_strength=None,
        )
        cfg_film = _compose_cfg(
            config_dir=config_dir,
            split=split,
            token=token,
            output_dir=out_dir,
            navsim_log_path=navsim_log_path,
            sensor_blobs_path=sensor_blobs_path,
            metric_cache_path=metric_cache_path,
            ckpt_path=film_ckpt,
            cluster_file_path=cluster_file_path,
            sim_reward_dict_path=sim_reward_dict_path,
            controller_ref_traj_path=controller_ref_traj_path,
            controller_exec_traj_path=controller_exec_traj_path,
            controller_response_predictor_path=controller_response_predictor_path,
            simulator_post_apply_mode=args.sim_post_apply_mode,
            ctrl_enable_world_model=bool(args.film_ctrl_wm),
            ctrl_world_model_fusion=str(args.film_wm_fusion),
            ctrl_world_model_strength=float(args.film_wm_strength),
        )

        # Run both.
        b_label = f"baseline_{baseline_ckpt.stem}"
        f_label = f"film_{film_ckpt.stem}"

        res_baseline = _run_one(
            cfg=cfg_baseline,
            token=token,
            out_dir=out_dir,
            out_name=f"{token}_{b_label}.png",
            log_name=log_name,
            compute_isolated_scores=bool(args.compute_isolated_scores),
        )
        res_film = _run_one(
            cfg=cfg_film,
            token=token,
            out_dir=out_dir,
            out_name=f"{token}_{f_label}.png",
            log_name=log_name,
            compute_isolated_scores=bool(args.compute_isolated_scores),
        )

    # Minimal stdout summary for quick sanity check.
    print("=== token_viz done ===")
    print(f"token: {token}")
    print(f"baseline image: {res_baseline['out']}")
    print(f"film image:     {res_film['out']}")
    print(f"baseline pdm_pred.score (fast/CSV): {res_baseline['pdm_pred'].get('score')}")
    if 'pdm_pred_full' in res_baseline:
        print(f"baseline pdm_pred.score (full/viz): {res_baseline['pdm_pred_full'].get('score')}")
    if res_baseline.get('isolated_scores_path'):
        print(f"baseline isolated scores:          {res_baseline['isolated_scores_path']}")
    print(f"film pdm_pred.score (fast/CSV):     {res_film['pdm_pred'].get('score')}")
    if 'pdm_pred_full' in res_film:
        print(f"film pdm_pred.score (full/viz):     {res_film['pdm_pred_full'].get('score')}")
    if res_film.get('isolated_scores_path'):
        print(f"film isolated scores:              {res_film['isolated_scores_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
