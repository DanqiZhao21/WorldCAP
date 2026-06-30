from __future__ import annotations

import argparse
import lzma
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from tool.visualize.wote_pdm_analysis import (
    compute_spearman_rho,
    plot_sorted_distribution,
    rank_trajectories_left_to_right,
    save_analysis_artifacts,
)


def _default_navsim_log_path(split: str) -> Path:
    root = os.environ.get("OPENSCENE_DATA_ROOT")
    if not root:
        raise RuntimeError("OPENSCENE_DATA_ROOT is required or pass --navsim-log-path")
    return Path(root) / "navsim_logs" / split


def _default_sensor_blobs_path(split: str) -> Path:
    root = os.environ.get("OPENSCENE_DATA_ROOT")
    if not root:
        raise RuntimeError("OPENSCENE_DATA_ROOT is required or pass --sensor-blobs-path")
    return Path(root) / "sensor_blobs" / split


def _default_metric_cache_path() -> Path:
    root = os.environ.get("NAVSIM_EXP_ROOT")
    if not root:
        raise RuntimeError("NAVSIM_EXP_ROOT is required or pass --metric-cache-path")
    return Path(root) / "metric_cache"


def _build_agent(ckpt_path: Path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    from navsim.agents.WoTE.WoTE_agent import WoTEAgent

    config_dir = Path(__file__).resolve().parents[2] / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass
    try:
        ctx = initialize_config_dir(config_dir=str(config_dir), version_base=None)
    except TypeError:
        ctx = initialize_config_dir(config_dir=str(config_dir))
    with ctx:
        cfg = compose(
            config_name="default_run_pdm_score",
            overrides=[
                "agent=WoTE_agent",
                "worker=sequential",
                f"agent.checkpoint_path='{ckpt_path}'",
            ],
        )
    agent: WoTEAgent = instantiate(cfg.agent)
    agent._checkpoint_path = str(ckpt_path)
    agent.initialize()
    agent.is_eval = True
    return agent


def _build_scene_loader(
    *,
    token: str,
    split: str,
    navsim_log_path: Path,
    sensor_blobs_path: Path,
    agent,
):
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=1,
        log_names=None,
        tokens=[token],
    )
    return SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )


def _build_simulator_and_scorer():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    config_dir = Path(__file__).resolve().parents[2] / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
    if not config_dir.exists():
        raise FileNotFoundError(config_dir)
    try:
        GlobalHydra.instance().clear()
    except Exception:
        pass
    try:
        ctx = initialize_config_dir(config_dir=str(config_dir), version_base=None)
    except TypeError:
        ctx = initialize_config_dir(config_dir=str(config_dir))
    with ctx:
        cfg = compose(config_name="default_run_pdm_score", overrides=["worker=sequential"])

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer_cfg = {
        "_target_": cfg.scorer._target_,
        "_convert_": getattr(cfg.scorer, "_convert_", "all"),
        "proposal_sampling": cfg.scorer.proposal_sampling,
        "config": cfg.scorer.config,
    }
    scorer: PDMScorer = instantiate(scorer_cfg)
    return simulator, scorer


def _load_metric_cache(metric_cache_path: Path, token: str):
    from navsim.common.dataloader import MetricCacheLoader

    loader = MetricCacheLoader(metric_cache_path)
    metric_cache_file = loader.metric_cache_paths[token]
    with lzma.open(metric_cache_file, "rb") as f:
        return pickle.load(f)


def _collect_full_scores(metric_cache, agentout, simulator, scorer):
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory

    initial_ego_state = metric_cache.ego_state
    pdm_trajectory = metric_cache.trajectory
    pdm_states = get_trajectory_as_array(
        pdm_trajectory, simulator.proposal_sampling, initial_ego_state.time_point
    )

    pred_input_states_list = []
    for traj in agentout["trajectories"]:
        pred_traj = transform_trajectory(traj, initial_ego_state)
        pred_states = get_trajectory_as_array(
            pred_traj, simulator.proposal_sampling, initial_ego_state.time_point
        )
        pred_input_states_list.append(pred_states)

    pred_input_all = np.stack(pred_input_states_list, axis=0)
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_input_all], axis=0)
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )
    return np.asarray(scores[1:], dtype=np.float64), simulated_states[1:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--navsim-log-path", default=None)
    parser.add_argument("--sensor-blobs-path", default=None)
    parser.add_argument("--metric-cache-path", default=None)
    args = parser.parse_args()

    token = args.token
    split = args.split
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    navsim_log_path = Path(args.navsim_log_path).expanduser().resolve() if args.navsim_log_path else _default_navsim_log_path(split)
    sensor_blobs_path = Path(args.sensor_blobs_path).expanduser().resolve() if args.sensor_blobs_path else _default_sensor_blobs_path(split)
    metric_cache_path = Path(args.metric_cache_path).expanduser().resolve() if args.metric_cache_path else _default_metric_cache_path()

    agent = _build_agent(ckpt_path)
    simulator, scorer = _build_simulator_and_scorer()
    metric_cache = _load_metric_cache(metric_cache_path, token)
    scene_loader = _build_scene_loader(
        token=token,
        split=split,
        navsim_log_path=navsim_log_path,
        sensor_blobs_path=sensor_blobs_path,
        agent=agent,
    )

    agent_input = scene_loader.get_agent_input_from_token(token, use_fut_frames=False)
    scene = scene_loader.get_scene_from_token(token)
    agentout = agent.compute_trajectory(agent_input, scene) if agent.requires_scene else agent.compute_trajectory(agent_input)

    wote_scores = np.asarray(agentout["trajectory_scores"], dtype=np.float64).reshape(-1)
    trajectories = np.asarray([traj.poses for traj in agentout["trajectories"]], dtype=np.float64)
    pdm_scores, _simulated_states = _collect_full_scores(metric_cache, agentout, simulator, scorer)

    order = rank_trajectories_left_to_right(trajectories)
    rho = compute_spearman_rho(wote_scores, pdm_scores)

    rows = []
    for rank_idx, traj_idx in enumerate(order.tolist()):
        rows.append(
            {
                "rank": rank_idx,
                "trajectory_index": traj_idx,
                "wote_score": float(wote_scores[traj_idx]),
                "pdm_score": float(pdm_scores[traj_idx]),
            }
        )

    csv_path, npz_path = save_analysis_artifacts(
        out_dir,
        {
            "rows": rows,
            "trajectories": trajectories,
            "wote_scores": wote_scores,
            "pdm_scores": pdm_scores,
            "rank_order": order,
            "spearman_rho": rho,
        },
    )
    plot_path = plot_sorted_distribution(
        out_dir / f"{token}_wote_pdm_distribution.png",
        trajectories=trajectories,
        wote_scores=wote_scores,
        pdm_scores=pdm_scores,
    )

    print(f"saved_csv={csv_path}")
    print(f"saved_npz={npz_path}")
    print(f"saved_plot={plot_path}")
    print(f"spearman_rho={rho:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
