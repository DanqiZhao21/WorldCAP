import pandas as pd
from tqdm import tqdm
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from dataclasses import asdict
from datetime import datetime
import logging
import lzma
import pickle
import os
import uuid

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataclasses import SensorConfig


from navsim.evaluate.pdm_score import pdm_score_multiTraj

from navsim.visualization.plots import plot_bev_with_agent_and_simulation

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"💚Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"💚Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    single_eval = getattr(cfg, 'single_eval', False)
    # single-threaded worker_map
    if single_eval:
        print("Running single-threaded worker_map")
        score_rows = run_pdm_score(data_points)
    else:
    # mutli-threaded worker_map
        score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)
    

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}. 
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
    """)
#FIXME:
def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    # Sanitize scorer config to avoid unexpected root-level keys being passed in
    scorer_cfg = {
        "_target_": cfg.scorer._target_,
        "_convert_": getattr(cfg.scorer, "_convert_", "all"),
        "proposal_sampling": cfg.scorer.proposal_sampling,
        "config": cfg.scorer.config,
    }
    scorer: PDMScorer = instantiate(scorer_cfg)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.is_eval = True

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter =instantiate(cfg.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            
            # print("💚💚Metric cache path:", metric_cache_path)
            # print("💚💚Metric cache dir:", os.path.dirname(metric_cache_path))
            
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            
            use_future_frames = agent.config.use_fut_frames if hasattr(agent.config, 'use_fut_frames') else False
            agent_input = scene_loader.get_agent_input_from_token(token, use_fut_frames=use_future_frames)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                agentout = agent.compute_trajectory(agent_input, scene)
            else:
                agentout = agent.compute_trajectory(agent_input)
            
            # pdm_result = pdm_score(
            #     metric_cache=metric_cache,
            #     model_trajectory=trajectory,
            #     future_sampling=simulator.proposal_sampling,
            #     simulator=simulator,
            #     scorer=scorer,
            # )
            
            # Resolve anchor saving options from root-level overrides or scorer defaults
            anchor_save_dir = getattr(cfg, 'anchor_save_dir', None) or getattr(cfg.scorer, 'anchor_save_dir', None)
            anchor_save_name = getattr(cfg, 'anchor_save_name', None) or getattr(cfg.scorer, 'anchor_save_name', None)
            anchor_overwrite = getattr(cfg, 'anchor_overwrite', None)
            if anchor_overwrite is None:
                anchor_overwrite = getattr(cfg.scorer, 'anchor_overwrite', False)

            # logger.info(f"Anchor save opts → dir: {anchor_save_dir}, name: {anchor_save_name}, overwrite: {anchor_overwrite}")

            pdm_result, pdm_best_result, simulation_states_all_noHuman, pred_states_all_noHuman, best_pdm_idx,pred_idx =pdm_score_multiTraj(
                metric_cache=metric_cache,
                model_trajectories=agentout["trajectories"],
                trajectory_Anchor=agentout["trajectoryAnchor"],
                model_scores=agentout["trajectory_scores"],#模型在256条轨迹上的得分（predict的 不是用pdm算的）
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                anchor_save_dir=anchor_save_dir,
                anchor_save_name=anchor_save_name,
                anchor_overwrite=anchor_overwrite,
            )
            # print(f"💚💚💚 Best PDM idx for token {token} is {best_pdm_idx},   Model_best idx for  is {pred_idx}")
            scene = scene_loader.get_scene_from_token(token)
            logger.warning(f"----------- ☀️☀️☀️Agent for token {token}:")
#PRINT           
            # human_traj = scene.get_future_trajectory()
            # plot_bev_with_agent_and_simulation(
            #     scene,
            #     agent,
            #     best_pdm_idx=best_pdm_idx,
            #     best_pred_idx=pred_idx,
            #     human_trajectory=human_traj,
            #     agent_trajectory=agentout,
            #     simulation_state=simulation_states_all_noHuman,
            #     pdm_result_pred=pdm_result,
            #     pdm_result_best=pdm_best_result,
            #     # oncoming_mask_pred=oncoming_mask_pred,
            #     # oncoming_mask_best=oncoming_mask_best,
            #     save_path="/home/zhaodanqi/clone/WoTE/EvaluationResult/bev0109",
            #     file_name=f"{token}_bev.png"
            # )
            # logger.warning(f"----------- 🌈🌈🌈Agent for token {token}:")
#PRINT           
            
            score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- ⛈️⛈️⛈️Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
        
        

    return pdm_results

if __name__ == "__main__":
    main()
