import gzip
import os
from pathlib import Path

import logging
import pickle
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch

import psutil
import sys
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

logger = logging.getLogger(__name__)
MEMORY_THRESHOLD_GB = 20

def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)

def check_memory_and_exit():
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    if available_gb < MEMORY_THRESHOLD_GB:
        print(f"⚠️  Available memory is low: {available_gb:.2f}GB. Exiting to avoid OOM.")
        sys.exit(1)  # 安全退出
        
        
class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
        use_fut_frames: bool = False,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()
        
        self.use_fut_frames = use_fut_frames

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(
        self, token: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
            """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
        #TODO:
            check_memory_and_exit()
            self._cache_scene_with_token(token)

    def __len__(self):
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input(self.use_fut_frames)
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        return (features, targets)

#===============================或许我发现也许可以所有的都cache避免动态计算还是会很慢=================================
#TODO:自己重新写的一个Dataset类型
# class Dataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         scene_loader: SceneLoader,
#         feature_builders: List[AbstractFeatureBuilder],
#         target_builders: List[AbstractTargetBuilder],
#         cache_path: Optional[str] = None,
#         force_cache_computation: bool = False,
#         use_fut_frames: bool = False,
#     ):
#         super().__init__()
#         self._scene_loader = scene_loader
#         self._feature_builders = feature_builders
#         self._target_builders = target_builders
#         self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
#         self._force_cache_computation = force_cache_computation

#         self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
#             self._cache_path, feature_builders, target_builders
#         )

#         if self._cache_path is not None:
#             self.cache_dataset()

#         self.use_fut_frames = use_fut_frames

#     @staticmethod
#     def _load_valid_caches(
#         cache_path: Optional[Path],
#         feature_builders: List[AbstractFeatureBuilder],
#         target_builders: List[AbstractTargetBuilder],
#     ) -> Dict[str, Path]:
#         """Only check feature caches; target caches may be partial."""
#         valid_cache_paths: Dict[str, Path] = {}
#         if (cache_path is not None) and cache_path.is_dir():
#             for log_path in cache_path.iterdir():
#                 for token_path in log_path.iterdir():
#                     found_features = []
#                     for builder in feature_builders+ target_builders:#仍然检查是否target缓存文件
#                         data_dict_path = token_path / (builder.get_unique_name() + ".gz")
#                         found_features.append(data_dict_path.is_file())
#                     if all(found_features):
#                         valid_cache_paths[token_path.name] = token_path
#         return valid_cache_paths

#     def _cache_scene_with_token(self, token: str) -> None:
#         """Cache both feature and target. Targets may be partial."""
#         scene = self._scene_loader.get_scene_from_token(token)
#         agent_input = scene.get_agent_input()

#         metadata = scene.scene_metadata
#         token_path = self._cache_path / metadata.log_name / metadata.initial_token
#         os.makedirs(token_path, exist_ok=True)

#         # cache features (always complete)
#         for builder in self._feature_builders:
#             data_dict_path = token_path / (builder.get_unique_name() + ".gz")
#             data_dict = builder.compute_features(agent_input)
#             dump_feature_target_to_pickle(data_dict_path, data_dict)

#         # # cache targets (partial)
#         # for builder in self._target_builders:
#         #     data_dict_path = token_path / (builder.get_unique_name() + ".gz")
#         #     data_dict = builder.compute_targets(scene)  # may contain only reusable keys
#         #     dump_feature_target_to_pickle(data_dict_path, data_dict)
#         # cache targets (only static/reusable keys)
#         for builder in self._target_builders:
#             data_dict_path = token_path / (builder.get_unique_name() + ".gz")
#             all_targets = builder.compute_targets(scene)
#             # 只保留可复用的 target
#             #可能有的： result['sim_reward']+ result["agent_states"]+  result["agent_labels"] +  result["bev_semantic_map"]
            
#             #每个都有的： result['trajectory']  result["fut_agent_labels"] result["fut_agent_states"]
#             #需要动态缓存的：result["sampled_trajs_index"]  result["fut_bev_semantic_map"]
#             static_keys = ["sim_reward", "agent_states","agent_labels", "bev_semantic_map", "trajectory","fut_agent_labels", "fut_agent_states","fut_bev_semantic_map_base"]
#             static_targets = {k: v for k, v in all_targets.items() if k in static_keys}
#             dump_feature_target_to_pickle(data_dict_path, static_targets)

#         self._valid_cache_paths[token] = token_path


#     def _load_scene_with_token(
#         self, token: str
#     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#         """Load features and targets, compute missing target keys dynamically."""
#         token_path = self._valid_cache_paths[token]

#         # Load features (always complete)
#         features: Dict[str, torch.Tensor] = {}
#         for builder in self._feature_builders:
#             data_dict_path = token_path / (builder.get_unique_name() + ".gz")
#             data_dict = load_feature_target_from_pickle(data_dict_path)
#             features.update(data_dict)

#         # Load targets (may be partial)
#         targets: Dict[str, torch.Tensor] = {}
#         scene = None  # only load if necessary
#         for builder in self._target_builders:
#             data_dict_path = token_path / (builder.get_unique_name() + ".gz")
#             if data_dict_path.is_file():
#                 data_dict = load_feature_target_from_pickle(data_dict_path)
#             else:
#                 data_dict = {}

#             # Determine missing keys
#             missing_keys = []
#             for key in ["fut_bev_semantic_map", "sampled_trajs_index"]:
#                 if key not in data_dict:
#                     missing_keys.append(key)

#             # Compute missing keys dynamically
#             if missing_keys:
#                 if scene is None:
#                     scene = self._scene_loader.get_scene_from_token(token)
#                 # computed_targets = builder.compute_targets_partial(scene, missing_keys)
#                 computed_targets = builder.compute_targets_partial(scene, missing_keys, cached_targets=data_dict)
#                 data_dict.update(computed_targets)

#             targets.update(data_dict)

#         return features, targets

#     def cache_dataset(self) -> None:
#         assert self._cache_path is not None, "Dataset did not receive a cache path!"
#         os.makedirs(self._cache_path, exist_ok=True)

#         if self._force_cache_computation:
#             tokens_to_cache = self._scene_loader.tokens
#         else:
#             tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
#             tokens_to_cache = list(tokens_to_cache)
#             logger.info(
#                 f"Starting caching of {len(tokens_to_cache)} tokens. "
#                 "Note: Caching tokens within the training loader is slow. "
#                 "Use `run_dataset_caching.py` for large numbers of tokens."
#             )

#         for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
#             self._cache_scene_with_token(token)

#     def __len__(self):
#         return len(self._scene_loader)

#     def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#         token = self._scene_loader.tokens[idx]
#         if self._cache_path is not None:
#             assert token in self._valid_cache_paths.keys(), \
#                 f"The token {token} has not been cached yet, please call cache_dataset first!"
#             features, targets = self._load_scene_with_token(token)
#         else:
#             scene = self._scene_loader.get_scene_from_token(token)
#             agent_input = scene.get_agent_input(self.use_fut_frames)
#             features = {}
#             targets = {}
#             for builder in self._feature_builders:
#                 features.update(builder.compute_features(agent_input))
#             for builder in self._target_builders:
#                 targets.update(builder.compute_targets(scene))

#         return features, targets