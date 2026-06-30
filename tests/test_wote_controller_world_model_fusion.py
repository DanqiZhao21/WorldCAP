from types import SimpleNamespace
import sys
import types

import torch

stub = types.ModuleType("navsim.agents.WoTE.WoTE_targets")


class BoundingBox2DIndex:
    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4
    X = _X
    Y = _Y
    HEADING = _HEADING
    LENGTH = _LENGTH
    WIDTH = _WIDTH
    POINT = slice(_X, _Y + 1)
    STATE_SE2 = slice(_X, _HEADING + 1)

    @classmethod
    def size(cls):
        return 5


stub.BoundingBox2DIndex = BoundingBox2DIndex
sys.modules["navsim.agents.WoTE.WoTE_targets"] = stub

from navsim.agents.WoTE.WoTE_model import WoTEModel


class _IdentityControllerEncoder(torch.nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, ref_traj, exec_traj):
        n_ctrl = ref_traj.shape[0]
        return torch.arange(
            n_ctrl * self.emb_dim,
            device=ref_traj.device,
            dtype=ref_traj.dtype,
        ).view(n_ctrl, self.emb_dim)


def _light_model(fusion: str = "attn_film", strength: float = 1.0):
    model = object.__new__(WoTEModel)
    torch.nn.Module.__init__(model)
    model._config = SimpleNamespace(
        controller_condition_on_world_model=True,
        controller_world_model_fusion=fusion,
        controller_world_model_strength=strength,
        controller_world_model_inject_target="all",
        controller_world_model_inject_first_step_only=False,
    )
    model.controller_condition_on_world_model = True
    model.controller_world_model_strength = strength
    model.controller_world_model_inject_target = "all"
    model.controller_world_model_inject_first_step_only = False
    model.controller_emb_dim = 64
    model.controller_encoder = _IdentityControllerEncoder(model.controller_emb_dim)
    model.ctrl_bank_proj = torch.nn.Linear(model.controller_emb_dim, 256)
    model.ctrl_bank_ln = torch.nn.LayerNorm(256)
    model.ctrl_fuse_attn = torch.nn.MultiheadAttention(256, 8, batch_first=True)
    model.ctrl_wm_film_scale = torch.nn.Linear(256, 256)
    model.ctrl_wm_film_shift = torch.nn.Linear(256, 256)
    model.ctrl_wm_film_ln = torch.nn.LayerNorm(256)
    model.scene_position_embedding = torch.nn.Embedding(65, 256)
    model.latent_world_model = torch.nn.Identity()
    return model


def test_controller_bank_tokens_follow_active_bank_size():
    model = _light_model()
    model._active_ref_trajs = torch.zeros((128, 8, 3))
    model._active_exec_trajs = torch.ones((128, 8, 3))

    bank_tokens = model._compute_controller_bank_tokens(batch_size=2, num_traj=3, device=torch.device("cpu"))

    assert bank_tokens.shape == (6, 128, 256)


def test_attn_film_modulates_scene_tokens_with_dynamic_controller_bank():
    model = _light_model(fusion="attn_film")
    model._active_ref_trajs = torch.zeros((64, 8, 3))
    model._active_exec_trajs = torch.ones((64, 8, 3))

    fut_ego, fut_bev = model._latent_world_model_processing(
        flatten_bev_feature_multi_trajs=torch.zeros((2, 64, 256)),
        ego_feat=torch.zeros((2, 1, 256)),
        batch_size=1,
        num_traj=2,
        wm_step=0,
    )

    assert fut_ego.shape == (2, 1, 256)
    assert fut_bev.shape == (2, 64, 256)


def test_attn_film_does_not_use_manual_strength_gate():
    torch.manual_seed(0)
    model = _light_model(fusion="attn_film", strength=0.0)
    model._active_ref_trajs = torch.zeros((16, 8, 3))
    model._active_exec_trajs = torch.ones((16, 8, 3))
    scene_pos = model.scene_position_embedding.weight.unsqueeze(0).expand(1, -1, -1)

    fut_ego, fut_bev = model._latent_world_model_processing(
        flatten_bev_feature_multi_trajs=torch.zeros((1, 64, 256)),
        ego_feat=torch.zeros((1, 1, 256)),
        batch_size=1,
        num_traj=1,
        wm_step=0,
    )
    output_scene = torch.cat([fut_ego, fut_bev], dim=1)

    assert not torch.allclose(output_scene, scene_pos)


def test_reward_rollout_steps_follow_configured_num_fut_timestep():
    model = object.__new__(WoTEModel)
    torch.nn.Module.__init__(model)
    model._config = SimpleNamespace(controller_condition_on_world_model=False)
    model.num_fut_timestep = 1
    model.use_map_loss = False
    model.num_plan_queries = 64
    model._compose_future_bev_targets_from_base = lambda targets: None
    model._inject_cur_ego_into_bev = lambda scene, ego, num_traj: scene
    model._inject_fut_ego_into_bev = lambda scene, ego, num_traj, fut_idx: scene
    model._latent_world_model_processing = (
        lambda bev, ego, batch_size, num_traj, **kwargs: (ego, bev)
    )
    seen_lengths = []

    def _compute_reward_feature(ego_feat_list, bev_feat_list, batch_size, num_traj):
        seen_lengths.append((len(ego_feat_list), len(bev_feat_list)))
        return torch.zeros((batch_size, num_traj, 256))

    model._compute_reward_feature = _compute_reward_feature
    trajectory_outputs = {
        "results": {},
        "batch_size": 1,
        "num_traj": 1,
        "flatten_bev_feature": torch.zeros((1, 64, 256)),
        "ego_feat": torch.zeros((1, 1, 1, 256)),
    }

    model.extract_reward_feature(trajectory_outputs, targets=None)

    assert seen_lengths == [(2, 2)]
