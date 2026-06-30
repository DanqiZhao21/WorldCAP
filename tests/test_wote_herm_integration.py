import sys
import types
from types import SimpleNamespace

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


class WoTETargetBuilder:
    pass


stub.BoundingBox2DIndex = BoundingBox2DIndex
stub.WoTETargetBuilder = WoTETargetBuilder
sys.modules["navsim.agents.WoTE.WoTE_targets"] = stub

from navsim.agents.WoTE.HERM.model import HERMConfig, HERMOutput, SupportConditionalHERM
from navsim.agents.WoTE.WoTE_model import WoTEModel


class FakeHERM(torch.nn.Module):
    def forward(self, support_plan, support_exec, query_plan):
        offset = (
            support_exec[:, :1, :, :2].mean(dim=1, keepdim=True)
            - support_plan[:, :1, :, :2].mean(dim=1, keepdim=True)
        )
        pred = query_plan.clone()
        pred[..., :2] = pred[..., :2] + offset
        return HERMOutput(
            exec_traj=pred,
            residual=torch.zeros_like(query_plan),
            params=torch.zeros(
                query_plan.shape[0],
                query_plan.shape[1],
                query_plan.shape[2] - 1,
                9,
                device=query_plan.device,
                dtype=query_plan.dtype,
            ),
        )


def _owner():
    owner = SimpleNamespace()
    owner.herm_enable = True
    owner.herm_apply_in_eval = True
    owner.herm_apply_in_train = False
    owner.herm_support_size = 2
    owner.herm_support_seed = 0
    owner.herm_query_chunk_size = 2
    owner.herm_model = FakeHERM()
    owner.herm_checkpoint_path = ""
    owner.training = False
    owner.is_eval = True
    owner._active_ref_trajs = torch.zeros(4, 8, 3)
    owner._active_exec_trajs = torch.zeros(4, 8, 3)
    owner._active_exec_trajs[..., 0] = 1.5
    owner._select_herm_support_indices = types.MethodType(WoTEModel._select_herm_support_indices, owner)
    owner._ensure_herm_loaded = types.MethodType(WoTEModel._ensure_herm_loaded, owner)
    owner._execute_candidates_with_herm = types.MethodType(WoTEModel._execute_candidates_with_herm, owner)
    return owner


def test_support_conditional_herm_checkpoint_loads_model_state_format(tmp_path):
    config = HERMConfig(
        num_poses=8,
        hidden_dim=32,
        num_layers=1,
        controller_emb_dim=16,
        dropout=0.0,
    )
    model = SupportConditionalHERM(config, style_emb_dim=16, style_hidden_dim=24)
    ckpt_path = tmp_path / "herm.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "style_emb_dim": 16,
            "style_hidden_dim": 24,
        },
        ckpt_path,
    )

    owner = SimpleNamespace(
        herm_enable=True,
        herm_model=None,
        herm_checkpoint_path=str(ckpt_path),
        herm_device="cpu",
    )
    owner._load_herm_model = types.MethodType(WoTEModel._load_herm_model, owner)
    owner._ensure_herm_loaded = types.MethodType(WoTEModel._ensure_herm_loaded, owner)

    owner._ensure_herm_loaded()

    assert isinstance(owner.herm_model, SupportConditionalHERM)
    assert owner.herm_model.training is False
    assert all(not p.requires_grad for p in owner.herm_model.parameters())


def test_herm_is_not_registered_before_delayed_load():
    model = object.__new__(WoTEModel)
    torch.nn.Module.__init__(model)
    cfg = SimpleNamespace(
        herm_enable=True,
        herm_apply_in_eval=True,
        herm_apply_in_train=False,
        herm_checkpoint_path="/tmp/not-loaded-yet.pt",
        herm_support_size=768,
        herm_support_seed=0,
        herm_query_chunk_size=256,
        herm_device="cpu",
    )

    model._configure_herm(cfg)

    assert model.herm_model is None
    assert not any(key.startswith("herm_model.") for key in model.state_dict())


def test_execute_candidates_with_herm_preserves_shape_and_uses_support():
    owner = _owner()
    query = torch.zeros(1, 3, 8, 3)

    out = owner._execute_candidates_with_herm(query)

    assert out.shape == query.shape
    assert torch.allclose(out[..., 0], torch.full_like(out[..., 0], 1.5))


def test_execute_candidates_with_herm_disabled_returns_input_object():
    owner = SimpleNamespace(herm_enable=False, herm_model=None)
    owner._execute_candidates_with_herm = types.MethodType(WoTEModel._execute_candidates_with_herm, owner)
    query = torch.zeros(1, 2, 8, 3)

    out = owner._execute_candidates_with_herm(query)

    assert out is query


def test_reward_rollout_exposes_scored_candidates_to_future_ego_injection():
    model = object.__new__(WoTEModel)
    torch.nn.Module.__init__(model)
    model._config = SimpleNamespace(controller_condition_on_world_model=False)
    model.num_fut_timestep = 1
    model.use_map_loss = False
    model.num_plan_queries = 64
    model._compose_future_bev_targets_from_base = lambda targets: None
    model._inject_cur_ego_into_bev = lambda scene, ego, num_traj: scene
    model._latent_world_model_processing = (
        lambda bev, ego, batch_size, num_traj, **kwargs: (ego, bev)
    )
    seen_anchor_value = []

    def _inject_fut(scene, ego, num_traj, fut_idx):
        anchors = getattr(model, "_reward_trajectory_anchors", None)
        seen_anchor_value.append(float(anchors[0, 0, fut_idx - 1, 0].item()))
        return scene

    model._inject_fut_ego_into_bev = _inject_fut
    model._compute_reward_feature = (
        lambda ego_feat_list, bev_feat_list, batch_size, num_traj: torch.zeros((batch_size, num_traj, 256))
    )
    trajectory_outputs = {
        "results": {},
        "batch_size": 1,
        "num_traj": 1,
        "flatten_bev_feature": torch.zeros((1, 64, 256)),
        "ego_feat": torch.zeros((1, 1, 1, 256)),
        "scored_trajectory_anchors": torch.full((1, 1, 8, 3), 7.0),
    }

    model.extract_reward_feature(trajectory_outputs, targets=None)

    assert seen_anchor_value == [7.0]
    assert not hasattr(model, "_reward_trajectory_anchors")
