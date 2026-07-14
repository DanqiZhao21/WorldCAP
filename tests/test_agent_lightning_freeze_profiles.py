import os
import sys
import types
from types import SimpleNamespace

import torch

pl_stub = types.ModuleType("pytorch_lightning")
pl_stub.LightningModule = torch.nn.Module
sys.modules.setdefault("pytorch_lightning", pl_stub)

abstract_agent_stub = types.ModuleType("navsim.agents.abstract_agent")


class AbstractAgent:
    pass


abstract_agent_stub.AbstractAgent = AbstractAgent
sys.modules.setdefault("navsim.agents.abstract_agent", abstract_agent_stub)

from navsim.planning.training.agent_lightning_module import AgentLightningModule


class FakeAgent:
    def __init__(self):
        self.config = SimpleNamespace(
            freeze_all=True,
            trainable_groups=[],
            trainable_prefixes=[],
            frozen_prefixes=[],
            freeze_strict=True,
        )
        names = [
            "WoTE_model.controller_encoder.feat_proj.0.weight",
            "WoTE_model.ctrl_proj.weight",
            "WoTE_model.ctrl_token_ln.weight",
            "WoTE_model.ctrl_bank_proj.weight",
            "WoTE_model.ctrl_bank_ln.weight",
            "WoTE_model.ctrl_fuse_attn.in_proj_weight",
            "WoTE_model.ctrl_wm_film_scale.weight",
            "WoTE_model.ctrl_wm_film_shift.weight",
            "WoTE_model.ctrl_wm_film_ln.weight",
            "WoTE_model.latent_world_model.weight",
            "WoTE_model.reward_conv_net.conv1.weight",
            "WoTE_model.reward_cat_head.0.weight",
            "WoTE_model.reward_head.0.weight",
            "WoTE_model.sim_reward_heads.0.0.weight",
            "WoTE_model._bev_upscale.weight",
            "WoTE_model.bev_upsample_head.up_conv5.weight",
            "WoTE_model.bev_semantic_head.0.weight",
            "WoTE_model._backbone.weight",
            "WoTE_model.scene_position_embedding.weight",
            "WoTE_model.encode_ego_feat_mlp.weight",
            "WoTE_model.offset_tf_decoder.layers.0.self_attn.in_proj_weight",
            "WoTE_model.offset_head.weight",
            "WoTE_model.offset_score_head.0.weight",
        ]
        self.params = [(name, torch.nn.Parameter(torch.zeros(1))) for name in names]

    def named_parameters(self):
        return iter(self.params)


def _module_with_fake_agent():
    module = object.__new__(AgentLightningModule)
    module.agent = FakeAgent()
    return module


def test_trainable_groups_unfreeze_controller_latent_and_reward_heads():
    module = _module_with_fake_agent()
    module.agent.config.trainable_groups = [
        "controller_embedding",
        "latent_world_model",
        "reward_heads",
    ]

    module.setup()

    trainable = {name for name, param in module.agent.named_parameters() if param.requires_grad}
    assert trainable == {
        "WoTE_model.controller_encoder.feat_proj.0.weight",
        "WoTE_model.ctrl_proj.weight",
        "WoTE_model.ctrl_token_ln.weight",
        "WoTE_model.ctrl_bank_proj.weight",
        "WoTE_model.ctrl_bank_ln.weight",
        "WoTE_model.ctrl_fuse_attn.in_proj_weight",
        "WoTE_model.ctrl_wm_film_scale.weight",
        "WoTE_model.ctrl_wm_film_shift.weight",
        "WoTE_model.ctrl_wm_film_ln.weight",
        "WoTE_model.latent_world_model.weight",
        "WoTE_model.reward_conv_net.conv1.weight",
        "WoTE_model.reward_cat_head.0.weight",
        "WoTE_model.reward_head.0.weight",
        "WoTE_model.sim_reward_heads.0.0.weight",
    }


def test_trainable_and_frozen_prefixes_override_groups():
    module = _module_with_fake_agent()
    module.agent.config.trainable_groups = ["controller_embedding", "reward_heads"]
    module.agent.config.trainable_prefixes = ["WoTE_model.bev_semantic_head"]
    module.agent.config.frozen_prefixes = [
        "WoTE_model.ctrl_fuse_attn",
        "WoTE_model.reward_head",
    ]

    module.setup()

    trainable = {name for name, param in module.agent.named_parameters() if param.requires_grad}
    assert "WoTE_model.bev_semantic_head.0.weight" in trainable
    assert "WoTE_model.ctrl_fuse_attn.in_proj_weight" not in trainable
    assert "WoTE_model.reward_head.0.weight" not in trainable
    assert "WoTE_model.reward_cat_head.0.weight" in trainable


def test_freeze_strict_rejects_unknown_group():
    module = _module_with_fake_agent()
    module.agent.config.trainable_groups = ["not_a_group"]

    try:
        module.setup()
    except ValueError as exc:
        assert "Unknown trainable group" in str(exc)
    else:
        raise AssertionError("Expected setup to reject an unknown trainable group")


def test_freeze_strict_rejects_unmatched_prefix():
    module = _module_with_fake_agent()
    module.agent.config.trainable_prefixes = ["WoTE_model.missing_module"]

    try:
        module.setup()
    except ValueError as exc:
        assert "matched no parameters" in str(exc)
    else:
        raise AssertionError("Expected setup to reject an unmatched prefix")
