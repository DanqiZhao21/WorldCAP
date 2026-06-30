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
            controller_injection_mode="film",
            controller_style_pooling="attn",
            controller_world_model_fusion="attn",
        )
        names = [
            "WoTE_model.controller_encoder.feat_proj.0.weight",
            "WoTE_model.ctrl_proj.weight",
            "WoTE_model.ctrl_token_ln.weight",
            "WoTE_model.ctrl_style_attn.weight",
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


def test_controller_wm_fusion_only_freezes_original_wote_modules(monkeypatch):
    monkeypatch.setenv("WOTE_TRAIN_PROFILE", "controller_wm_fusion_only")
    module = _module_with_fake_agent()

    module.setup()

    trainable = {name for name, param in module.agent.named_parameters() if param.requires_grad}
    assert trainable == {
        "WoTE_model.controller_encoder.feat_proj.0.weight",
        "WoTE_model.ctrl_proj.weight",
        "WoTE_model.ctrl_token_ln.weight",
        "WoTE_model.ctrl_style_attn.weight",
        "WoTE_model.ctrl_bank_proj.weight",
        "WoTE_model.ctrl_bank_ln.weight",
        "WoTE_model.ctrl_fuse_attn.in_proj_weight",
    }


def test_controller_wm_fusion_only_trains_attn_film_modules(monkeypatch):
    monkeypatch.setenv("WOTE_TRAIN_PROFILE", "controller_wm_fusion_only")
    module = _module_with_fake_agent()
    module.agent.config.controller_world_model_fusion = "attn_film"

    module.setup()

    trainable = {name for name, param in module.agent.named_parameters() if param.requires_grad}
    assert trainable == {
        "WoTE_model.controller_encoder.feat_proj.0.weight",
        "WoTE_model.ctrl_proj.weight",
        "WoTE_model.ctrl_token_ln.weight",
        "WoTE_model.ctrl_style_attn.weight",
        "WoTE_model.ctrl_bank_proj.weight",
        "WoTE_model.ctrl_bank_ln.weight",
        "WoTE_model.ctrl_fuse_attn.in_proj_weight",
        "WoTE_model.ctrl_wm_film_scale.weight",
        "WoTE_model.ctrl_wm_film_shift.weight",
        "WoTE_model.ctrl_wm_film_ln.weight",
    }


def test_controller_wm_fusion_only_trains_attn_film_modules_with_current_config_name(monkeypatch):
    monkeypatch.setenv("WOTE_TRAIN_PROFILE", "controller_wm_fusion_only")
    module = _module_with_fake_agent()
    delattr(module.agent.config, "controller_world_model_fusion")
    module.agent.config.controller_wm_fusion = "attn_film"

    module.setup()

    trainable = {name for name, param in module.agent.named_parameters() if param.requires_grad}
    assert {
        "WoTE_model.ctrl_wm_film_scale.weight",
        "WoTE_model.ctrl_wm_film_shift.weight",
        "WoTE_model.ctrl_wm_film_ln.weight",
    }.issubset(trainable)


def test_wote_no_controller_trains_original_wote_modules_without_controller_embedding(monkeypatch):
    monkeypatch.setenv("WOTE_TRAIN_PROFILE", "wote_no_controller")
    module = _module_with_fake_agent()

    module.setup()

    trainable = {name for name, param in module.agent.named_parameters() if param.requires_grad}
    assert trainable == {
        "WoTE_model.latent_world_model.weight",
        "WoTE_model.reward_conv_net.conv1.weight",
        "WoTE_model.reward_cat_head.0.weight",
        "WoTE_model.reward_head.0.weight",
        "WoTE_model.sim_reward_heads.0.0.weight",
        "WoTE_model._bev_upscale.weight",
        "WoTE_model.bev_upsample_head.up_conv5.weight",
        "WoTE_model.bev_semantic_head.0.weight",
        "WoTE_model.offset_tf_decoder.layers.0.self_attn.in_proj_weight",
        "WoTE_model.offset_head.weight",
        "WoTE_model.offset_score_head.0.weight",
    }
    assert not any("controller" in name or ".ctrl_" in name for name in trainable)
