from types import SimpleNamespace

import torch

from navsim.agents.WoTE import WoTE_loss


def test_zero_im_and_metric_weights_disable_reward_losses(monkeypatch):
    monkeypatch.setattr(
        WoTE_loss,
        "compute_traj_offset_loss",
        lambda predictions, targets, config: torch.tensor(3.0),
    )
    monkeypatch.setattr(
        WoTE_loss,
        "compute_im_reward_loss",
        lambda targets, rewards, anchors: torch.tensor(5.0),
    )
    monkeypatch.setattr(
        WoTE_loss,
        "compute_sim_reward_loss",
        lambda targets, rewards: torch.tensor(7.0),
    )

    config = SimpleNamespace(
        traj_offset_loss_weight=0.0,
        offset_im_reward_weight=0.0,
        im_loss_weight=0.0,
        metric_loss_weight=0.0,
        use_agent_loss=False,
        use_map_loss=False,
    )
    targets = {}
    predictions = {
        "trajectory_anchors": torch.zeros(1, 1, 3),
        "trajectory_offset_rewards": torch.zeros(1, 1),
        "im_rewards": torch.zeros(1, 1),
        "sim_rewards": torch.zeros(1, 1),
    }

    loss_dict = WoTE_loss.compute_wote_loss(targets, predictions, config)

    assert loss_dict["traj_offset_loss"].item() == 0.0
    assert loss_dict["offset_im_reward_loss"].item() == 0.0
    assert loss_dict["im_reward_loss"].item() == 0.0
    assert loss_dict["sim_reward_loss"].item() == 0.0
