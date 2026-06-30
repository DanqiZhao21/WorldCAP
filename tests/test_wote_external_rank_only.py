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


def test_external_rank_only_uses_external_candidates_without_offset_for_reward_features():
    model = object.__new__(WoTEModel)
    torch.nn.Module.__init__(model)
    model.is_eval = True
    model.use_agent_loss = False
    model.use_map_loss = False
    model.trajectory_anchors = torch.nn.Parameter(torch.zeros((2, 1, 3)), requires_grad=False)
    model._external_trajectory_anchors = torch.tensor(
        [[[[1.0, 2.0, 0.0]], [[3.0, 4.0, 0.0]]]],
        dtype=torch.float32,
    )
    model._external_rank_only = True
    model._process_backbone_features = lambda camera, lidar: (
        torch.empty(0),
        torch.zeros((1, 1, 4), dtype=torch.float32),
    )
    model._get_ego_status_feature = lambda status: torch.zeros((1, 1, 4), dtype=torch.float32)
    model._predict_offset = lambda ego_feat, bev: {
        "trajectory_offset": torch.full((1, 2, 1, 3), 10.0, dtype=torch.float32),
        "trajectory_offset_rewards": torch.zeros((1, 2), dtype=torch.float32),
    }
    seen_anchors = []

    def encode_traj_into_ego_feat(ego_status_feat, anchors, batch_size):
        seen_anchors.append(anchors.detach().clone())
        return torch.zeros((batch_size, anchors.shape[1], 1, 4), dtype=torch.float32), anchors.shape[1]

    model.encode_traj_into_ego_feat = encode_traj_into_ego_feat
    features = {
        "camera_feature": torch.empty(1),
        "lidar_feature": torch.empty(1),
        "status_feature": torch.zeros((1, 8), dtype=torch.float32),
    }

    outputs = model.extract_trajectory_feature(features)

    assert len(seen_anchors) == 2
    torch.testing.assert_close(seen_anchors[0], model._external_trajectory_anchors)
    torch.testing.assert_close(seen_anchors[1], model._external_trajectory_anchors)
    torch.testing.assert_close(outputs["candidate_anchors"], model._external_trajectory_anchors)
    torch.testing.assert_close(outputs["scored_trajectory_anchors"], model._external_trajectory_anchors)


def test_external_rank_only_forward_test_selects_original_candidate_not_offset_candidate():
    model = object.__new__(WoTEModel)
    torch.nn.Module.__init__(model)
    model._external_rank_only = True
    model._external_trajectory_anchors = torch.tensor(
        [[[[1.0, 0.0, 0.0]], [[5.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    model.trajectory_anchors = torch.nn.Parameter(torch.zeros((2, 1, 3)), requires_grad=False)
    model.process_trajectory_and_reward = lambda features: {
        "reward_feature": torch.zeros((1, 2, 4), dtype=torch.float32),
        "trajectory_offset": torch.full((1, 2, 1, 3), 100.0, dtype=torch.float32),
        "scored_trajectory_anchors": model._external_trajectory_anchors,
    }
    model.reward_head = torch.nn.Linear(4, 1, bias=False)
    model.reward_head.weight.data.zero_()
    model.sim_reward_heads = torch.nn.ModuleList([])
    model.weighted_reward_calculation = lambda im_rewards, sim_rewards: torch.tensor([[0.1, 0.9]])
    model.select_best_trajectory = WoTEModel.select_best_trajectory.__get__(model, WoTEModel)
    features = {"status_feature": torch.zeros((1, 8), dtype=torch.float32)}

    output = model.forward_test(features)

    torch.testing.assert_close(output["trajectory"], model._external_trajectory_anchors[:, 1])
    torch.testing.assert_close(output["all_trajectory"], model._external_trajectory_anchors.squeeze(0))
    torch.testing.assert_close(output["trajectoryAnchor"], model._external_trajectory_anchors.squeeze(0))
