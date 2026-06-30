import torch

from navsim.agents.WoTE.HERM.inference_support import execute_with_support_herm
from navsim.agents.WoTE.HERM.model import HERMConfig, SupportConditionalHERM


def test_execute_with_support_herm_preserves_query_shape():
    model = SupportConditionalHERM(
        HERMConfig(num_poses=8, dt=0.5, hidden_dim=16, num_layers=1, controller_emb_dim=8),
        style_emb_dim=8,
        style_hidden_dim=16,
    )
    support_plan = torch.zeros((2, 4, 8, 3), dtype=torch.float32)
    support_exec = support_plan.clone()
    query_plan = torch.zeros((2, 3, 8, 3), dtype=torch.float32)

    pred = execute_with_support_herm(model, support_plan, support_exec, query_plan)

    assert pred.shape == (2, 3, 8, 3)
