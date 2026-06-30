import torch

from navsim.agents.WoTE.WoTE_agent import WarmupCosLR


def test_warmup_cos_lr_constructs_with_current_torch_scheduler_signature():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([param], lr=1e-4)

    scheduler = WarmupCosLR(
        optimizer=optimizer,
        min_lr=1e-6,
        lr=1e-4,
        warmup_epochs=1,
        epochs=2,
    )

    assert scheduler.get_last_lr() == [1e-4]
