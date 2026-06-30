import torch

from navsim.agents.WoTE.HERM import (
    FrenetErrorDynamicsHERM,
    HERMConfig,
    SupportConditionalHERM,
    SupportStyleEncoder,
)
from navsim.agents.WoTE.HERM.inference import execute_with_herm
from navsim.agents.WoTE.HERM.losses import herm_loss


def _straight_plan(batch: int = 2, horizon: int = 8) -> torch.Tensor:
    plan = torch.zeros((batch, horizon, 3), dtype=torch.float32)
    plan[:, :, 0] = torch.arange(horizon, dtype=torch.float32)
    return plan


def test_herm_forward_shape_without_controller_embedding():
    model = FrenetErrorDynamicsHERM(HERMConfig(num_poses=8, dt=0.5, hidden_dim=64))
    plan = _straight_plan(batch=4, horizon=8)

    out = model(plan, return_debug=True)

    assert out.exec_traj.shape == (4, 8, 3)
    assert out.residual.shape == (4, 8, 3)
    assert out.params.shape == (4, 7, 9)
    assert out.intrinsics is not None


def test_herm_zero_head_is_identity_execution():
    model = FrenetErrorDynamicsHERM(HERMConfig(num_poses=8, dt=0.5, hidden_dim=64))
    torch.nn.init.zeros_(model.param_head.weight)
    torch.nn.init.zeros_(model.param_head.bias)
    plan = _straight_plan(batch=2, horizon=8)

    out = model(plan)

    assert torch.allclose(out.exec_traj, plan, atol=1e-5)
    assert torch.allclose(out.residual, torch.zeros_like(out.residual), atol=1e-5)


def test_herm_requires_controller_embedding_when_configured():
    model = FrenetErrorDynamicsHERM(
        HERMConfig(num_poses=8, dt=0.5, hidden_dim=64, controller_emb_dim=16)
    )
    plan = _straight_plan(batch=2, horizon=8)

    try:
        model(plan)
    except ValueError as exc:
        assert "controller_emb" in str(exc)
    else:
        raise AssertionError("expected missing controller_emb to raise")


def test_herm_accepts_controller_embedding():
    model = FrenetErrorDynamicsHERM(
        HERMConfig(num_poses=8, dt=0.5, hidden_dim=64, controller_emb_dim=16)
    )
    plan = _straight_plan(batch=2, horizon=8)
    controller_emb = torch.zeros((2, 16), dtype=torch.float32)

    out = model(plan, controller_emb=controller_emb)

    assert out.exec_traj.shape == (2, 8, 3)
    assert out.params.shape == (2, 7, 9)


def test_herm_loss_zero_when_prediction_matches_target():
    pred = torch.zeros((2, 8, 3))
    target = torch.zeros((2, 8, 3))
    residual = torch.zeros((2, 8, 3))
    params = torch.zeros((2, 7, 9))

    loss, metrics = herm_loss(pred, target, residual, params)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    assert metrics["pos_l1"].item() == 0.0


def test_execute_with_herm_supports_candidate_dimension():
    model = FrenetErrorDynamicsHERM(HERMConfig(num_poses=8, dt=0.5, hidden_dim=64))
    torch.nn.init.zeros_(model.param_head.weight)
    torch.nn.init.zeros_(model.param_head.bias)
    planned = _straight_plan(batch=6, horizon=8).reshape(2, 3, 8, 3)

    executed = execute_with_herm(model, planned)

    assert executed.shape == (2, 3, 8, 3)
    assert torch.allclose(executed, planned, atol=1e-5)


def test_herm_can_learn_simple_lateral_bias():
    torch.manual_seed(0)
    model = FrenetErrorDynamicsHERM(HERMConfig(num_poses=8, dt=0.5, hidden_dim=64))
    plan = _straight_plan(batch=32, horizon=8)
    target = plan.clone()
    target[:, :, 1] = 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    with torch.no_grad():
        initial = torch.mean(torch.abs(model(plan).exec_traj[..., :2] - target[..., :2])).item()

    for _ in range(80):
        out = model(plan)
        loss = torch.mean(torch.abs(out.exec_traj[..., :2] - target[..., :2]))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = torch.mean(torch.abs(model(plan).exec_traj[..., :2] - target[..., :2])).item()

    assert final < initial * 0.5


def test_support_style_encoder_returns_controller_embedding():
    encoder = SupportStyleEncoder(HERMConfig(num_poses=8, dt=0.5), emb_dim=16, hidden_dim=32)
    support_plan = _straight_plan(batch=6, horizon=8).reshape(2, 3, 8, 3)
    support_exec = support_plan.clone()
    support_exec[..., 1] += 0.5

    emb = encoder(support_plan, support_exec)

    assert emb.shape == (2, 16)


def test_support_style_encoder_uses_conv1d_and_attention_pooling():
    encoder = SupportStyleEncoder(HERMConfig(num_poses=8, dt=0.5), emb_dim=16, hidden_dim=32)

    assert any(isinstance(module, torch.nn.Conv1d) for module in encoder.modules())

    pair_emb = torch.randn(2, 5, 16)
    style = encoder.support_pool(pair_emb)
    weights = torch.softmax(encoder.support_pool.score(pair_emb), dim=1)

    assert style.shape == (2, 16)
    assert weights.shape == (2, 5, 1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1), atol=1e-6)


def test_support_conditional_herm_predicts_query_shape():
    model = SupportConditionalHERM(
        HERMConfig(num_poses=8, dt=0.5, hidden_dim=64, controller_emb_dim=16),
        style_emb_dim=16,
        style_hidden_dim=32,
    )
    support_plan = _straight_plan(batch=6, horizon=8).reshape(2, 3, 8, 3)
    support_exec = support_plan.clone()
    query_plan = _straight_plan(batch=8, horizon=8).reshape(2, 4, 8, 3)

    out = model(support_plan, support_exec, query_plan)

    assert out.exec_traj.shape == (2, 4, 8, 3)
    assert out.residual.shape == (2, 4, 8, 3)
    assert out.params.shape == (2, 4, 7, 9)
