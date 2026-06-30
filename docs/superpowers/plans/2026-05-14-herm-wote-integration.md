# HERM WoTE Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the trained support-conditioned HERM model into WoTE so WoTE can score predicted controller-executed candidate trajectories instead of only raw planned candidates.

**Architecture:** WoTE already loads an active controller bank as `(ref_trajs, exec_trajs)` for the current controller style. HERM should load a `SupportConditionalHERM` checkpoint, use the active controller bank as support pairs, transform WoTE candidate plan trajectories into predicted exec trajectories, and feed those predicted exec candidates into the reward/world-model path when enabled by config.

**Tech Stack:** PyTorch, NAVSIM WoTE model code, `navsim.agents.WoTE.HERM.SupportConditionalHERM`, pytest.

---

## File Structure

- Modify: `navsim/agents/WoTE/configs/default.py`
  - Add HERM config flags and checkpoint path.
- Modify: `navsim/agents/WoTE/WoTE_model.py`
  - Load HERM checkpoint.
  - Add a helper that executes candidate plan trajectories with active support pairs.
  - Use HERM predicted exec candidates in the eval scoring path.
  - Keep training behavior disabled by default unless explicitly enabled.
- Modify: `navsim/agents/WoTE/HERM/inference_support.py`
  - Add a batch-friendly helper if the existing helper is too narrow for WoTE candidate tensors.
- Create: `tests/test_wote_herm_integration.py`
  - Unit tests for HERM loading, support selection, shape preservation, and disabled fallback behavior.
- Optional modify: `docs/agents.md`
  - Add a short note explaining how to enable HERM in WoTE eval.

## Design Contract

Input candidate plans from WoTE:

```text
candidate_plans: [B, K, T, 3]
```

Active controller support bank already available in WoTE:

```text
_active_ref_trajs:  [N_ctrl, T, 3]
_active_exec_trajs: [N_ctrl, T, 3]
```

HERM support input:

```text
support_plan: [B, S, T, 3]
support_exec: [B, S, T, 3]
query_plan:   [B, K, T, 3]
```

HERM output used by WoTE:

```text
pred_exec: [B, K, T, 3]
```

Recommended default:

```text
herm_enable = False
herm_apply_in_eval = True
herm_apply_in_train = False
herm_support_size = 768
```

The default keeps existing WoTE behavior unchanged until HERM is explicitly enabled.

### Task 1: Add HERM Config

**Files:**
- Modify: `navsim/agents/WoTE/configs/default.py`

- [ ] **Step 1: Add config fields**

Add these fields near existing controller config fields:

```python
    # HERM execution predictor.
    # When enabled, WoTE candidate plan trajectories are converted into
    # predicted controller-executed trajectories before reward/world-model scoring.
    herm_enable: bool = False
    herm_apply_in_eval: bool = True
    herm_apply_in_train: bool = False
    herm_checkpoint_path: str = ""
    herm_support_size: int = 768
    herm_support_seed: int = 0
    herm_query_chunk_size: int = 256
    herm_device: str = "auto"
```

- [ ] **Step 2: Verify config imports**

Run:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE python - <<'PY'
from navsim.agents.WoTE.configs.default import WoTEConfig
c = WoTEConfig()
print(c.herm_enable, c.herm_support_size, c.herm_query_chunk_size)
PY
```

Expected:

```text
False 768 256
```

- [ ] **Step 3: Commit**

```bash
git add navsim/agents/WoTE/configs/default.py
git commit -m "config: add HERM WoTE integration flags"
```

### Task 2: Add HERM Checkpoint Loader

**Files:**
- Modify: `navsim/agents/WoTE/WoTE_model.py`
- Test: `tests/test_wote_herm_integration.py`

- [ ] **Step 1: Write failing checkpoint-loader test**

Create `tests/test_wote_herm_integration.py` with a small helper test that validates checkpoint config extraction without constructing a full NAVSIM model:

```python
import torch

from navsim.agents.WoTE.HERM.model import HERMConfig, SupportConditionalHERM


def test_support_conditional_herm_checkpoint_round_trip(tmp_path):
    config = HERMConfig(
        num_poses=8,
        hidden_dim=32,
        num_layers=1,
        controller_emb_dim=16,
        dropout=0.0,
    )
    model = SupportConditionalHERM(config, style_hidden_dim=24)
    ckpt_path = tmp_path / "herm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "style_hidden_dim": 24,
        },
        ckpt_path,
    )

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    loaded_config = HERMConfig(**loaded["config"])
    loaded_model = SupportConditionalHERM(
        loaded_config,
        style_hidden_dim=int(loaded.get("style_hidden_dim", 128)),
    )
    loaded_model.load_state_dict(loaded["model_state_dict"])

    assert loaded_config.controller_emb_dim == 16
    assert loaded_model.training is True
```

- [ ] **Step 2: Run test to verify baseline**

Run:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE pytest tests/test_wote_herm_integration.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 3: Add loader imports**

In `navsim/agents/WoTE/WoTE_model.py`, add imports near existing imports:

```python
from navsim.agents.WoTE.HERM.model import HERMConfig, SupportConditionalHERM
```

- [ ] **Step 4: Add `_load_herm_model` method**

Add this method inside the WoTE model class:

```python
    def _load_herm_model(self, config) -> None:
        self.herm_enable = bool(getattr(config, "herm_enable", False))
        self.herm_apply_in_eval = bool(getattr(config, "herm_apply_in_eval", True))
        self.herm_apply_in_train = bool(getattr(config, "herm_apply_in_train", False))
        self.herm_support_size = int(getattr(config, "herm_support_size", 768) or 768)
        self.herm_support_seed = int(getattr(config, "herm_support_seed", 0) or 0)
        self.herm_query_chunk_size = int(getattr(config, "herm_query_chunk_size", 256) or 256)
        self.herm_model = None

        ckpt_path = str(getattr(config, "herm_checkpoint_path", "") or "")
        if not self.herm_enable:
            return
        if ckpt_path == "":
            raise ValueError("herm_enable=True requires herm_checkpoint_path")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"HERM checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        herm_config = HERMConfig(**ckpt["config"])
        style_hidden_dim = int(ckpt.get("style_hidden_dim", 128))
        herm_model = SupportConditionalHERM(herm_config, style_hidden_dim=style_hidden_dim)
        herm_model.load_state_dict(ckpt["model_state_dict"])
        herm_model.eval()
        for p in herm_model.parameters():
            p.requires_grad_(False)
        self.herm_model = herm_model
```

- [ ] **Step 5: Call loader from `__init__`**

After the controller bank loading block in `__init__`, call:

```python
        self._load_herm_model(config)
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE pytest tests/test_wote_herm_integration.py tests/test_herm_model.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 7: Commit**

```bash
git add navsim/agents/WoTE/WoTE_model.py tests/test_wote_herm_integration.py
git commit -m "feat: load HERM checkpoint in WoTE"
```

### Task 3: Add Candidate Execution Helper

**Files:**
- Modify: `navsim/agents/WoTE/WoTE_model.py`
- Test: `tests/test_wote_herm_integration.py`

- [ ] **Step 1: Add a shape-preserving fake-HERM test**

Append this test:

```python
import types


class FakeHERM(torch.nn.Module):
    def forward(self, support_plan, support_exec, query_plan):
        offset = support_exec[:, :1, :, :2].mean(dim=1, keepdim=True) - support_plan[:, :1, :, :2].mean(dim=1, keepdim=True)
        pred = query_plan.clone()
        pred[..., :2] = pred[..., :2] + offset
        residual = torch.zeros_like(query_plan)
        params = torch.zeros(query_plan.shape[0], query_plan.shape[1], query_plan.shape[2], 9, device=query_plan.device)
        return pred, residual, params


def test_execute_candidates_with_herm_preserves_shape_and_uses_support():
    owner = types.SimpleNamespace()
    owner.herm_enable = True
    owner.herm_apply_in_eval = True
    owner.herm_apply_in_train = False
    owner.herm_support_size = 2
    owner.herm_support_seed = 0
    owner.herm_query_chunk_size = 2
    owner.herm_model = FakeHERM()
    owner.training = False
    owner.is_eval = True
    owner._active_ref_trajs = torch.zeros(4, 8, 3)
    owner._active_exec_trajs = torch.zeros(4, 8, 3)
    owner._active_exec_trajs[..., 0] = 1.5

    from navsim.agents.WoTE.WoTE_model import WoTEModel

    owner._select_herm_support_indices = types.MethodType(WoTEModel._select_herm_support_indices, owner)
    owner._execute_candidates_with_herm = types.MethodType(WoTEModel._execute_candidates_with_herm, owner)

    query = torch.zeros(1, 3, 8, 3)
    out = owner._execute_candidates_with_herm(query)

    assert out.shape == query.shape
    assert torch.allclose(out[..., 0], torch.full_like(out[..., 0], 1.5))
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE pytest tests/test_wote_herm_integration.py::test_execute_candidates_with_herm_preserves_shape_and_uses_support -q
```

Expected failure:

```text
AttributeError: type object 'WoTEModel' has no attribute '_execute_candidates_with_herm'
```

- [ ] **Step 3: Add support index selection helper**

Add this method inside the WoTE model class:

```python
    def _select_herm_support_indices(self, num_support_bank: int, device: torch.device) -> torch.Tensor:
        support_size = min(int(self.herm_support_size), int(num_support_bank))
        if support_size <= 0:
            raise ValueError("herm_support_size must be positive when HERM is enabled")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.herm_support_seed))
        perm = torch.randperm(int(num_support_bank), generator=generator)
        return perm[:support_size].to(device=device)
```

- [ ] **Step 4: Add candidate execution helper**

Add this method inside the WoTE model class:

```python
    def _execute_candidates_with_herm(self, candidate_plans: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self, "herm_enable", False)):
            return candidate_plans
        if getattr(self, "herm_model", None) is None:
            return candidate_plans
        if getattr(self, "is_eval", False):
            if not bool(getattr(self, "herm_apply_in_eval", True)):
                return candidate_plans
        else:
            if not bool(getattr(self, "herm_apply_in_train", False)):
                return candidate_plans

        if not hasattr(self, "_active_ref_trajs") or not hasattr(self, "_active_exec_trajs"):
            return candidate_plans

        ref = self._active_ref_trajs.to(device=candidate_plans.device, dtype=candidate_plans.dtype)
        exe = self._active_exec_trajs.to(device=candidate_plans.device, dtype=candidate_plans.dtype)
        if ref.ndim != 3 or exe.ndim != 3:
            raise ValueError(f"HERM support bank must be [N,T,3], got ref={tuple(ref.shape)} exec={tuple(exe.shape)}")
        if ref.shape != exe.shape:
            raise ValueError(f"HERM support ref/exec shapes must match, got ref={tuple(ref.shape)} exec={tuple(exe.shape)}")
        if ref.shape[1] != candidate_plans.shape[2]:
            raise ValueError(f"HERM support T={ref.shape[1]} does not match candidate T={candidate_plans.shape[2]}")

        idx = self._select_herm_support_indices(ref.shape[0], candidate_plans.device)
        support_plan = ref.index_select(0, idx).unsqueeze(0).expand(candidate_plans.shape[0], -1, -1, -1)
        support_exec = exe.index_select(0, idx).unsqueeze(0).expand(candidate_plans.shape[0], -1, -1, -1)

        herm = self.herm_model.to(candidate_plans.device)
        was_training = herm.training
        herm.eval()
        chunks = []
        chunk_size = max(1, int(getattr(self, "herm_query_chunk_size", 256) or 256))
        with torch.no_grad():
            for start in range(0, candidate_plans.shape[1], chunk_size):
                end = min(candidate_plans.shape[1], start + chunk_size)
                pred_exec, _, _ = herm(support_plan, support_exec, candidate_plans[:, start:end])
                chunks.append(pred_exec)
        if was_training:
            herm.train()
        return torch.cat(chunks, dim=1)
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE pytest tests/test_wote_herm_integration.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

```bash
git add navsim/agents/WoTE/WoTE_model.py tests/test_wote_herm_integration.py
git commit -m "feat: execute WoTE candidates with HERM"
```

### Task 4: Wire HERM Into WoTE Candidate Scoring

**Files:**
- Modify: `navsim/agents/WoTE/WoTE_model.py`
- Test: `tests/test_wote_herm_integration.py`

- [ ] **Step 1: Add behavior test for disabled fallback**

Append this test:

```python
def test_execute_candidates_with_herm_disabled_returns_input_object():
    owner = types.SimpleNamespace()
    owner.herm_enable = False
    owner.herm_model = None

    from navsim.agents.WoTE.WoTE_model import WoTEModel

    owner._execute_candidates_with_herm = types.MethodType(WoTEModel._execute_candidates_with_herm, owner)
    query = torch.zeros(1, 2, 8, 3)
    out = owner._execute_candidates_with_herm(query)

    assert out is query
```

- [ ] **Step 2: Run test**

Run:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE pytest tests/test_wote_herm_integration.py::test_execute_candidates_with_herm_disabled_returns_input_object -q
```

Expected:

```text
1 passed
```

- [ ] **Step 3: Modify eval candidate path**

In `extract_trajectory_feature`, replace the eval branch after `offseted_trajectory_anchors` is computed:

```python
            scored_trajectory_anchors = self._execute_candidates_with_herm(offseted_trajectory_anchors)

            ego_feat_for_reward_network, _ = self.encode_traj_into_ego_feat(
                ego_status_feat,
                scored_trajectory_anchors,
                batch_size,
            )
```

Keep `offseted_trajectory_anchors` as the planned candidate output and use `scored_trajectory_anchors` as the HERM-executed candidate for scoring.

- [ ] **Step 4: Modify training path only behind flag**

In the training branch of `extract_trajectory_feature`, keep existing default behavior:

```python
        else:
            if bool(getattr(self, "herm_enable", False)) and bool(getattr(self, "herm_apply_in_train", False)):
                train_candidates = self._execute_candidates_with_herm(init_trajectory_anchor)
                scored_trajectory_anchors = train_candidates
                ego_feat_for_reward_network, _ = self.encode_traj_into_ego_feat(
                    ego_status_feat,
                    scored_trajectory_anchors,
                    batch_size,
                )
            else:
                ego_feat_for_reward_network = ego_feat_fixed_anchor_WoTE
```

- [ ] **Step 5: Preserve outputs for debugging**

In the returned `trajectory_outputs`, include both raw and scored candidates:

```python
            "candidate_anchors": init_trajectory_anchor,
            "scored_trajectory_anchors": scored_trajectory_anchors,
```

In `forward_test`, use `encoder_results.get("scored_trajectory_anchors", trajectory_anchors)` for `all_trajectory_exec_hat`:

```python
        scored = encoder_results.get("scored_trajectory_anchors", trajectory_anchors)
        results = {
            "trajectory": poses,
            "final_rewards": final_rewards,
            "trajectoryAnchor": trajectory_anchors_ori.squeeze(0) if batch_size == 1 else trajectory_anchors_ori,
            "all_trajectory": trajectory_anchors.squeeze(0) if batch_size == 1 else trajectory_anchors,
            "all_trajectory_exec_hat": scored.squeeze(0) if batch_size == 1 else scored,
            "im_rewards": im_rewards_softmax,
        }
```

- [ ] **Step 6: Run HERM and WoTE tests**

Run:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE pytest \
  tests/test_wote_herm_integration.py \
  tests/test_herm_geometry.py \
  tests/test_herm_data.py \
  tests/test_herm_model.py \
  tests/test_herm_train_support.py \
  tests/test_herm_inference_support.py \
  -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 7: Commit**

```bash
git add navsim/agents/WoTE/WoTE_model.py tests/test_wote_herm_integration.py
git commit -m "feat: score WoTE candidates with HERM execution"
```

### Task 5: Add Smoke Command and Documentation

**Files:**
- Optional modify: `docs/agents.md`

- [ ] **Step 1: Add a WoTE HERM eval example**

Add this section to `docs/agents.md`:

```markdown
## WoTE HERM Execution Predictor

HERM can be enabled in WoTE by setting:

```python
config.herm_enable = True
config.herm_checkpoint_path = "/home/zhaodanqi/clone/WoTE/trainingResult/HERM/herm_support_1024_768_256_convattn_wandb_20260514.pt"
config.herm_support_size = 768
config.herm_query_chunk_size = 256
config.herm_apply_in_eval = True
config.herm_apply_in_train = False
```

When enabled in eval, WoTE still generates candidate plan trajectories normally. Before scoring, HERM uses the active controller bank `(ref, exec)` pairs as support and predicts controller-executed versions of the candidate plans.
```

- [ ] **Step 2: Run a pure import smoke check**

Run:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE python - <<'PY'
from navsim.agents.WoTE.configs.default import WoTEConfig
from navsim.agents.WoTE.HERM.model import SupportConditionalHERM
c = WoTEConfig()
c.herm_enable = True
c.herm_checkpoint_path = "/home/zhaodanqi/clone/WoTE/trainingResult/HERM/herm_support_1024_768_256_convattn_wandb_20260514.pt"
print(c.herm_enable)
print(SupportConditionalHERM.__name__)
PY
```

Expected:

```text
True
SupportConditionalHERM
```

- [ ] **Step 3: Run full targeted test suite**

Run:

```bash
PYTHONPATH=/home/zhaodanqi/clone/WoTE pytest \
  tests/test_wote_herm_integration.py \
  tests/test_herm_geometry.py \
  tests/test_herm_data.py \
  tests/test_herm_model.py \
  tests/test_herm_train.py \
  tests/test_herm_train_support.py \
  tests/test_herm_inference_support.py \
  -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 4: Commit**

```bash
git add docs/agents.md
git commit -m "docs: document HERM WoTE integration"
```

## Validation After Implementation

After all tasks, run a WoTE eval job twice:

1. HERM disabled:

```text
herm_enable = False
```

2. HERM enabled:

```text
herm_enable = True
herm_checkpoint_path = trainingResult/HERM/herm_support_1024_768_256_convattn_wandb_20260514.pt
```

Compare:

- NAVSIM score.
- Selected trajectory visualization.
- `all_trajectory` vs `all_trajectory_exec_hat`.
- Runtime and GPU memory.
- Whether ranking changes in scenes with aggressive curvature or controller lag.

The expected first success criterion is not necessarily a higher NAVSIM score immediately. The first success criterion is that WoTE can rank candidate trajectories using predicted executed trajectories without shape errors, device errors, or controller-style mismatch.

## Risks

- Distribution shift: HERM trained on bank trajectories, but WoTE candidate trajectories are produced by model offsets.
- Runtime cost: support size 768 and query size 256 can be expensive inside eval.
- Train/eval mismatch: enabling HERM only in eval changes the scoring distribution compared with WoTE training.
- Controller style mismatch: HERM must use the same active style support bank that the simulator/controller uses for execution.

## Recommended Rollout

1. Integrate HERM disabled by default.
2. Enable HERM only in eval for one fixed controller style using `WOTE_CTRL_STYLE_IDX`.
3. Compare raw planned candidates against `all_trajectory_exec_hat`.
4. Enable style sampling after fixed-style behavior is verified.
5. Consider HERM-in-training only after eval behavior is stable.
