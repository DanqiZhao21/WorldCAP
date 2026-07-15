# Stage 2 Controller World-Model Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace WoTE's legacy per-trajectory controller memory with the frozen Stage-1 support-set encoder and four trainable style tokens for Stage-2 world-model training.

**Architecture:** A focused Stage-2 module loads and permanently freezes the Stage-1 `ControllerEncoder`, while a trainable expander maps its one 64-dimensional system-identification embedding into four 256-dimensional tokens. WoTE computes these tokens from the active 41-point support bank, broadcasts them across planner candidates, and retains the existing cross-attention, FiLM, and latent-world-model transition path.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, Hydra/YAML, NumPy, pytest

---

### Task 1: Add The Frozen Encoder And Style-Token Expander

**Files:**
- Create: `navsim/agents/WoTE/controller/stage2.py`
- Create: `tests/test_wote_stage2_controller_encoder.py`

- [ ] **Step 1: Write failing checkpoint-loader and token-expander tests**

```python
def test_frozen_encoder_loads_stage1_checkpoint_and_stays_in_eval(tmp_path):
    checkpoint_path = write_tiny_stage1_encoder_checkpoint(tmp_path)
    encoder = FrozenControllerEncoder.from_checkpoint(checkpoint_path)
    encoder.train()
    assert not encoder.training
    assert not encoder.encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_style_token_expander_returns_fixed_token_count():
    expander = ControllerStyleTokenExpander(style_dim=64, token_count=4, token_dim=256)
    output = expander(torch.zeros(3, 64))
    assert output.shape == (3, 4, 256)
```

- [ ] **Step 2: Run the tests and confirm the new API is absent**

Run: `pytest -q tests/test_wote_stage2_controller_encoder.py`

Expected: FAIL because `controller.stage2` does not exist.

- [ ] **Step 3: Implement strict Stage-1 checkpoint loading**

Create `stage2.py` with `FrozenControllerEncoder`. Its
`from_checkpoint(path)` method must:

```python
required = {"controller_encoder_state", "model_config", "feature_mean", "feature_std"}
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
missing = required.difference(checkpoint)
if missing:
    raise ValueError(f"controller encoder checkpoint missing keys: {sorted(missing)}")
config = ModelConfig(**checkpoint["model_config"])
encoder = ControllerEncoder(config, checkpoint["feature_mean"], checkpoint["feature_std"])
encoder.load_state_dict(checkpoint["controller_encoder_state"], strict=True)
```

The wrapper must call `requires_grad_(False)`, keep itself and the inner encoder
in evaluation mode even when the parent WoTE model enters training mode, and
run support encoding under `torch.no_grad()`.

- [ ] **Step 4: Implement the fixed-count token expander**

```python
class ControllerStyleTokenExpander(nn.Module):
    def __init__(self, style_dim: int, token_count: int, token_dim: int = 256):
        super().__init__()
        if style_dim <= 0 or token_count <= 0 or token_dim <= 0:
            raise ValueError("style and token dimensions must be positive")
        self.token_count = token_count
        self.token_dim = token_dim
        self.projection = nn.Linear(style_dim, token_count * token_dim)
        self.normalization = nn.LayerNorm(token_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2:
            raise ValueError("controller embedding must be [B,D]")
        tokens = self.projection(embedding).reshape(
            embedding.shape[0], self.token_count, self.token_dim
        )
        return self.normalization(tokens)
```

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_wote_stage2_controller_encoder.py`

Expected: PASS.

### Task 2: Replace WoTE's Legacy Controller Memory

**Files:**
- Modify: `navsim/agents/WoTE/WoTE_model.py`
- Modify: `tests/test_wote_controller_world_model_fusion.py`

- [ ] **Step 1: Replace the old bank-size test with failing fixed-token tests**

Update the light model to expose a fake frozen support encoder and a real
`ControllerStyleTokenExpander`. Add these tests:

```python
@pytest.mark.parametrize("support_size", [4, 32, 128])
def test_controller_style_tokens_are_independent_of_support_size(support_size):
    model = light_stage2_model(token_count=4)
    model._active_ref_trajs = torch.zeros(support_size, 41, 3)
    model._active_exec_trajs = torch.ones(support_size, 41, 3)
    tokens = model._compute_controller_bank_tokens(2, 3, torch.device("cpu"))
    assert tokens.shape == (6, 4, 256)


def test_style_token_expander_receives_world_model_gradients():
    model = light_stage2_model(token_count=4)
    tokens = model._compute_controller_bank_tokens(1, 2, torch.device("cpu"))
    tokens.sum().backward()
    assert model.controller_style_token_expander.projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.pretrained_controller_encoder.parameters())
```

- [ ] **Step 2: Run the focused tests and observe the legacy token-count failure**

Run: `pytest -q tests/test_wote_controller_world_model_fusion.py`

Expected: FAIL because WoTE still returns one token per support trajectory.

- [ ] **Step 3: Remove the legacy encoder modules from `WoTEModel.__init__`**

Delete the `ControllerEmbedding` import and active modules:

```text
controller_encoder
ctrl_proj
ctrl_token_ln
ctrl_bank_proj
ctrl_bank_ln
```

Load `FrozenControllerEncoder` from the required
`controller_pretrain_checkpoint_path`, read its `style_emb_dim`, and instantiate
`ControllerStyleTokenExpander(style_dim, controller_style_token_count, 256)`.

- [ ] **Step 4: Replace `_compute_controller_bank_tokens`**

The replacement must validate matching `[S,T,3]` tensors and require
`T == pretrained_controller_encoder.config.num_poses`. It then executes:

```python
style_embedding = self.pretrained_controller_encoder(
    ref_traj.unsqueeze(0), exec_traj.unsqueeze(0)
)  # [1, 64]
style_tokens = self.controller_style_token_expander(style_embedding)  # [1, 4, 256]
style_tokens = style_tokens.expand(batch_size, -1, -1)
style_tokens = style_tokens[:, None].expand(batch_size, num_traj, -1, -1)
return style_tokens.reshape(batch_size * num_traj, self.controller_style_token_count, 256)
```

Missing active support banks must raise `RuntimeError`; they must not return
zero tokens.

- [ ] **Step 5: Preserve external encoder state after base checkpoint loading**

Add `WoTEModel.load_post_checkpoint_modules()` that strictly reloads the
configured Stage-1 encoder and reasserts its frozen/eval state. The existing
`WoTEAgent` and `AgentLightningModule` hooks will call it after restoring the
base WoTE checkpoint.

- [ ] **Step 6: Run fusion and Stage-2 tests**

Run:

```bash
pytest -q tests/test_wote_stage2_controller_encoder.py \
  tests/test_wote_controller_world_model_fusion.py
```

Expected: PASS.

### Task 3: Enforce Dense Support And Matched Candidate Styles

**Files:**
- Modify: `navsim/agents/WoTE/WoTE_model.py`
- Modify: `tests/test_wote_controller_world_model_fusion.py`
- Generate: `outputs/controller_stage2/controller_styles_structured_v1_256x8.npz`

- [ ] **Step 1: Add failing bundle-contract tests**

Extend the existing `_controller_bank_files` helper with `support_poses` and
`candidate_style_names` arguments. Add these separate contract assertions:

```python
def test_load_controller_banks_accepts_dense_support_and_matching_styles(tmp_path):
    paths = _controller_bank_files(
        tmp_path, planner, support_poses=41, candidate_style_names=["calm", "sport"]
    )
    model._load_controller_banks(_bank_config(paths))
    assert model._active_ref_trajs.shape == (4, 41, 3)


def test_load_controller_banks_rejects_mismatched_style_names(tmp_path):
    paths = _controller_bank_files(
        tmp_path, planner, support_poses=41, candidate_style_names=["sport", "calm"]
    )
    with pytest.raises(ValueError, match="style_names"):
        model._load_controller_banks(_bank_config(paths))


def test_load_controller_banks_rejects_sparse_support(tmp_path):
    paths = _controller_bank_files(tmp_path, planner, support_poses=8)
    with pytest.raises(ValueError, match="41"):
        model._load_controller_banks(_bank_config(paths))
```

- [ ] **Step 2: Run the bundle tests and confirm old permissive loading fails the contract**

Run: `pytest -q tests/test_wote_controller_world_model_fusion.py -k 'style or support'`

Expected: at least one FAIL because the old loader permits sparse support and
legacy style sets.

- [ ] **Step 3: Tighten `_load_controller_banks`**

Require support horizon equal to the loaded encoder's `num_poses`, require the
candidate and support style counts to match, and require both bundles to expose
identical `style_names` in identical order. Keep the current 256-anchor endpoint
alignment validation.

- [ ] **Step 4: Generate the matched structured candidate bundle**

Run:

```bash
PYTHONPATH=.:nuplan-devkit python \
  navsim/agents/WoTE/controller/scripts/generate_controller_bundle_clean.py \
  --anchors-path extra_data/planning_vb/trajectory_anchors_256.npy \
  --out-path outputs/controller_stage2/controller_styles_structured_v1_256x8.npz \
  --output-num-poses 8 --style-seed 42
```

Expected shapes:

```text
ref_traj:  [256, 8, 3]
exec_trajs: [105, 256, 8, 3]
style_names: [105]
```

- [ ] **Step 5: Verify style identity and candidate endpoint alignment**

Run a read-only Python check asserting:

```python
np.array_equal(support["style_names"], candidate["style_names"])
candidate["exec_trajs"].shape == (105, 256, 8, 3)
```

Then run: `pytest -q tests/test_wote_controller_world_model_fusion.py`

Expected: PASS.

### Task 4: Add Stage-2 Configuration And Freeze Policy

**Files:**
- Modify: `navsim/agents/WoTE/configs/default.py`
- Modify: `navsim/planning/training/agent_lightning_module.py`
- Modify: `tool/training/train_from_yaml.sh`
- Modify: `tests/test_agent_lightning_freeze_profiles.py`
- Modify: `tests/test_wote_config_fields.py`
- Modify: `tests/tool/test_worldcap_newctrl_paths.py`
- Create: `tool/training/Configymal/20260715_stage2_controller_world_model.yaml`

- [ ] **Step 1: Add failing config and freeze-profile tests**

Tests must require these config fields:

```python
controller_pretrain_checkpoint_path: str
controller_style_token_count: int = 4
```

Replace the legacy `controller_embedding` freeze-group expectation with:

```python
"controller_wm_fusion": [
    "WoTE_model.controller_style_token_expander",
    "WoTE_model.ctrl_fuse_attn",
    "WoTE_model.ctrl_wm_film_scale",
    "WoTE_model.ctrl_wm_film_shift",
    "WoTE_model.ctrl_wm_film_ln",
]
```

Assert the frozen encoder prefix is never in this group.

- [ ] **Step 2: Run config and freeze tests and confirm failure**

Run:

```bash
pytest -q tests/test_agent_lightning_freeze_profiles.py \
  tests/test_wote_config_fields.py tests/tool/test_worldcap_newctrl_paths.py
```

Expected: FAIL because the new fields/group/YAML are absent.

- [ ] **Step 3: Add config fields and Hydra forwarding**

Add the two fields to `WoTEConfig`. In `train_from_yaml.sh`, forward:

```text
++agent.config.controller_pretrain_checkpoint_path=<controller.pretrain_checkpoint_path>
++agent.config.controller_style_token_count=<controller.style_token_count>
```

Also validate the encoder checkpoint as a required path before launching.
Remove forwarding of the obsolete `controller_feature_mode` setting.

- [ ] **Step 4: Replace the freeze group**

Remove `controller_embedding` from `TRAINABLE_GROUP_PREFIXES` and add
`controller_wm_fusion` with only the expander, attention, and FiLM prefixes.
The Stage-1 encoder must remain frozen even if users select all documented
Stage-2 groups.

- [ ] **Step 5: Create the Stage-2 YAML**

The YAML must use:

```yaml
model:
  init_ckpt: ${run.root}/tool/epoch=29-step=19950.ckpt
  controller:
    style_exec_path: ${run.root}/CtrlNew/controller/pretrain/controller_styles_structured_v1_41.npz
    candidate_exec_path: ${run.root}/outputs/controller_stage2/controller_styles_structured_v1_256x8.npz
    pretrain_checkpoint_path: ${run.root}/outputs/controller_pretrain/controller_encoder_structured_v1_ddp2_encoder.pt
    style_token_count: 4
    style_split: train
    use_world_model: true
    fusion: attn_film

train:
  freeze:
    freeze_all: true
    trainable_groups: [controller_wm_fusion, latent_world_model]
  losses:
    agent: false
    current_bev: 0.0
    future_bev: 1.0
    traj_offset: 0.0
    offset_im_reward: 0.0
    imitation_reward: 0.0
    metric_reward: 0.0
```

- [ ] **Step 6: Run config and freeze tests**

Run the command from Step 2.

Expected: PASS.

### Task 5: Verify The Complete Stage-2 Path

- [ ] **Step 1: Run the focused regression suite**

```bash
pytest -q \
  tests/test_wote_stage2_controller_encoder.py \
  tests/test_wote_controller_world_model_fusion.py \
  tests/test_agent_lightning_freeze_profiles.py \
  tests/test_wote_config_fields.py \
  tests/test_controller_pretrain_data.py \
  tests/test_controller_pretrain_features.py \
  tests/test_controller_pretrain_model.py \
  tests/test_controller_pretrain_train.py
```

Expected: PASS.

- [ ] **Step 2: Dry-run the Stage-2 launcher**

```bash
WOTE_DRY_RUN=1 bash tool/training/train_from_yaml.sh \
  tool/training/Configymal/20260715_stage2_controller_world_model.yaml
```

Expected output includes the real Stage-1 encoder checkpoint, four style
tokens, `controller_wm_fusion`, `latent_world_model`, and no legacy
`controller_feature_mode` override.

- [ ] **Step 3: Run a real checkpoint-loading token smoke test**

Construct the Stage-2 support encoder from
`controller_encoder_structured_v1_ddp2_encoder.pt`, select one controller from
the dense support bundle, and verify:

```text
controller embedding: [1, 64]
style tokens: [1, 4, 256]
encoder requires_grad: false
expander requires_grad: true
```

- [ ] **Step 4: Check formatting and changed-file scope**

Run:

```bash
python -m compileall -q navsim/agents/WoTE/controller
git diff --check
git status --short
```

Expected: no syntax or whitespace errors; unrelated existing changes remain
untouched.
