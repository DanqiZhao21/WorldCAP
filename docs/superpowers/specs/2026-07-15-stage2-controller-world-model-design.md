# Stage 2 Controller World-Model Integration Design

## Goal

Replace WoTE's legacy per-trajectory controller memory with the frozen Stage-1
support-set ControllerEncoder, expand one identified controller embedding into
four style tokens, and train the existing cross-attention, FiLM, and latent
world-model modules to use those tokens.

Stage 2 must preserve the system-identification semantics learned in Stage 1.
World-model gradients must not update the ControllerEncoder.

## Scope

This change deliberately removes the legacy controller-token path. Backward
compatibility with checkpoints that depend on the old `ControllerEmbedding` or
1,024 per-trajectory tokens is out of scope.

The Stage-1 response decoder is not loaded into WoTE. Only the trained
ControllerEncoder, its model configuration, and feature-normalization buffers
are required.

## Inputs And Checkpoint

The Stage-1 encoder checkpoint is:

`outputs/controller_pretrain/controller_encoder_structured_v1_ddp2_encoder.pt`

It provides:

- `controller_encoder_state`
- `model_config`
- `feature_mean`
- `feature_std`

WoTE configuration will require a non-empty
`controller_pretrain_checkpoint_path` whenever controller-conditioned world
model execution is enabled. Loading must fail immediately if the file is
missing, required keys are absent, or the checkpoint style dimension does not
match the configured style-token expander.

The active controller support bank must contain matching reference and
execution tensors with shape `[S, 41, 3]`. The ControllerEncoder receives them
as `[1, S, 41, 3]` and returns one style embedding with shape `[1, 64]`.

## Model Architecture

WoTE will instantiate the Stage-1 `ControllerEncoder` directly from the
checkpoint's `ModelConfig`, feature mean, and feature standard deviation. It
will load `controller_encoder_state` strictly, switch the encoder to evaluation
mode, and set every encoder parameter to `requires_grad=False`.

The old `ControllerEmbedding`, `ctrl_proj`, `ctrl_token_ln`, `ctrl_bank_proj`,
and `ctrl_bank_ln` modules will be removed from the active model.

A new `ControllerStyleTokenExpander` will map one 64-dimensional controller
embedding into four 256-dimensional tokens:

```text
z_c [B, 64]
  -> Linear(64, 1024)
  -> reshape [B, 4, 256]
  -> LayerNorm(256)
```

The expander is trainable in Stage 2. Four tokens are fixed by the default
configuration through `controller_style_token_count=4`.

The existing `ctrl_fuse_attn`, `ctrl_wm_film_scale`,
`ctrl_wm_film_shift`, `ctrl_wm_film_ln`, and `latent_world_model` remain the
conditioning and transition path. The four style tokens are expanded across
planner candidates to `[B * N, 4, 256]` before cross-attention. No tensor with
one token per support trajectory is exposed to the world model.

## Data Flow

The active support bank identifies one controller once per forward pass:

```text
active ref/exec support bank [S, 41, 3]
  -> frozen Stage-1 ControllerEncoder
  -> controller embedding [1, 64]
  -> trainable style-token expander [1, 4, 256]
  -> expand across batch and candidate plans [B * N, 4, 256]
  -> existing cross-attention and FiLM
  -> latent world model
```

The support bank remains separate from the 256 candidate execution trajectories.
Candidate execution alignment and world-model rollout behavior are unchanged.

Because the active bank is controller-specific but not scene-specific, the
implementation may compute one style-token set and broadcast it across the
batch. It must not detach the expander output, because Stage-2 gradients must
train the expander. The frozen encoder output may be computed under
`torch.no_grad()`.

## Training And Freezing

Stage 2 starts from the existing WoTE base checkpoint and separately loads the
Stage-1 encoder checkpoint after the WoTE checkpoint restore. This avoids
coupling the external encoder state to legacy WoTE checkpoint keys.

The trainable Stage-2 modules are:

- `controller_style_token_expander`
- `ctrl_fuse_attn`
- `ctrl_wm_film_scale`
- `ctrl_wm_film_shift`
- `ctrl_wm_film_ln`
- `latent_world_model`

The frozen modules include:

- `pretrained_controller_encoder`
- perception backbone and scene encoders
- planner and trajectory heads
- reward heads for the initial Stage-2 experiment

The freeze-profile registry will expose a dedicated
`controller_wm_fusion` group for the expander and fusion modules. The existing
`latent_world_model` group remains separate. The Stage-2 YAML will select both
groups while leaving the encoder frozen.

The initial Stage-2 objective is the existing future-BEV loss. Existing reward
or imitation losses will not be added merely to make frozen heads trainable.
Any later reward-head experiment is a separate configuration change.

## Configuration

WoTE configuration will add:

- `controller_pretrain_checkpoint_path: str`
- `controller_style_token_count: int = 4`

The YAML launcher will forward both fields through Hydra overrides. A new
Stage-2 YAML will point to the completed Stage-1 encoder checkpoint, use
`attn_film`, enable controller-conditioned world-model execution, and select
only `controller_wm_fusion` plus `latent_world_model` as trainable groups.

The YAML will use the existing multi-GPU Lightning DDP training path. It will
not launch training automatically as part of model-construction tests.

## Error Handling

Model construction must raise a clear error when:

- controller conditioning is enabled without an encoder checkpoint path;
- the checkpoint file does not exist;
- a required checkpoint key is missing;
- support reference and execution tensors have different shapes;
- support trajectories are not `[S, 41, 3]` according to checkpoint config;
- the style embedding dimension is invalid for the expander.

The code must not silently fall back to the legacy encoder or zero controller
tokens when controller conditioning is enabled. Missing active support banks
are configuration errors.

## Testing

Focused tests will verify:

1. A Stage-1 encoder checkpoint loads strictly into WoTE and remains frozen.
2. The support bank produces exactly one 64-dimensional controller embedding.
3. The expander produces exactly four 256-dimensional style tokens.
4. Candidate expansion yields `[B * N, 4, 256]`, independent of support size.
5. Gradients reach the expander, cross-attention, FiLM, and latent world model,
   but not the frozen ControllerEncoder.
6. Freeze profiles select the new fusion group without selecting the encoder.
7. Missing checkpoints and malformed support banks fail explicitly.
8. The Stage-2 YAML forwards the checkpoint path and token count correctly.

Existing controller pretraining, freeze-profile, loss-weight, and world-model
fusion tests must remain green.

## Success Criteria

The integration is complete when a Stage-2 smoke forward/backward pass loads
the real Stage-1 encoder checkpoint, constructs four style tokens from a
support bank, conditions the latent world model, updates only the intended
Stage-2 modules, and passes the focused regression suite. The old 1,024-token
controller-memory path must no longer exist in active WoTE code.
