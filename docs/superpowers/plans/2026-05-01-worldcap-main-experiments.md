# WorldCAP Main Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the main WorldCAP paper experiment using the new `newCtrl` data layout, clean controller bundles, and a controlled training/evaluation path that is consistent with the paper's Table 1 setup.

**Architecture:** Use the new `newCtrl` controller assets as the sole default source for controller reference trajectories and controller bundles. Keep the core WoTE/WorldCAP model code unchanged for now, and drive reproduction through a minimal set of training/evaluation wrapper scripts plus explicit environment settings. Validate each stage separately: inputs, training launch, checkpoint selection, and test-split evaluation.

**Tech Stack:** Bash, Python, Hydra-based navsim training/evaluation scripts, NumPy `.npz` controller bundles, pytest.

---

### Task 1: Freeze the main-experiment asset contract

**Files:**
- Modify: `/home/zhaodanqi/clone/WoTE/docs/superpowers/plans/2026-05-01-worldcap-main-experiments.md`
- Verify: `/home/zhaodanqi/clone/WoTE/newCtrl/controller/bundles/64/controller_styles_64.npz`
- Verify: `/home/zhaodanqi/clone/WoTE/newCtrl/controller/bundles/128/controller_styles_128.npz`
- Verify: `/home/zhaodanqi/clone/WoTE/newCtrl/controller/bundles/1024/controller_styles_1024.npz`

- [ ] **Step 1: Record the main experiment inputs**

Main experiment asset contract:

```text
Controller refs:
- newCtrl/controller/ref_trajs/Anchors_Original_64_centered.npy
- newCtrl/controller/ref_trajs/Anchors_Original_128_centered.npy
- newCtrl/controller/ref_trajs/Anchors_Original_1024_centered.npy

Controller bundles:
- newCtrl/controller/bundles/64/controller_styles_64.npz
- newCtrl/controller/bundles/128/controller_styles_128.npz
- newCtrl/controller/bundles/1024/controller_styles_1024.npz

Planner anchors:
- default main run: extra_data/planning_vb/trajectory_anchors_256.npy
- default reward dict: extra_data/planning_vb/formatted_pdm_score_256.npy
```

- [ ] **Step 2: Verify all files exist before any training**

Run:

```bash
ls -lh \
  newCtrl/controller/ref_trajs/Anchors_Original_64_centered.npy \
  newCtrl/controller/ref_trajs/Anchors_Original_128_centered.npy \
  newCtrl/controller/ref_trajs/Anchors_Original_1024_centered.npy \
  newCtrl/controller/bundles/64/controller_styles_64.npz \
  newCtrl/controller/bundles/128/controller_styles_128.npz \
  newCtrl/controller/bundles/1024/controller_styles_1024.npz \
  extra_data/planning_vb/trajectory_anchors_256.npy \
  extra_data/planning_vb/formatted_pdm_score_256.npy
```

Expected: all files present, no `No such file or directory`.

- [ ] **Step 3: Verify bundle metadata consistency**

Run:

```bash
python - <<'PY'
import numpy as np
for p in [
  'newCtrl/controller/bundles/64/controller_styles_64.npz',
  'newCtrl/controller/bundles/128/controller_styles_128.npz',
  'newCtrl/controller/bundles/1024/controller_styles_1024.npz',
]:
    d = np.load(p, allow_pickle=True)
    print(p)
    print(' exec_trajs', d['exec_trajs'].shape)
    print(' ref_traj', d['ref_traj'].shape)
    print(' styles', d['style_names'].shape[0])
    print(' train', d['train_style_indices'].shape[0], 'val', d['val_style_indices'].shape[0])
    print(' ref_payload_mode', d['ref_payload_mode'][0])
    print()
PY
```

Expected: `styles=213`, `train=149`, `val=64`, and `ref_payload_mode source_ref` for all three bundles.

- [ ] **Step 4: Commit the asset contract documentation**

```bash
git add docs/superpowers/plans/2026-05-01-worldcap-main-experiments.md
git commit -m "docs: record worldcap main experiment asset contract"
```

### Task 2: Standardize the training entrypoints for the main experiment

**Files:**
- Modify: `/home/zhaodanqi/clone/WoTE/tool/training/0302train_wote_ctrlbundle_split_train_attn.sh`
- Modify: `/home/zhaodanqi/clone/WoTE/tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl64.sh`
- Modify: `/home/zhaodanqi/clone/WoTE/tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl128.sh`
- Modify: `/home/zhaodanqi/clone/WoTE/tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl1024.sh`
- Test: `/home/zhaodanqi/clone/WoTE/tests/tool/test_worldcap_newctrl_paths.py`

- [ ] **Step 1: Decide the canonical main-experiment training path**

Use these wrappers as the canonical main-experiment training entrypoints:

```text
tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl64.sh
tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl128.sh
tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl1024.sh
```

Reason: they already encode the bundle-size training split experiment used by the paper-style reference-count study.

- [ ] **Step 2: Verify wrapper defaults resolve to `newCtrl`**

Run:

```bash
pytest -q tests/tool/test_worldcap_newctrl_paths.py
```

Expected: `3 passed`.

- [ ] **Step 3: Dry-run the Python preflight without starting long training**

Run:

```bash
ROOT=/home/zhaodanqi/clone/WoTE \
CUDA_VISIBLE_DEVICES=0 \
WOTE_MAX_EPOCHS=1 \
WOTE_BATCH_SIZE=1 \
RUN_FUSIONS=attn \
OPENSCENE_DATA_ROOT=/path/to/dataset \
bash tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl64.sh
```

Expected: the script prints preflight information showing `ctrl_ref` and `ctrl_exec` under `newCtrl`. Stop after confirming the launch path is correct if dataset/GPU time is not yet allocated.

- [ ] **Step 4: Commit the training-entrypoint stabilization**

```bash
git add \
  tool/common/worldcap_newctrl_paths.sh \
  tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl64.sh \
  tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl128.sh \
  tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl1024.sh \
  tests/tool/test_worldcap_newctrl_paths.py
git commit -m "chore: point main worldcap training wrappers to newCtrl assets"
```

### Task 3: Define the main paper run matrix

**Files:**
- Modify: `/home/zhaodanqi/clone/WoTE/docs/superpowers/plans/2026-05-01-worldcap-main-experiments.md`

- [ ] **Step 1: Declare the primary run to target first**

Start with the strongest paper-aligned run first:

```text
Bundle size: 256-equivalent main planner anchors + 1024 controller bundle
Fusion: attn
Controller ref: newCtrl/controller/ref_trajs/Anchors_Original_1024_centered.npy
Controller bundle: newCtrl/controller/bundles/1024/controller_styles_1024.npz
Planner anchors: extra_data/planning_vb/trajectory_anchors_256.npy
Reward dict: extra_data/planning_vb/formatted_pdm_score_256.npy
```

- [ ] **Step 2: Define the full Table-1 reproduction run set**

Run set:

```text
1. Main WorldCAP report run:
   tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl1024.sh

2. Supporting reference-count runs for later ablations:
   tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl64.sh
   tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl128.sh
```

- [ ] **Step 3: Define checkpoint selection criteria**

Checkpoint selection rule:

```text
- Use the validation split already encoded in the bundle.
- Prefer the checkpoint with best held-out controller-style evaluation PDMS.
- If multiple are close, keep the one with the strongest TTC/EP balance.
- Record the exact checkpoint path before any test-split evaluation.
```

- [ ] **Step 4: Commit the run matrix documentation**

```bash
git add docs/superpowers/plans/2026-05-01-worldcap-main-experiments.md
git commit -m "docs: define worldcap main experiment run matrix"
```

### Task 4: Standardize the evaluation entrypoint for the main experiment

**Files:**
- Modify: `/home/zhaodanqi/clone/WoTE/tool/evaluate/eval_ckpts_20260225_valstyles_v3_20260303_ref64_128_1024.sh`
- Modify: `/home/zhaodanqi/clone/WoTE/tool/evaluate/evaluate_ckpt_20260207_000522_wm.sh`
- Test: `/home/zhaodanqi/clone/WoTE/tests/tool/test_worldcap_newctrl_paths.py`

- [ ] **Step 1: Choose the canonical evaluation script**

Use this as the canonical reference-count evaluation script:

```text
tool/evaluate/eval_ckpts_20260225_valstyles_v3_20260303_ref64_128_1024.sh
```

Reason: it already handles fixed-style matched evaluation across `64/128/1024` controller bundle sizes.

- [ ] **Step 2: Verify the evaluation script points to `newCtrl` by default**

Run:

```bash
pytest -q tests/tool/test_worldcap_newctrl_paths.py
```

Expected: `3 passed`.

- [ ] **Step 3: Dry-run evaluation preflight with explicit environment**

Run:

```bash
OPENSCENE_DATA_ROOT=/path/to/dataset \
SPLIT=test \
SCENE_FILTER=navtest \
GPU=0 \
EVAL_WORKER=ray_distributed_no_torch \
EVAL_CPU_THREADS=1 \
RAY_STOP_FIRST=1 \
bash tool/evaluate/eval_ckpts_20260225_valstyles_v3_20260303_ref64_128_1024.sh
```

Expected: preflight reports `CTRL_EXEC_64/128/1024` under `newCtrl/controller/bundles/...` and does not fail on missing path mismatches.

- [ ] **Step 4: Commit the evaluation-entrypoint stabilization**

```bash
git add tool/evaluate/eval_ckpts_20260225_valstyles_v3_20260303_ref64_128_1024.sh tests/tool/test_worldcap_newctrl_paths.py
git commit -m "chore: point main worldcap evaluation script to newCtrl assets"
```

### Task 5: Execute the main experiment in a controlled order

**Files:**
- Verify: `/home/zhaodanqi/clone/WoTE/newCtrl/runs/train/main`
- Verify: `/home/zhaodanqi/clone/WoTE/newCtrl/runs/eval/main`

- [ ] **Step 1: Launch the main 1024-bundle training run**

Run:

```bash
mkdir -p newCtrl/runs/train/main
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NAVSIM_EXP_ROOT=/home/zhaodanqi/clone/WoTE/newCtrl/runs/train/main \
OPENSCENE_DATA_ROOT=/path/to/dataset \
RUN_FUSIONS=attn \
bash tool/training/0302train_wote_ctrlbundle_split_train_attn_ctrl1024.sh
```

Expected: Hydra outputs under `newCtrl/runs/train/main/...` and preflight confirms `newCtrl` controller assets.

- [ ] **Step 2: Select the checkpoint for the main report run**

Record:

```text
Selected checkpoint path:
/home/zhaodanqi/clone/WoTE/newCtrl/runs/train/main/<...>/checkpoints/<best>.ckpt
```

Selection basis: held-out style validation performance, with PDMS as primary metric.

- [ ] **Step 3: Run test-split evaluation for the selected checkpoint**

Run:

```bash
mkdir -p newCtrl/runs/eval/main
OPENSCENE_DATA_ROOT=/path/to/dataset \
NAVSIM_EXP_ROOT=/home/zhaodanqi/clone/WoTE/newCtrl/runs/eval/main \
CKPT=/abs/path/to/selected.ckpt \
CKPT_ATTN_CTRL1024=/abs/path/to/selected.ckpt \
CKPT_ATTN_CTRL128=/abs/path/to/selected.ckpt \
CKPT_ATTN_CTRL64=/abs/path/to/selected.ckpt \
GPU=0 \
EVAL_WORKER=ray_distributed_no_torch \
EVAL_CPU_THREADS=1 \
bash tool/evaluate/eval_ckpts_20260225_valstyles_v3_20260303_ref64_128_1024.sh
```

Expected: evaluation outputs under `newCtrl/runs/eval/main/...` with matched `newCtrl` controller assets.

- [ ] **Step 4: Archive the main result summary**

Record in a local note or CSV:

```text
- checkpoint path
- NC / DAC / EP / TTC / Comfort / PDMS
- dataset split
- controller bundle size
- planner anchor set
- git commit hash
```

- [ ] **Step 5: Commit any final script/document updates**

```bash
git add docs/superpowers/plans/2026-05-01-worldcap-main-experiments.md
git commit -m "docs: finalize worldcap main experiment execution order"
```
