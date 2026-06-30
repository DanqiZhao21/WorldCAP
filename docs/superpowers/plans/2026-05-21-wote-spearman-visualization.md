# WoTE Spearman Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a token-level analysis tool that saves WoTE reward scores and Navsim PDM scores for every candidate trajectory, computes Spearman correlation, and renders a left-to-right trajectory-distribution visualization ordered by trajectory shape.

**Architecture:** Keep model inference unchanged. Add a standalone analysis script under `tool/visualize/` plus a small reusable utility module for score/ranking logic. The script will reuse the existing Navsim agent/scoring pipeline to load one token, produce candidate trajectories and WoTE scores, compute per-trajectory PDM scores in full mode, save a structured artifact file, then render a distribution plot ordered by trajectory curvature / heading change from left-turn to straight to right-turn.

**Tech Stack:** Python, NumPy, SciPy or a local Spearman implementation, Matplotlib, existing Navsim/Hydra config entrypoints.

---

### Task 1: Add reusable analysis helpers

**Files:**
- Create: `tool/visualize/wote_pdm_analysis.py`
- Test: `tests/tool/visualize/test_wote_pdm_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from tool.visualize.wote_pdm_analysis import (
    compute_spearman_rho,
    rank_trajectories_left_to_right,
)


def test_compute_spearman_rho_perfect_inverse():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    rho = compute_spearman_rho(x, y)
    assert rho == -1.0


def test_rank_trajectories_left_to_right_orders_by_yaw_then_lateral_offset():
    trajectories = np.array(
        [
            [[0.0, 0.0, -0.6], [4.0, -2.0, -0.6]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.6], [4.0, 2.0, 0.6]],
        ],
        dtype=np.float32,
    )
    order = rank_trajectories_left_to_right(trajectories)
    assert order.tolist() == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tool/visualize/test_wote_pdm_analysis.py -q`
Expected: fail because the module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def compute_spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    ...


def rank_trajectories_left_to_right(trajectories: np.ndarray) -> np.ndarray:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tool/visualize/test_wote_pdm_analysis.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tool/visualize/wote_pdm_analysis.py tests/tool/visualize/test_wote_pdm_analysis.py
git commit -m "feat: add woTe pdm analysis helpers"
```

### Task 2: Add single-token analysis script

**Files:**
- Create: `tool/visualize/visualize_wote_pdm_spearman.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from tool.visualize.wote_pdm_analysis import save_analysis_artifacts


def test_save_analysis_artifacts_writes_csv_and_npz(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tool/visualize/test_wote_pdm_analysis.py -q`
Expected: fail until save/load helpers exist.

- [ ] **Step 3: Write minimal implementation**

```python
def save_analysis_artifacts(output_dir: Path, data: dict) -> tuple[Path, Path]:
    ...


def main() -> int:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tool/visualize/test_wote_pdm_analysis.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tool/visualize/visualize_wote_pdm_spearman.py tool/visualize/wote_pdm_analysis.py
git commit -m "feat: add woTe pdm spearman visualization script"
```

### Task 3: Add distribution plot

**Files:**
- Modify: `tool/visualize/wote_pdm_analysis.py`
- Modify: `tool/visualize/visualize_wote_pdm_spearman.py`
- Test: `tests/tool/visualize/test_wote_pdm_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
def test_plot_distribution_creates_file(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tool/visualize/test_wote_pdm_analysis.py -q`
Expected: fail until plotting helper exists.

- [ ] **Step 3: Write minimal implementation**

```python
def plot_sorted_distribution(output_path: Path, trajectories: np.ndarray, wote_scores: np.ndarray, pdm_scores: np.ndarray) -> Path:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tool/visualize/test_wote_pdm_analysis.py -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tool/visualize/wote_pdm_analysis.py tool/visualize/visualize_wote_pdm_spearman.py tests/tool/visualize/test_wote_pdm_analysis.py
git commit -m "feat: add woTe pdm distribution plot"
```

### Task 4: Verify end-to-end token execution

**Files:**
- Modify: none if the script works cleanly

- [ ] **Step 1: Run the script on one known token**

Run:
```bash
python tool/visualize/visualize_wote_pdm_spearman.py --token <TOKEN> --split val --ckpt <CKPT> --out-dir <OUT_DIR>
```

- [ ] **Step 2: Confirm outputs**

Expected:
- one structured data file with WoTE scores, PDM scores, trajectory order, and Spearman rho
- one plot image ordered from left-turn to right-turn
- no changes to `navsim/agents/WoTE/WoTE_model.py`

- [ ] **Step 3: Commit**

```bash
git add tool/visualize/wote_pdm_analysis.py tool/visualize/visualize_wote_pdm_spearman.py
git commit -m "feat: add woTe pdm token visualization"
```

