#!/bin/bash
set -euo pipefail

# Train Controller Response Predictor (CAP plug-in) from generated controller style data.
# This is intended to be trained *separately* from WoTE planner training.
#
# Inputs you mentioned:
#   - /home/zhaodanqi/clone/WoTE/ControllerExp/generated/controller_styles.npz
#   - /home/zhaodanqi/clone/WoTE/ControllerExp/generated/debug_default_none.npy (optional fallback)
#
# Output:
#   - a .pt checkpoint containing controller_encoder + response_predictor state_dicts

ROOT=/home/zhaodanqi/clone/WoTE

# Prefer the currently active environment's python.
# You can override by exporting PYTHON=/abs/path/to/python
if [[ -n "${PYTHON:-}" ]]; then
    :
elif command -v python >/dev/null 2>&1; then
    PYTHON=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=$(command -v python3)
else
    echo "[ERR] Cannot find python. Activate your conda/venv first or export PYTHON=/abs/path/to/python" >&2
    exit 1
fi

BUNDLE_IN=${BUNDLE_IN:-"${ROOT}/ControllerExp/generated/controller_styles.npz"}
DEBUG_DEFAULT_EXEC=${DEBUG_DEFAULT_EXEC:-"${ROOT}/ControllerExp/generated/debug_default_none.npy"}

OUT_CKPT=${OUT_CKPT:-"${ROOT}/ControllerExp/generated/controller_response_predictor.pt"}
DEVICE=${DEVICE:-"cuda"}
EPOCHS=${EPOCHS:-30}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-2000}
BATCH_SIZE=${BATCH_SIZE:-16}
PHI_BANK=${PHI_BANK:-32}
LR=${LR:-1e-3}
SEED=${SEED:-0}
FEATURE_MODE=${FEATURE_MODE:-full}   # full | lateral_only

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Make GPUs visible (predictor training itself runs on a single GPU, typically cuda:0 of the visible set).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

echo "[INFO] Using PYTHON=${PYTHON}"
"${PYTHON}" -c "import sys; print('[INFO] python:', sys.executable); print('[INFO] version:', sys.version.replace('\n',' '))" || true

if [[ ! -f "${BUNDLE_IN}" ]]; then
  echo "[ERR] BUNDLE_IN not found: ${BUNDLE_IN}" >&2
  exit 1
fi

TMP_BUNDLE=$(mktemp --suffix=.npz)
cleanup() {
  rm -f "${TMP_BUNDLE}" || true
}
trap cleanup EXIT

# Normalize the bundle to have keys:
#   exec_trajs: [S,N,T,3]
#   ref_traj:   [N,T,3]
# If exec_trajs is missing but DEBUG_DEFAULT_EXEC exists as [N,T,3], we will create S=1 exec_trajs.

"${PYTHON}" - <<PY
import os
import numpy as np

bundle_in = os.environ.get('BUNDLE_IN', r'''${BUNDLE_IN}''')
debug_exec = os.environ.get('DEBUG_DEFAULT_EXEC', r'''${DEBUG_DEFAULT_EXEC}''')
tmp_out = os.environ.get('TMP_BUNDLE', r'''${TMP_BUNDLE}''')

data = np.load(bundle_in, allow_pickle=True)
keys = set(data.keys())

# Candidate key aliases
exec_keys = ['exec_trajs', 'exec', 'exec_traj', 'exec_trajectories', 'executed_trajs']
ref_keys  = ['ref_traj', 'ref', 'ref_trajs', 'reference_traj', 'reference_trajs']
style_keys = ['style_names', 'styles', 'style']

def pick_first(existing, candidates):
    for k in candidates:
        if k in existing:
            return k
    return None

k_exec = pick_first(keys, exec_keys)
k_ref = pick_first(keys, ref_keys)
k_style = pick_first(keys, style_keys)

exec_trajs = data[k_exec] if k_exec is not None else None
ref_traj = data[k_ref] if k_ref is not None else None
style_names = data[k_style] if k_style is not None else None

# If ref_traj is missing, we cannot proceed.
if ref_traj is None:
    raise RuntimeError(f"bundle missing ref_traj-like key; keys={sorted(list(keys))}")

# Normalize ref_traj to [N,T,3]
ref_traj = np.asarray(ref_traj)
if ref_traj.ndim == 2 and ref_traj.shape[-1] == 3:
    # [T,3] -> treat as single trajectory (N=1)
    ref_traj = ref_traj[None, ...]
if ref_traj.ndim != 3 or ref_traj.shape[-1] != 3:
    raise RuntimeError(f"ref_traj must be [N,T,3], got {ref_traj.shape}")

N, T, D = ref_traj.shape

# Normalize exec_trajs
if exec_trajs is None:
    if os.path.isfile(debug_exec):
        dbg = np.load(debug_exec, allow_pickle=True)
        dbg = np.asarray(dbg)
        if dbg.ndim == 3 and dbg.shape[-1] == 3:
            # assume [N,T,3]
            if dbg.shape[0] != N or dbg.shape[1] != T:
                raise RuntimeError(f"debug exec shape {dbg.shape} mismatches ref {ref_traj.shape}")
            exec_trajs = dbg[None, ...]  # [1,N,T,3]
            style_names = np.array(['default_none'], dtype=object)
        else:
            raise RuntimeError(f"debug exec must be [N,T,3], got {dbg.shape}")
    else:
        raise RuntimeError("bundle missing exec_trajs and DEBUG_DEFAULT_EXEC not found")
else:
    exec_trajs = np.asarray(exec_trajs)
    if exec_trajs.ndim == 3 and exec_trajs.shape[-1] == 3:
        # [N,T,3] -> [1,N,T,3]
        exec_trajs = exec_trajs[None, ...]
    if exec_trajs.ndim != 4 or exec_trajs.shape[-1] != 3:
        raise RuntimeError(f"exec_trajs must be [S,N,T,3], got {exec_trajs.shape}")
    if exec_trajs.shape[1] != N or exec_trajs.shape[2] != T:
        raise RuntimeError(f"exec_trajs {exec_trajs.shape} mismatches ref {ref_traj.shape}")

S = exec_trajs.shape[0]

if style_names is None:
    style_names = np.array([f'style_{i}' for i in range(S)], dtype=object)
else:
    style_names = np.asarray(style_names)
    if style_names.shape[0] != S:
        style_names = np.array([f'style_{i}' for i in range(S)], dtype=object)

np.savez_compressed(
    tmp_out,
    exec_trajs=exec_trajs.astype(np.float32),
    ref_traj=ref_traj.astype(np.float32),
    style_names=style_names,
)

print('[OK] Normalized bundle saved to:', tmp_out)
print('     exec_trajs:', exec_trajs.shape, exec_trajs.dtype)
print('     ref_traj  :', ref_traj.shape, ref_traj.dtype)
print('     styles    :', S)
PY

# Train the plug-in predictor.
"${PYTHON}" "${ROOT}/tool/controller_response/train_controller_response_predictor.py" \
  --bundle "${TMP_BUNDLE}" \
  --out "${OUT_CKPT}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --steps-per-epoch "${STEPS_PER_EPOCH}" \
  --batch-size "${BATCH_SIZE}" \
  --phi-bank "${PHI_BANK}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --pool mean \
  --feature-mode "${FEATURE_MODE}"

echo "[DONE] Saved predictor ckpt: ${OUT_CKPT}"

# Next: enable CAP risk in evaluation
#   export WOTE_USE_CTRL_RISK=1
#   export WOTE_CTRL_RP_CKPT="${OUT_CKPT}"
#   export WOTE_CTRL_RISK_W=0.2
#   bash ${ROOT}/tool/evaluate/evaluate_all_0202.sh
