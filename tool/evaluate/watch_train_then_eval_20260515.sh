#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhaodanqi/clone/WoTE
PYTHON=/home/zhaodanqi/anaconda3/bin/python
EXP_ROOT=/mnt/data/navsim_workspace/exp
TRAIN_PID=3166356
CUR_DIR="$ROOT/trainingResult/ckpts_20260515_162313"
BASE_CKPT="$ROOT/trainingResult/ckpts_20260514_213849/epoch=31-step=42560.ckpt"
OUT_BASE="eval/navtest_and_hard_20260515_auto"
LOG_DIR="$ROOT/EvaluationResult/auto_eval_20260515"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/watch_train_then_eval.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%F %T')] watcher started"

echo "[$(date '+%F %T')] waiting train pid=$TRAIN_PID to finish..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  latest=$(ls -1 "$CUR_DIR"/epoch=*-step=*.ckpt 2>/dev/null | sort -V | tail -n 1 || true)
  echo "[$(date '+%F %T')] training alive, latest_ckpt=${latest:-N/A}"
  sleep 120
done

echo "[$(date '+%F %T')] train pid ended. searching current final ckpt in $CUR_DIR"
CUR_CKPT=$(ls -1 "$CUR_DIR"/epoch=31-step=*.ckpt 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -z "${CUR_CKPT}" ]]; then
  echo "[$(date '+%F %T')] WARN: epoch=31 not found, fallback to last.ckpt"
  CUR_CKPT="$CUR_DIR/last.ckpt"
fi
if [[ ! -f "$CUR_CKPT" ]]; then
  echo "[$(date '+%F %T')] ERROR: current ckpt not found: $CUR_CKPT"
  exit 1
fi

echo "[$(date '+%F %T')] BASE_CKPT=$BASE_CKPT"
echo "[$(date '+%F %T')] CUR_CKPT=$CUR_CKPT"

SCENES=(
  navtest
  navtest_hard_curved_p90
  navtest_hard_curved_p95
  navtest_hard_dynamic_p90
  navtest_hard_fast_curve_p90
  navtest_hard_interaction_p90
  navtest_hard_composite_p90
)

run_one() {
  local tag="$1"
  local ckpt="$2"
  local scene="$3"
  local out="$EXP_ROOT/$OUT_BASE/$tag/$scene"
  mkdir -p "$out"
  echo "[$(date '+%F %T')] RUN tag=$tag scene=$scene"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$ROOT/navsim/planning/script/run_pdm_score_multiTraj.py" \
    agent=WoTE_agent \
    "agent.checkpoint_path='${ckpt}'" \
    "experiment_name=${OUT_BASE}/${tag}/${scene}" \
    split=test \
    "scene_filter=${scene}" \
    worker=sequential \
    max_number_of_workers=1 \
    +stream_worker_csv=true \
    evaluate_all_trajectories=false \
    verbose=true \
    2>&1 | tee "$out/run.log"
}

for scene in "${SCENES[@]}"; do
  run_one base_epoch31_42560 "$BASE_CKPT" "$scene"
  run_one cur_32ep_final "$CUR_CKPT" "$scene"
done

SUMMARY_CSV="$LOG_DIR/summary_metrics.csv"
SUMMARY_MD="$LOG_DIR/summary_metrics.md"

echo "subset,tag,score,no_at_fault_collisions,drivable_area_compliance,driving_direction_compliance,ego_progress,time_to_collision_within_bound,comfort" > "$SUMMARY_CSV"
for scene in "${SCENES[@]}"; do
  for tag in base_epoch31_42560 cur_32ep_final; do
    dir="$EXP_ROOT/$OUT_BASE/$tag/$scene"
    csv=$(find "$dir" -maxdepth 1 -name '*.csv' ! -name 'partial_*.csv' -type f | sort | tail -n 1)
    if [[ -z "${csv:-}" ]]; then
      continue
    fi
    "$PYTHON" - "$scene" "$tag" "$csv" >> "$SUMMARY_CSV" <<'PY'
import csv,sys
scene,tag,path=sys.argv[1],sys.argv[2],sys.argv[3]
with open(path) as f:
    r=csv.DictReader(f)
    avg=None
    for row in r:
        if row.get('token')=='average':
            avg=row
    if avg is None:
        raise SystemExit(0)
keys=['score','no_at_fault_collisions','drivable_area_compliance','driving_direction_compliance','ego_progress','time_to_collision_within_bound','comfort']
vals=[avg.get(k,'') for k in keys]
print(','.join([scene,tag]+vals))
PY
  done
done

"$PYTHON" - "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import pandas as pd,sys
inp,out=sys.argv[1],sys.argv[2]
df=pd.read_csv(inp)
if df.empty:
    print('no rows')
    raise SystemExit(0)
pivot=df.pivot(index='subset', columns='tag', values='score')
if {'cur_32ep_final','base_epoch31_42560'}.issubset(pivot.columns):
    delta=(pivot['cur_32ep_final']-pivot['base_epoch31_42560']).rename('delta_cur_minus_base')
    res=pd.concat([pivot,delta],axis=1).reset_index()
else:
    res=pivot.reset_index()
with open(out,'w') as f:
    f.write(res.to_markdown(index=False, floatfmt='.4f'))
print(res.to_markdown(index=False, floatfmt='.4f'))
PY

echo "[$(date '+%F %T')] done. summary: $SUMMARY_CSV ; $SUMMARY_MD"
