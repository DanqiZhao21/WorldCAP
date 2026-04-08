#!/usr/bin/env bash
set -euo pipefail

# Run from repo root (recommended):
#   bash tool/smalltool/worldCAPdraw0002-0189_allviews_0025-0041.sh

# You can override GPU before running, e.g.:
#   CUDA_VISIBLE_DEVICES=2 bash tool/smalltool/worldCAPdraw0002-0189_allviews_0025-0041.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

SPLIT=${SPLIT:-val}

BASELINE_CKPT=${BASELINE_CKPT:-/home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt}
WORLDCAP_CKPT=${WORLDCAP_CKPT:-/home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt}

run_one_style () {
  local style_idx="$1"
  local token_file="$2"
  local out_dir="$3"

  echo ""
  echo "==============================="
  echo "style_idx=${style_idx}"
  echo "tokens=${token_file}"
  echo "out_dir=${out_dir}"
  echo "split=${SPLIT}"
  echo "==============================="

  # 1) BEV compare (WoTE vs WorldCAP)
#   python tool/smalltool/visualize_token_compare.py \
#     --tokens "${token_file}" \
#     --style-idx "${style_idx}" \
#     --split "${SPLIT}" \
#     --baseline-ckpt "${BASELINE_CKPT}" \
#     --film-ckpt "${WORLDCAP_CKPT}" \
#     --out-dir "${out_dir}"

  # 2) Front camera trajectory overlay (GT + WoTE + WorldCAP)
  python tool/smalltool/visualize_token_frontcam_traj_compare.py \
    --tokens "${token_file}" \
    --split "${SPLIT}" \
    --baseline-ckpt "${BASELINE_CKPT}" \
    --worldcap-ckpt "${WORLDCAP_CKPT}" \
    --out-dir "${out_dir}"

  # 3) 6-view (2x3) mosaic
  python tool/smalltool/visualize_token_cameras_6view.py \
    --tokens "${token_file}" \
    --split "${SPLIT}" \
    --out-dir "${out_dir}"

  # 4) 8-camera individual frames
  python tool/smalltool/visualize_token_cameras_8cam.py \
    --tokens "${token_file}" \
    --split "${SPLIT}" \
    --out-dir "${out_dir}"
}

run_one_style \
  25 \
  WorldCAP_pic/WorldCAPdataFor0025/compare_default_vs_attn0025_tokens_1_200_copy.txt \
  WorldCAP_pic/WorldCAPdataFor0025

run_one_style \
  41 \
  WorldCAP_pic/WorldCAPdataFor0041/compare_default_vs_attn0041_tokens_1_200_copy.txt \
  WorldCAP_pic/WorldCAPdataFor0041
