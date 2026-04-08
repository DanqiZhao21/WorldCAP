#!/usr/bin/env bash
set -euo pipefail

# Run from repo root (recommended):
#   bash tool/smalltool/worldCAPdraw0002-0189_allviews_0087-0189.sh

# You can override GPU before running, e.g.:
#   CUDA_VISIBLE_DEVICES=1 bash tool/smalltool/worldCAPdraw0002-0189_allviews_0087-0189.sh
export CUDA_VISIBLE_DEVICES=1

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
  87 \
  WorldCAP_pic/WorldCAPdataFor0087/compare_default_vs_attn0087_tokens_1_182_copy.txt \
  WorldCAP_pic/WorldCAPdataFor0087

run_one_style \
  159 \
  WorldCAP_pic/WorldCAPdataFor0159/compare_default_vs_attn0159_tokens_1_173_copy.txt \
  WorldCAP_pic/WorldCAPdataFor0159

run_one_style \
  189 \
  WorldCAP_pic/WorldCAPdataFor0189/compare_default_vs_attn0189_tokens_1_61_copy.txt \
  WorldCAP_pic/WorldCAPdataFor0189
