export CUDA_VISIBLE_DEVICES=2
# python tool/smalltool/visualize_token_compare.py \
#   --tokens WorldCAP_pic/WorldCAPdataFor0025/compare_default_vs_attn0025_tokens_1_200_copy.txt \
#   --style-idx 25 \
#   --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
#   --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
#   --out-dir WorldCAP_pic/WorldCAPdataFor0025 


python tool/smalltool/visualize_token_compare.py \
  --tokens WorldCAP_pic/WorldCAPdataFor0025/compare_default_vs_attn0025_tokens_1_200_copy.txt \
  --style-idx 25 \
  --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
  --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
  --out-dir WorldCAP_pic/WorldCAPdataFor0025 

python tool/smalltool/visualize_token_compare.py \
  --tokens WorldCAP_pic/WorldCAPdataFor0041/compare_default_vs_attn0041_tokens_1_200_copy.txt \
  --style-idx 41 \
  --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
  --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
  --out-dir WorldCAP_pic/WorldCAPdataFor0041

# python tool/smalltool/visualize_token_compare.py \
#   --tokens WorldCAP_pic/WorldCAPdataFor0087/compare_default_vs_attn0087_tokens_1_182_copy.txt \
#   --style-idx 87 \
#   --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
#   --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
#   --out-dir WorldCAP_pic/WorldCAPdataFor0087

# python tool/smalltool/visualize_token_compare.py \
#   --tokens WorldCAP_pic/WorldCAPdataFor0159/compare_default_vs_attn0159_tokens_1_173_copy.txt \
#   --style-idx 159 \
#   --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
#   --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
#   --out-dir WorldCAP_pic/WorldCAPdataFor0159

# python tool/smalltool/visualize_token_compare.py \
#   --tokens WorldCAP_pic/WorldCAPdataFor0189/compare_default_vs_attn0189_tokens_1_61_copy.txt \
#   --style-idx 189 \
#   --baseline-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/epoch=29-step=19950.ckpt \
#   --film-ckpt /home/zhaodanqi/clone/WoTE/trainingResult/ckpts_20260225_051808/epoch=39-step=53200.ckpt \
#   --out-dir WorldCAP_pic/WorldCAPdataFor0189

