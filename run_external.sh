#!/bin/bash
set -e
MODEL_DIR=${MODEL_DIR:-models}
GPU=${GPU:-0}
ext() { CUDA_VISIBLE_DEVICES=$GPU python extract_external.py --family "$1" --model_path "$MODEL_DIR/$2" --tag "$3" --dataset "$4"; }
for ds in btv_eval mm_safety jailbreakv; do
  ext qwen3vl Qwen3-VL-2B-Instruct qwen3vl_2b "$ds"
  ext qwen3vl Qwen3-VL-4B-Instruct qwen3vl_4b "$ds"
  ext qwen3vl Qwen3-VL-8B-Instruct qwen3vl_8b "$ds"
done
python eval_beavertails.py
python eval_mm_safety.py
python eval_jailbreakv.py
