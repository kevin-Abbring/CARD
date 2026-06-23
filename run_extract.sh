#!/bin/bash
set -e
MODEL_DIR=${MODEL_DIR:-models}
GPU=${GPU:-0}
run() { CUDA_VISIBLE_DEVICES=$GPU python extract_curated.py --family "$1" --model_path "$MODEL_DIR/$2" --tag "$3"; }
run qwen3vl Qwen3-VL-2B-Instruct qwen3vl_2b
run qwen3vl Qwen3-VL-4B-Instruct qwen3vl_4b
run qwen3vl Qwen3-VL-8B-Instruct qwen3vl_8b
run gemma3  gemma-3-12b-it        gemma3_12b
run llava   llava-1.5-7b-hf       llava15_7b
python card_indomain.py
python card_method_search.py
python card_tables.py
python hiddendetect_baseline.py
