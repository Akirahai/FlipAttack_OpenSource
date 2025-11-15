#!/bin/bash
set -euo pipefail

# ---------- Configuration ----------
gpus="0 1 2 4"
batch=32
judge_llm="openai/gpt-oss-120b"
# judge_llm="openai/gpt-oss-20b"

#--------Set result and output dir ---------
result_dir="FlipAttack-FCS-CoT-LangGPT-Few-shot-Llama-2-7b-chat-hf-advbench-0_20_4068.json"
output_dir="result"
checkpoint_dir="checkpoint"
final_result_dir="final_result"

# ---------- Run ----------

python main_open_source_eval.py \
    --gpus $gpus \
    --batch $batch \
    --result_dir "$result_dir" \
    --output_dir "$output_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --final_result_dir "$final_result_dir" \
    --judge_llm $judge_llm