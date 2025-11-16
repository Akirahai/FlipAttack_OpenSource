#!/bin/bash
set -euo pipefail

# ---------- Configuration ----------
gpus="4"
batch=32
# judge_llm="openai/gpt-oss-120b"
# judge_llm="openai/gpt-oss-20b"
judge_llm=""
# judge_api="gpt-4.0"
# judge_api="gemini/gemini-2.5-flash-lite"
judge_api=""
# judge_harm_bench=""
judge_harm_bench="cais/HarmBench-Llama-2-13b-cls"


#--------Set result and output dir ---------

# model_name="Llama-3.1-8B-Instruct"
# model_name="Llama-2-7b-chat-hf"
# model_name="Qwen2.5-7B-Instruct"
model_name="vicuna-7b-v1.5"


result_file="FlipAttack-FCS-CoT-LangGPT-Few-shot-${model_name}-advbench-0_519.json"
# checkpoint_file=""
checkpoint_file="FlipAttack-gpt-oss-120b-FCS-CoT-LangGPT-Few-shot-${model_name}-advbench-0_519.json"
# checkpoint_file="FlipAttack-FCS-CoT-LangGPT-Few-shot-Llama-2-7b-chat-hf-advbench-0_20_4068.json"
output_dir="result"
checkpoint_dir="checkpoint"
final_result_dir="final_result"

# ---------- Run ----------

python main_open_source_eval.py \
    --gpus $gpus \
    --batch $batch \
    --result_file "$result_file" \
    --checkpoint_file "$checkpoint_file" \
    --output_dir "$output_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --final_result_dir "$final_result_dir" \
    --judge_llm "$judge_llm" \
    --judge_api "$judge_api" \
    --judge_harm_bench "$judge_harm_bench"