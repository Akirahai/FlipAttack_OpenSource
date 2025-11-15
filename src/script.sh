#!/bin/bash
set -euo pipefail

# ---------- Configuration ----------
gpus="2 4"
begin=0
end=20
batch=4

output_dir="result"
flip_mode="FCS"
max_token=4068

# ---------- Toggle flags (set "true" or "false") ----------
USE_COT="true"
USE_FEW_SHOT="true"
USE_LANG_GPT="true"

# ---------- Models ----------
models=(
    # "Qwen/Qwen2.5-7B-Instruct"
    # "lmsys/vicuna-7b-v1.5"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-2-7b-chat-hf"
)

# ---------- Run ----------
for model in "${models[@]}"; do
    echo "======================================="
    echo "Running model: $model"
    echo "======================================="

    # Build extra flags based on variables
    extra_flags=""

    if [[ "${USE_COT,,}" == "true" ]]; then
        extra_flags+=" --cot"
    fi

    if [[ "${USE_FEW_SHOT,,}" == "true" ]]; then
        extra_flags+=" --few_shot"
    fi

    if [[ "${USE_LANG_GPT,,}" == "true" ]]; then
        extra_flags+=" --lang_gpt"
    fi

    python main_open_source.py \
        --gpus $gpus \
        --victim_llm "$model" \
        --begin $begin \
        --end $end \
        --batch $batch \
        --output_dict "$output_dir" \
        --flip_mode $flip_mode \
        --max_token $max_token $extra_flags

    echo "Finished model: $model"
    echo ""
done
