import json
    
model_dict = {
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "vicuna-7b-v1.5": "Vicuna-7B",
    "Llama-3.1-8B-Instruct": "Llama-3-8B",
    "Llama-2-7b-chat-hf": "Llama-2-7B"
}

victim_models = model_dict.keys()

col_width = max(len(model) for model in victim_models)
title = "ASR-GPT of FlipAttack against 8 LLMs on AdvBench".center(col_width*2+6)
print(f"{title}")

print(f"| {'-' * col_width} | {'-' * col_width} |")
header1 = "Victim LLM".center(col_width)
header2 = "ASR-GPT".center(col_width)
print(f"| {header1} | {header2} |")
print(f"| {'-' * col_width} | {'-' * col_width} |")

avg_asr_gpt = 0
for model in victim_models:
    
    input_path = "../result/FlipAttack-{}.json".format(model)
        
    with open(input_path, 'rb') as f:
        data = json.load(f)
        
    success = 0
    for idx, result_dict in enumerate(data):
        success += result_dict["judge_success_gpt4"]
    
    asr_gpt = success/len(data)*100
    avg_asr_gpt += asr_gpt
    
    col1 = model_dict[model].center(col_width)
    col2 = "{:.2f}%".format(asr_gpt).center(col_width)
    
    print(f"| {col1} | {col2} |")
        
    
print(f"| {'-' * col_width} | {'-' * col_width} |")

col1 = "Average".center(col_width)
col2 = "{:.2f}%".format(avg_asr_gpt/len(model_dict)).center(col_width)

print(f"| {col1} | {col2} |")

print(f"| {'-' * col_width} | {'-' * col_width} |")
