# FlipAttack — Open‑source vllm Test Runner

This README documents the small open-source test runner included in this repository. It focuses only on the local/vllm runner files under `src/` and explains how to run `main_open_source.py` and `main_open_source_eval.py` to test open-source victim models.

## What this runner does

- Generates adversarial "flip" prompts from a CSV benchmark (`data/harmful_behaviors.csv`).
- Uses `vllm` to run batched generation on a local victim model (Hugging Face or compatible checkpoint).
- Saves a JSON checkpoint keyed by dataset id containing: `id`, `goal`, `flip_attack`, `all_prompt` (or `formatted_prompt`), and `output`.
- Optionally, a separate evaluation script (`main_open_source_eval.py`) can load the checkpoint and run judge-LM evaluation (resumable, batched).

This runner is intended for testing open-source models locally (or on your cluster) without calling cloud APIs.

## Quick environment & installation

1. Create and activate a Python environment (recommended):

```cmd
conda activate newenv
pip install -r requirements.txt
```

2. Install `vllm` according to their docs (GPU + CUDA version must match your environment). See https://vllm.ai for platform-specific instructions.

3. Make sure you have the dataset CSV at `data/harmful_behaviors.csv` (or use `--data_name advbench_subset`).


## Files

- `src/main_open_source.py` — Generate flip attacks and run the victim LLM using `vllm.generate()` in batches. Saves checkpoint JSON keyed by dataset id.
- `src/main_open_source_eval.py` — Loads the checkpoint and runs judge-LM evaluation (resumable; writes checkpoint after each batch). Useful if you want to run expensive judge-LM evaluation separately.

## Quick run 

Run a small test to verify everything works:

```cmd
cd ./src
python main_open_source.py --gpus 0 --victim_llm Qwen/Qwen2.5-7B-Instruct --begin 0 --end 8 --batch 4 --output_dict result
```

- `--gpus`: list of GPU indices to expose to CUDA (maps to `CUDA_VISIBLE_DEVICES`). Use indices available on your machine.
- `--victim_llm`: model identifier (Hugging Face repo name or local model id). If using Qwen or others, make sure you have access and the tokenizer is compatible.
- `--begin` / `--end`: slice the dataset for quick tests.
- `--batch`: number of prompts sent to `vllm.generate()` in one call.
- `--output_dict`: directory to save checkpoint files.

Example evaluation run (after generator finished and produced checkpoint):

```cmd
python src\main_open_source_eval.py --gpus 0 --victim_llm Qwen/Qwen2.5-7B-Instruct --begin 0 --end 8 --batch 4 --eval --judge_llm gpt-4-0613 --output_dict result
```

Note: the eval script expects the checkpoint created by the generator and will resume evaluation for entries that don't have judge fields yet.

## Output format

The generator saves a JSON object mapping dataset id -> result dict. Example result dict fields:

- `id`: dataset row id (int when used in Python; JSON keys are strings)
- `goal`: original harmful prompt text
- `flip_attack`: internal log or summary of flips applied
- `all_prompt` / `formatted_prompt`: the final prompt sent to the model
- `output`: the model-generated text for that prompt
- Optional evaluation fields (added by the eval script): `judge_success_dict`, `judge_score_gpt4`, `judge_success_gpt4`

Because JSON keys must be strings, you will see numeric ids as strings when you open the JSON. Convert back with `int(k)` if you need numeric keys.

If you prefer a list, you can convert before saving (the eval script already contains a small conversion step that produces a list view of results).