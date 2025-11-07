import os
import json
import pandas
import argparse
from tqdm import tqdm
from eval_util import Evaluator
from flip_attack import FlipAttack


from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("FlipAttack")
    
    # victim LLM
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 6], help='List of gpus to use')
    parser.add_argument("--victim_llm", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="name of victim LLM") # Our experiments for FlipAttack were conducted in 11/2025.
    parser.add_argument("--temperature", type=float, default=0, help="temperature of victim LLM")
    parser.add_argument("--max_token", type=int, default=512, help="max output tokens")
    # parser.add_argument("--retry_time", type=int, default=1000, help="max retry time of failed API calling")
    # parser.add_argument("--failed_sleep_time", type=int, default=1, help="sleep time of failed API calling")
    # parser.add_argument("--round_sleep_time", type=int, default=1, help="sleep time of round")
    parser.add_argument("--batch", type = int, default = 32, help="batch number of parallel process")

    # FlipAttack
    parser.add_argument("--flip_mode", type=str, default="FCS", choices=["FWO", "FCW", "FCS", "FMM"], 
                        help="flipping mode: \
                        (I) Flip Word Order (FWO)\
                        (II) Flip Chars in Word (FCW)\
                        (III) Flip Chas in Sentence (FCS)\
                        (IV) Fool Model Mode (FMM)")
    parser.add_argument("--cot", action="store_true", help="use chain-of-thought")
    parser.add_argument("--lang_gpt", action="store_true", help="use LangGPT")
    parser.add_argument("--few_shot", action="store_true", help="use task-oriented few-shot demo")
    
    # harmful data
    parser.add_argument("--data_name", type=str, default="advbench", choices=["advbench", "advbench_subset"], help="benchmark name")
    parser.add_argument("--begin", type=int, default=0, help="begin of test data for debug")
    parser.add_argument("--end", type=int, default=519, help="end of test data for debug")
    parser.add_argument("--output_dict", type=str, default="../reproduce_result", help="output path")
    
    # evaluation
    # parser.add_argument("--eval", action="store_true", help="evaluate the attack success rate")
    parser.add_argument("--judge_llm", type=str, default="gpt-4-0613", help="name of judge LLM")
    
    args = parser.parse_args()

    # GPU Usage
        
    GPU_list = ','.join(map(str, args.gpus))
    
    
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    os.environ["WORLD_SIZE"] = "1"
    print(f"Using GPU: {GPU_list}")

    # data path
    if args.data_name=='advbench':
        args.data_path = "../data/harmful_behaviors.csv"
    elif args.data_name=='advbench_subset':
        args.data_path = "../data/harmful_behaviors_subset.csv"
    else:
        raise NameError("Unknown Benchmark {}, please add the implementation.".format(args.data_name))

    # init victim llm with vllm library

    sampling_params = SamplingParams(temperature=args.temperature,top_p=1, max_tokens=args.max_token)
    llm = LLM(model=args.victim_llm, tensor_parallel_size = len(args.gpus))
    # Initialize tokenizer for chat template
    if "vicuna" in args.victim_llm.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        print("Using Llama-3.1-8B-Instruct tokenizer chat template for Vicuna model.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.victim_llm)


    # victim_llm = LLM(model_id=args.victim_llm,
    #            temperature=args.temperature,
    #            max_tokens=args.max_token,
    #            retry_time=args.retry_time,
    #            failed_sleep_time=args.failed_sleep_time,
    #            round_sleep_time=args.round_sleep_time)

    # load data
    adv_bench = pandas.read_csv(args.data_path)

    # result with key as id and value as dict
    result_dicts = {}

    # Attack the model in batches using vllm
    # We prepare a batch of prompts, call vllm once per batch and map responses back to items
    import math
    total_batches = math.ceil((args.end - args.begin) / args.batch)
    
    for batch_idx, batch_start in enumerate(range(args.begin, args.end, args.batch), start=1):
        batch_end = min(batch_start + args.batch, args.end)

        prompts = []
        batch_result_indices = []

        # prepare flip attacks for this batch and reserve result entries for the problems
        print(f"Preparing batch {batch_idx}/{total_batches} ({batch_start} to {batch_end})...")
        for idx in range(batch_start, batch_end):
            harm_prompt = adv_bench["goal"].iloc[idx]

            # FlipAttack (generate the adversarial prompt/attack)
            attack_model = FlipAttack(flip_mode=args.flip_mode,
                                      cot=args.cot,
                                      lang_gpt=args.lang_gpt,
                                      few_shot=args.few_shot,
                                      victim_llm=args.victim_llm)

            log, flip_attack = attack_model.generate(harm_prompt)
            
            formatted_prompt_flip_attack = tokenizer.apply_chat_template(flip_attack, 
                                                                         tokenize=False, 
                                                                         add_generation_prompt=True)

            # create placeholder result dict (output will be filled after LLM generation)
            result_dict = {
                "id": idx,
                "goal": harm_prompt,
                "flip_attack": log,
                "all_prompt": flip_attack,
                "formatted_prompt": formatted_prompt_flip_attack,
                "output": None,
            }

            if idx not in result_dicts:
                result_dicts[idx] = result_dict
            else:
                print(f"Warning: Duplicate idx {idx} in result_dicts.")
                result_dicts[idx].update(result_dict)

            # Prepare Batch Inputs
            prompts.append(formatted_prompt_flip_attack)
            batch_result_indices.append(idx)

        # call the victim LLM once for the batch
        if len(prompts) > 0:
            # vllm will handle multiple prompts in a single call

            llm_responses = llm.generate(prompts, sampling_params=sampling_params)

            # Map responses back to the corresponding result dicts.
            # We assume the generator yields one main response per input in order.
            for i, resp in enumerate(llm_responses):
                try:
                    text = resp.outputs[0].text
                except Exception:
                    # Fallback if response object shape differs
                    text = str(resp)

                # store as a set to keep previous behavior (outputs could be multiple variants)
                result_dicts[batch_result_indices[i]]["output"] = text
    


    # save result
    os.makedirs(args.output_dict, exist_ok=True)
    victim_llm_name = args.victim_llm.split("/")[-1]

    output_file_name = "{}/FlipAttack-{}{}{}{}-{}-{}-{}_{}_{}.json".format(args.output_dict,
                                                                        args.flip_mode, 
                                                                        "-CoT" if(args.cot) else "",
                                                                        "-LangGPT" if(args.lang_gpt) else "", 
                                                                        "-Few-shot" if(args.few_shot) else "", 
                                                                        victim_llm_name, 
                                                                        args.data_name, 
                                                                        args.begin, 
                                                                        args.end,
                                                                        args.max_token
                                                                        )

    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(result_dicts, f, ensure_ascii=False, indent=4)
    print(f"Saved FlipAttack results to {output_file_name}")
