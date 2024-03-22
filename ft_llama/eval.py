import json
import math
import os
from pathlib import Path
import sys

import argparse
import torch
from tqdm import tqdm
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from utils.prompter import Prompter
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from model import AdapterWrappertest, LlamaLoraWrapper, LlamaLoraWrappertest

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def main(
    test_file: str = None,
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",
    args = None
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    replace_llama_attn_with_flash_attn()
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            # load_in_8bit=load_8bit,
        )
        if "hylora" in lora_weights:
            model = AdapterWrappertest.from_pretrained(lora_weights).to(device)
            
        else:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)
    
    def load_features(dataset, hg_datapath, use_context=False, n_c=5):
        if 'NQ' in hg_datapath:
            data_path = 'Image/features/NQ'
        elif 'WQ' in hg_datapath:
            data_path = 'Image/features/WQ'
        else:
            data_path = 'Image/features/TQA'
        if use_context:
            data_path = f"{data_path}/context-{n_c}"
        else:
            data_path = f"{data_path}/context-0"
        print(f"load data_features from {data_path}")
        dataset_features = load_dataset("json", data_files={
                                        'train': f'{data_path}/train.json', 'validation': f'{data_path}/eval.json', 'test': f'{data_path}/test.json'})
        if 'test' not in hg_datapath:
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    dataset[split] = dataset[split].add_column(column=dataset_features[split]['features'], name='features')
        else:
            dataset = dataset.add_column(column=dataset_features['test']['features'], name='features')
        return dataset
    
    test_dataset = load_dataset('json', data_files={"test": test_file})
    test_dataset = load_features(test_dataset, lora_weights)
    print(test_dataset)
    
    prompts, results, answers, features = [], [], [], []
    
    for batch in tqdm(test_dataset['test']):
        instruction = batch['instruction']
        input = batch['input']
        answers.append(batch['output'])
        prompts.append(prompter.generate_prompt(instruction, input))
        features.append(batch['features'])
    
    
    temperature = 0.0
    do_sample = temperature > 0.0
    top_p=0.75
    top_k=40
    num_beams=1
    
    with torch.autocast(device, dtype=torch.bfloat16):
        for batched_prompts, batched_features in tqdm(zip(chunks(prompts, args.batch_size), chunks(features, args.batch_size)), total=math.ceil(len(prompts) / args.batch_size)):
            inputs = tokenizer(batched_prompts, truncation=False, return_tensors="pt", padding=True).to(model.device)
            if "hylora" in lora_weights:
                outputs = model.generate(
                    **inputs,
                    features=torch.Tensor([feature for feature in batched_features]).to(model.device),
                    max_new_tokens=32,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    return_dict_in_generate=False,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    return_dict_in_generate=False,
                )
            for i, generated_sequence in enumerate(outputs):
                input_ids = inputs["input_ids"][i]
                text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        tokenizer.decode(
                            input_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                    )
                new_text = text[prompt_length:]
                results.append(new_text)
    path = Path(lora_weights) / "results"
    path.mkdir(parents=True, exist_ok=True)
    with open(f'{path}/results.jsonl', "w", encoding="utf-8") as f:
        for pred, answer in zip(results, answers):
            result = {
                "answer": answer,
                "pred": pred,
            }
            f.write(json.dumps(result) + "\n")
    
    # while True:
    #     instruction = input("Input:")
    #     if len(instruction.strip()) == 0:
    #         break
    #     print("Response:", evaluate(instruction))


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--test_file', default=None, type=str, required=True)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lora_weights', default=None, type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    args = parser.parse_args()
    
    main(args.test_file, args.load_8bit, args.base_model, args.lora_weights, "", args)
