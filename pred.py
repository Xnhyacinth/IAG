import logging
import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from xopen import xopen
import pathlib
import sys
from copy import deepcopy
import dataclasses
import math
from tensor_parallel import tensor_parallel
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from src.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)
from pathlib import Path
from src.util import load_data
logger = logging.getLogger(__name__)

import re

def find_number_sequences(string):
    pattern = r'\d+'
    matches = re.findall(pattern, string)
    return matches if len(matches) > 0 else None

# Copied from https://github.com/DachengLi1/LongChat/blob/43d71f03d7711a2ab3b78ee8d1e38b65bb7fd22f/longeval/utils.py
def maybe_monkey_patch(model_name: str, longchat_flash_attn: bool, longchat_ratio: int):
    if "longchat" in model_name:
        from longchat.train.monkey_patch.llama_condense_monkey_patch import (
            replace_llama_with_condense,
        )

        replace_llama_with_condense(longchat_ratio)

        if longchat_flash_attn:
            from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import (
                replace_llama_attn_with_flash_attn,
            )

            replace_llama_attn_with_flash_attn()

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
    )
    parser.add_argument(
        "--p-type",
        default='rear',
        help="Type of truncation, include 'rear', 'initial', 'random'",
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--batch-size", help="Batch size use in generation", type=int, default=8)
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--use-truncation",
        action="store_true",
        help="Truncate the text.",
    )
    parser.add_argument(
        "--e",
        action="store_true",
    )
    parser.add_argument(
        "--image",
        action="store_true",
    )
    parser.add_argument(
        "--c-ratio",
        help="Ratio of the middle context.",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--n",
        help="Number of the subsequence.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_c",
        help="Number of the subsequence.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument(
        "--max-memory-per-gpu",
        help="Maximum memory to use per GPU (in GiB) for multi-device parallelism, e.g., 80",
        type=int,
    )
    parser.add_argument("--output-path", help="Path to write output file of generated responses")
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--longchat-flash-attn",
        action="store_true",
        help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.",
    )
    parser.add_argument(
        "--longchat-ratio",
        type=int,
        default=8,
        help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.",
    )
    return parser.parse_args(args)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, num_gpus, max_memory_per_gpu=None, longchat_flash_attn=None, longchat_ratio=None):
    logger.info("Loading model")
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, resume_download=True)
    elif "llama2" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, device_map='auto', torch_dtype=torch.bfloat16, resume_download=True)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        maybe_monkey_patch(model_name=path, longchat_flash_attn=longchat_flash_attn, longchat_ratio=longchat_ratio)
        # replace_llama_attn_with_flash_attn()
        model, tokenizer = load_model(
            path,
            device="cuda",
            num_gpus=num_gpus,
            max_gpu_memory=f"{max_memory_per_gpu}GiB",
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            resume_download=True,
        )
        model = tensor_parallel(model)
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)

    tokenizer.padding_side = "left"
    if "chatglm" not in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.eval()
    print(model)
    return model, tokenizer

def main(
    pre_prompts,
    answers,
    model,
    tokenizer,
    temperature,
    top_p,
    batch_size,
    prompt_mention_random_ordering,
    use_random_ordering,
    max_length,
    max_new_tokens,
    output_path,
    model_name,
    longchat_flash_attn,
    use_truncation,
    p_type,
    c_ratio,
    n
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = []
    did_format_warn = False
    f = 0 # 0: no truncation 1: < maxlength & truncation 2: > maxlength & truncation 3 : > maxlength & half 
    # Fetch all of the prompts
    for prompt in pre_prompts:
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            if use_truncation:
                if n > 100:
                    n = half
                prompt = truncate(tokenized_prompt, 0.25, max_length / len(tokenized_prompt), n, p_type)
            else:
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        elif use_truncation:
            if n > 100:
                n = int(len(tokenized_prompt) * 0.4 * c_ratio)
            prompt = truncate(tokenized_prompt, 0.3, c_ratio, n, p_type)
        
        prompt = format_instruct_prompt(tokenizer, prompt, model_name, did_format_warn)
        prompts.append(prompt)
        
    # Get responses for all of the prompts
    do_sample = temperature > 0.0

    responses = []
    with torch.autocast(device, dtype=torch.bfloat16):
        for batched_prompts in tqdm(chunks(prompts, batch_size), total=math.ceil(len(prompts) / batch_size)):
            inputs = tokenizer(batched_prompts, truncation=False, return_tensors="pt", padding=True).to(model.device)
            if "samsum" in output_path: # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=inputs["input_ids"][0].shape[-1]+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    num_beams=1,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    use_cache=True if "longchat" not in model_name else not ("longchat" in model_name and longchat_flash_attn),
                    eos_token_id=0 if 'mpt' in model_name else tokenizer.eos_token_id,
                    pad_token_id=0 if 'mpt' in model_name else tokenizer.pad_token_id,
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
                responses.append(new_text)
                # import pdb
                # pdb.set_trace()
    logger.info("saving at %s", output_path)
    with xopen(output_path + '.gz', "w") as f, open(output_path, "w", encoding="utf-8") as fo:
        for prompt, response, answer in zip(prompts, responses, answers):
            output_example = {}
            # Add some extra metadata to the output example
            output_example["pred"] = response
            output_example["answer"] = answer
            f.write(json.dumps(output_example) + "\n")
            fo.write(json.dumps(output_example) + "\n")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def random_select_subsequences(string, n, l):
    selected_string = ''
    substring_length = l // n
    p = 0
    for i in range(n):
        index = random.randint(p, len(string) - (n - i) * substring_length)
        selected_string += string[index:index + substring_length]
        selected_string += ' '  
        p = index + substring_length

    return selected_string.strip()  # 去除首尾空格

def truncate(c, p_ratio, c_ratio, n, p_type):
    pos = int(len(c) * p_ratio)
    c_middle = tokenizer.decode(c[pos:-pos], skip_special_tokens=True)
    if p_type == 'random':
        c_middle = random_select_subsequences(c_middle, n, int(c_ratio * len(c_middle)))
    elif p_type == 'equal':
        c_middle = c_middle[::round(1 / c_ratio)]
    elif p_type == 'initial':
        c_middle = c_middle[:int(c_ratio * len(c_middle))]
    elif p_type =='rear':
        c_middle = c_middle[-int(c_ratio * len(c_middle)):]
    return tokenizer.decode(c[:pos], skip_special_tokens=True) + c_middle + tokenizer.decode(c[-pos:], skip_special_tokens=True)
    
def format_instruct_prompt(tokenizer, prompt, model_name, did_format_warn):
    if "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()        
    # elif "llama2" in model_name:
    #     prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "instruct" in model_name:
        if did_format_warn is False:
            logger.warning(f"Model {model_name} appears to be an instruct model, applying instruct formatting")
            did_format_warn = True
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        )
        prompt = "{intro}\n{instruction_key}\n{instruction}\n{response_key}\n".format(
            intro=INTRO_BLURB,
            instruction_key=INSTRUCTION_KEY,
            instruction=prompt,
            response_key=RESPONSE_KEY,
        )
    return prompt

def load_local_data(input_path, closedbook, use_random_ordering, prompt_format):
    prompts = []
    # Fetch all of the documents
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            if closedbook:
                documents = []
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if use_random_ordering:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            # Format the documents into strings
            formatted_documents = []
            for document_index, document in enumerate(documents):
                formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
            prompt = prompt_format.format(input=question, context="\n".join(formatted_documents))
            
            prompts.append(prompt)
    return prompts
            

if __name__ == '__main__':
    seed_everything(42)
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    args = parse_args()
    logger.info("running %s", " ".join(sys.argv))
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    
    model_name = args.model
    checkpoint_path = Path(f"predictions/{model_name}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    args.output_dir = checkpoint_path
    # define your model
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, args.num_gpus, args.max_memory_per_gpu, args.longchat_flash_attn, args.longchat_ratio)
    max_length = model2maxlen[model_name]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if 'nq-open' in args.input_path:
        dataset = 'nq-open'
        if args.closedbook:
            dataset += '-cbqa'
        elif args.prompt_mention_random_ordering:
            dataset += '-random'
        elif args.query_aware_contextualization:
            dataset += '-query-aware'
        if not os.path.exists("predictions"):
            os.makedirs("predictions")
        datasets = [dataset]
    elif 'longbench' in args.input_path:
        if args.e:
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
                "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        else:
            datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        if not os.path.exists("pred"):
            os.makedirs("pred")
        if not os.path.exists("pred_e"):
            os.makedirs("pred_e")
    elif 'Image' in args.input_path:
        # if args.closedbook:
        #     datasets = ['NQ-cbqa', 'TQA-cbqa', 'WQ-cbqa']
        # elif args.image:
        #     datasets = ['NQ-image', 'TQA-image', 'WQ-image']
        # else:
        datasets = ['NQ', 'TQA', 'WQ']
    else:
        raise ValueError(f"Did not find any dataset for example: {args.input_path}")
    
    for dataset in datasets:
        print(dataset)
        prompt_format = dataset2prompt[dataset]
        max_new_tokens = dataset2maxlen[dataset]
        if 'longbench' in args.input_path:
            if args.e:
                data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
                if not os.path.exists(f"pred_e/{model_name}"):
                    os.makedirs(f"pred_e/{model_name}")
                if args.use_truncation:
                    dataset += f'-{args.p_type}_truncation'
                output_path = f"pred_e/{model_name}/{dataset}.jsonl"
            else:
                data = load_dataset('THUDM/LongBench', dataset, split='test')
                if not os.path.exists(f"pred/{model_name}"):
                    os.makedirs(f"pred/{model_name}")
                if args.use_truncation:
                    dataset += f'-{args.p_type}_truncation'
                output_path = f"pred/{model_name}/{dataset}.jsonl"
            prompts = [prompt_format.format(**json_obj) for json_obj in tqdm(data)]
        elif 'Image' in args.input_path:
            data = load_data(args, f'{args.input_path}/{dataset}/test')
            if args.closedbook:
                dataset += '-cbqa'
            elif args.image:
                dataset += '-image'
            prompt_format = dataset2prompt[dataset]
            if '-' in dataset:
                prompts = [prompt_format.format(question=json_obj['question'], context=json_obj['context']['compressed_prompt'][204:] if 'NQ' in dataset and args.n_c == 5 else json_obj['context']['compressed_prompt'][194:]) for json_obj in tqdm(data)]
            else:
                f = 'title:' + " {} " + 'context:' + " {}"
                prompts = [prompt_format.format(question=json_obj['question'], context=' '.join([f.format(c['title'], c['text']) for c in json_obj['ctxs'][:10]])) for json_obj in tqdm(data)]
            answers = [json_obj['answers'] for json_obj in tqdm(data)]
            output_path = f"{args.output_dir}/{dataset}.jsonl"
        else:
            prompts = load_local_data(args.input_path, args.closedbook, args.use_random_ordering, prompt_format)

            docs = find_number_sequences(args.input_path.split('/')[-1])
            if docs:
                num_docs, pos = docs[0], docs[1]
                dataset += f'-{num_docs}_docs_at_{pos}'
            if not os.path.exists(f"predictions/{model_name}"):
                os.makedirs(f"predictions/{model_name}")
            if args.use_truncation:
                dataset += f'-{args.p_type}_truncation'
            output_path = f"predictions/{model_name}/{dataset}.jsonl"
            o = output_path.split('.')

            if len(max(tokenized_prompts, key=len)) > max_length:
                if args.use_truncation:
                    o[0] += f'_beyond_{args.c_ratio}'
                    if args.p_type == 'random':
                        if args.n > 100:
                            o[0] += f'_max'
                        else:
                            o[0] += f'_{args.n}'
                else:
                    o[0] += f'_beyond_half'
            elif args.use_truncation:
                o[0] += f'_{args.c_ratio}'
                if args.p_type == 'random':
                    if args.n > 100:
                        o[0] += f'_max'
                    else:
                        o[0] += f'_{args.n}'
            output_path = '.'.join(o)
        
        tokenized_prompts = [tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0] for prompt in prompts]
        
        print(output_path)
        main(
            prompts,
            answers,
            model,
            tokenizer,
            args.temperature,
            args.top_p,
            args.batch_size,
            args.prompt_mention_random_ordering,
            args.use_random_ordering,
            max_length,
            max_new_tokens,
            output_path,
            model_name,
            args.longchat_flash_attn,
            args.use_truncation,
            args.p_type,
            args.c_ratio,
            args.n
        )
        logger.info("finished running %s", sys.argv[0])