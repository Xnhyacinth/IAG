from transformers import AutoTokenizer
import numpy as np
import os
import argparse
import torch
from huggingface_hub import HfFolder
from datasets import concatenate_datasets
from datasets import load_dataset
import data
# from data import load_data_keywords

prompt_template = f"Imagine and Compress contexts based on the question:\n{{input}}\nContexts:\n"

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-base", help="Model id to use for training.")
    parser.add_argument("--dataset_path", type=str, default="compress_data/NQ", help="Path to the already processed dataset.")
    parser.add_argument("--save_dataset_path", type=str, default="data/keywords_data/TQA", help="Path to the already processed dataset.")
    parser.add_argument('--train_data', type=str, default='none', help='path of train data')
    parser.add_argument('--dev_data', type=str, default='none', help='path of eval data')
    parser.add_argument('--test_data', type=str, default='none', help='path of test data')
    args = parser.parse_known_args()
    return args

def preprocess_function(sample, padding="max_length"):
    # created prompted input
    inputs = [prompt_template.format(input=item) for item in sample['question']]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=[item['compressed_prompt'] for item in sample['compressed_prompt']], max_length=tokenizer.model_max_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def analyze_data(tokenizer, dataset, max_sample_length):
    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x['question'], truncation=True), batched=True, remove_columns=['question', 'keywords'])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    max_source_length = min(max_source_length, max_sample_length)
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x['keywords'], truncation=True), batched=True, remove_columns=['question', 'keywords'])
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # use 95th percentile as max target length
    max_target_length = int(np.percentile(target_lenghts, 95))
    print(f"Max target length: {max_target_length}")
    
    # Prompt length: 16
    # Max input length: 496
    # Max source length: 277
    # Max target length: 512

def main():
    opt, _ = parse_arge()
    # training_function(args)
    # Load tokenizer of FLAN-t5-base
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(opt.model_id)
    prompt_length = len(tokenizer(prompt_template.format(input=""))["input_ids"])
    max_sample_length = tokenizer.model_max_length - prompt_length
    print(f"Prompt length: {prompt_length}")
    print(f"Max input length: {max_sample_length}")
    
    dataset = load_dataset("json", data_files={'train':opt.train_data, 'dev':opt.dev_data, 'test':opt.test_data})
    # analyze_data(tokenizer, dataset, max_sample_length)

    # process dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features))

    # save dataset to disk
    tokenized_dataset["train"].save_to_disk(os.path.join(opt.save_dataset_path,"train"))
    tokenized_dataset["dev"].save_to_disk(os.path.join(opt.save_dataset_path,"dev"))
    tokenized_dataset["test"].save_to_disk(os.path.join(opt.save_dataset_path,"test"))

    
if __name__ == "__main__":
    main()
