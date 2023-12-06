# Copyright (c) 2023 Huanxuan Liao
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import BertTokenizer, AutoTokenizer, AutoModel
import json 
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from tqdm import tqdm
from data import load_data
import numpy as np
from datasets import load_from_disk

def load_data_compress(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k % world_size == global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples

class Simcsewrap(nn.Module):
    def __init__(self, model_path_simcse_roberta, length): #code_length为fc映射到的维度大小
        super(Simcsewrap, self).__init__()
        self.model = AutoModel.from_pretrained(model_path_simcse_roberta, cache_dir="/data2/huanxuan/.cache/huggingface/hub/", resume_download=True)
        embedding_dim = self.model.config.hidden_size

        self.fc = nn.Linear(embedding_dim, length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, input_masks):
        output = self.model(tokens, attention_mask=input_masks, output_hidden_states=True,
            return_dict=True)
        text_embeddings = output[0][:, 0, :]
        #output[0](batch size, sequence length, model hidden dimension)
        features = self.fc(text_embeddings)
        features=self.tanh(features)
        return features


def compute_kernel_bias(vecs, n_components=256):
    """compute kernel and bias
    vecs.shape = [num_samples, embedding_size]
    transfer:y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """ normalization
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def get_features(sample, tokenizer, encoder):
    question = ['Question: Please write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Only give me the answer and do not output any other words.'\
                'Question: ' + item + '\nAnswer:' for item in sample['question']]
    inputs = tokenizer(question, return_tensors='pt', padding=True)
    output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
    pooled_sentence = output.last_hidden_state # shape is [batch_size, seq_len, hidden_size]
    pooled_sentence = np.array(torch.mean(pooled_sentence, dim=1).cpu().detach().numpy())  
    kernel, bias = compute_kernel_bias(pooled_sentence, 255)
    pooled_sentence = transform_and_normalize(pooled_sentence, kernel=kernel, bias=bias)
    sample['features'] = pooled_sentence
    return sample

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str, default='data')
parser.add_argument("--dataset", type=str, default='TQA')
parser.add_argument("--d", type=str, default='train')
parser.add_argument("--num", type=int, default=5)
parser.add_argument("--cuda", type=int, default=0)
opt = parser.parse_args()
checkpoint_path = Path(f"features/{opt.dataset}/context-{opt.num}")
checkpoint_path.mkdir(parents=True, exist_ok=True)
# data = load_data_compress(f"{opt.datapath}/{opt.dataset}/{opt.d}.json")
dataset = load_from_disk(f'dataset/Image/{opt.dataset}')
model_path = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = Simcsewrap(model_path_simcse_roberta, 255).cuda()
model = AutoModel.from_pretrained(model_path).to(f'cuda:0')
# for d in tqdm(data, desc='Length'):
for split in ["train", "eval", "test"]:
    new_data = []
    data = dataset[split]
    questions = data["question"]
    context = data[f"compressed_ctxs_{opt.num}"]
    for d in tqdm(range(0, len(data), 256), desc='Length'):
        # qs = ['Question: ' + item['question'] + '\nAnswer:' for item in data[d:d+1024]]
        qs = ['Question: ' + q + " " + c["compressed_prompt"][194:] + '\nAnswer:' for (q, c) in zip(questions[d:d+256], context[d:d+256])]
        # inputs = tokenizer(d['question'], return_tensors='pt', padding=True).to('cuda:0')
        inputs = tokenizer(qs, #d['question']
                            max_length=512,
                            padding='max_length',
                            truncation='longest_first',
                            return_tensors="pt").to(f'cuda:0')
        with torch.no_grad():
            embeddings = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
        pooled_sentence = embeddings.last_hidden_state # shape is [batch_size, seq_len, hidden_size]
        pooled_sentence = np.array(torch.mean(pooled_sentence, dim=1).cpu().detach().numpy())
        # print(pooled_sentence)
        kernel, bias = compute_kernel_bias(pooled_sentence, 255)
        pooled_sentence = transform_and_normalize(pooled_sentence, kernel=kernel, bias=bias)
        # d['features'] = embeddings.cpu().numpy().tolist()
        # new_d = {}
        # new_d['features'] = pooled_sentence
        for i in range(pooled_sentence.shape[0]):
            new_d = {}
            new_d['features'] = pooled_sentence[i].tolist()
            new_data.append(new_d)
        # new_data.append(new_d)
    print(len(new_data))
    with open(f"{checkpoint_path}/{split}.json", "w") as f:
        json.dump(new_data, f, indent=4)
# da = "dev"
# data0 = load_data(f"/home/huanxuan/FiD/open_domain_data/NQ/{da}.json")
# data = load_data_compress(f"/home/huanxuan/FiD/pl/data/NQ/{da}/{da}.json")
# for d, d0 in tqdm(zip(data, data0), desc='Length'):
#     d["ctxs"] = d0["ctxs"]
# with open(f"/home/huanxuan/FiD/pl/data/NQ/{da}.json", "w") as f:
#     json.dump(data, f, indent=4)