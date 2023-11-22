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

# parser = argparse.ArgumentParser()
# parser.add_argument("--datapath", type=str, default='data')
# parser.add_argument("--dataset", type=str, default='NQ')
# parser.add_argument("--d", type=str, default='test')
# opt = parser.parse_args()
# checkpoint_path = Path(f"data/{opt.dataset}/{opt.d}")
# checkpoint_path.mkdir(parents=True, exist_ok=True)
# data = load_data_compress(f"{opt.datapath}/{opt.dataset}/{opt.d}.json")
# model_path_simcse_roberta = "princeton-nlp/sup-simcse-roberta-large"
# tokenizer = AutoTokenizer.from_pretrained(model_path_simcse_roberta)
# model = Simcsewrap(model_path_simcse_roberta, 255).cuda()
# new_data = []
# for d in tqdm(data, desc='Length'):
#     inputs = tokenizer(d['question'],
#                         max_length=256,
#                         padding='max_length',
#                         truncation='longest_first',
#                         return_tensors="pt").to("cuda")
#     with torch.no_grad():
#         embeddings = model(
#             inputs["input_ids"],
#             # inputs["token_type_ids"],
#             inputs["attention_mask"],
#             )
#     d['features'] = embeddings.cpu().numpy().tolist()
#     new_data.append(d)

da = "dev"
data0 = load_data(f"/home/huanxuan/FiD/open_domain_data/NQ/{da}.json")
data = load_data_compress(f"/home/huanxuan/FiD/pl/data/NQ/{da}/{da}.json")
for d, d0 in tqdm(zip(data, data0), desc='Length'):
    d["ctxs"] = d0["ctxs"]
with open(f"/home/huanxuan/FiD/pl/data/NQ/{da}.json", "w") as f:
    json.dump(data, f, indent=4)