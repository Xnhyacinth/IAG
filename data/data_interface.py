# Copyright 2023 Huanxuan Liao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import random
import torch

class ImageDataset(Dataset):
    def __init__(self,
        data,
        n_context=None,
        question_prefix='question:',
        title_prefix='title:',
        passage_prefix='context:'):
        super().__init__()

        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()
        
    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' '
        elif 'answers' in example:
            return random.choice(example['answers']) + ' '
        else:
            return None
    
    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context != 0:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages = None

        return {
            'index': index,
            'question': question,
            'target': target,
            'passages': passages,
            'context': example['context']['compressed_prompt'],
            'features': example['features'],
            'answers': example['answers']
        }
        
    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

class ImageDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('ImageDataModel')
        parser.add_argument('--num_workers', default=10, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument("--gold", action="store_true", help="Whether use gold passage")
        parser.add_argument("--cbqa", action="store_true", help="Whether use none passage")
        parser.add_argument("--train_data", type=str, default=None, help="path of train data")
        parser.add_argument("--eval_data", type=str, default=None, help="path of eval data")
        parser.add_argument("--test_data", type=str, default=None, help="path of test data")
        parser.add_argument("--hg_datapath", type=str, default=None, help="path of huggingface dataset")
        parser.add_argument("--n_c", type=int, default=100, help="number of passages in compress")
        parser.add_argument("--answer_maxlength", type=int, default=200, help="maximum number of tokens in passages")
        parser.add_argument("--context_maxlength", type=int, default=200, help="maximum number of tokens in context")
        parser.add_argument("--text_maxlength", type=int, default=200, help="maximum number of tokens used to train the model, no truncation if -1")

        return parent_args

    def __init__(self, tokenizer, args, train_data=None, val_data=None, test_data=None, dataset=None):
        super().__init__()
        self.batchsize = args.batch_size
        self.args = args
        self.tokenizer = tokenizer
        if dataset is None:
            self.train_data = ImageDataset(train_data, args.n_context)
            self.valid_data = ImageDataset(val_data, args.n_context)
            self.test_data = ImageDataset(test_data, args.n_context)
        else:
            self.train_data = ImageDataset(dataset['train'], args.n_context)
            self.valid_data = ImageDataset(dataset['eval'], args.n_context)
            self.test_data = ImageDataset(dataset['test'], args.n_context)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=self.collate_fn, batch_size=self.batchsize,
                          pin_memory=False,
                          num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=self.collate_fn, batch_size=self.batchsize,
                          pin_memory=False,
                          num_workers=self.args.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, collate_fn=self.collate_fn, batch_size=self.batchsize,
                          pin_memory=False,
                          num_workers=self.args.num_workers)

    def collate_fn(self, batch):
        '''
        Aggregate a batch data.
        '''
        assert (batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.args.answer_maxlength if self.args.answer_maxlength > 0 else None,
            # pad_to_max_length=True,
            padding='max_length',
            return_tensors='pt',
            truncation='longest_first'
            # truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example, gold=False, cbqa=False):
            if example['passages'] is None or cbqa:
                return [example['question']]
            if gold:
                return [example['question'] + " " + t for t in example['passages'][:1]]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]

        def encode_passages(batch_text_passages, tokenizer, max_length):
            passage_ids, passage_masks = [], []
            for k, text_passages in enumerate(batch_text_passages):
                p = tokenizer.batch_encode_plus(
                    text_passages,
                    max_length=max_length,
                    # pad_to_max_length=True,
                    padding='max_length',
                    return_tensors='pt',
                    truncation='longest_first'
                )
                passage_ids.append(p['input_ids'][None])
                passage_masks.append(p['attention_mask'][None])

            passage_ids = torch.cat(passage_ids, dim=0)
            passage_masks = torch.cat(passage_masks, dim=0)
            return passage_ids, passage_masks.bool()
        
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.args.text_maxlength)
        if self.args.cbqa or self.args.gold:
            context = [append_question(example, self.args.gold, self.args.cbqa) for example in batch]
        else:
            context = [[ex['question'] + "\n" + ex['context']] for ex in batch]
        context_ids, context_masks = encode_passages(context,
                                                     self.tokenizer,
                                                     self.args.context_maxlength)
        # print(context[0])
        # print(self.tokenizer.decode(context_ids[0]))
        answers = [ex['answers'] for ex in batch]
        features = torch.Tensor([ex['features'] for ex in batch])
        return (index, target_ids, target_mask, context_ids, context_masks, passage_ids, passage_masks, features, answers)