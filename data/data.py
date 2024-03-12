import torch
import random
import json
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        n_context=None,
        t_n_context=None,
        question_prefix='question:',
        title_prefix='title:',
        passage_prefix='context:',
    ):
        self.data = data
        self.n_context = n_context
        self.t_n_context = t_n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            # return target + ' </s>'
            return target + ' '
        elif 'answers' in example:
            return random.choice(example['answers']) + ' '
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)
        f = self.title_prefix + " {} " + self.passage_prefix + " {}"
        if 'ctxs' in example and self.n_context !=0 :
            contexts = example['ctxs'][: self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
            
        else:
            passages, scores = None, None
        
        if self.t_n_context is not None:
                t_contexts = example['ctxs'][: self.t_n_context]
                t_passages = [f.format(c['title'], c['text']) for c in t_contexts]
                t_scores = [float(c['score']) for c in t_contexts]
                t_scores = torch.tensor(t_scores)
                if len(t_contexts) == 0:
                    t_contexts = [question]
        else:
            t_passages, t_scores = None, None

        return {
            'index': index,
            'question': question,
            'target': target,
            'passages': passages,
            'scores': scores,
            't_passages': t_passages,
            't_scores': t_scores,
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation='longest_first'
            # truncation=True,
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert batch[0]['target'] != None
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='max_length',
            return_tensors='pt',
            truncation='longest_first'
            # truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example, key):
            if example[key] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example[key]]

        text_passages = [append_question(example, 'passages') for example in batch]
        passage_ids, passage_masks = encode_passages(
            text_passages,
            self.tokenizer,
            self.text_maxlength,
        )
        t_text_passages = [append_question(example, 't_passages') for example in batch]
        t_passage_ids, t_passage_masks = encode_passages(
            t_text_passages,
            self.tokenizer,
            self.text_maxlength,
        )
        qs = [example['question'] for example in batch]
        q_ids, q_masks = encode_passages(
            [qs],
            self.tokenizer,
            235
        )
        # dialect_feats = []

        # for f in batch:
        #     # if f["dialect_features"] is None:
        #     f["dialect_features"] = [0]*236
        #     dialect_feats.append(torch.Tensor(f["dialect_features"][1:]))
        #     del f["dialect_features"]
        # print(text_passages)
        # print(t_text_passages[0])
        return (
            index,
            target_ids,
            target_mask,
            passage_ids,
            passage_masks,
            t_passage_ids,
            t_passage_masks,
            q_ids[0].to(torch.float32)
            # dialect_feats[0]
        )

class keyword_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        n_context=None,
        t_n_context=None,
        question_prefix='question:',
        title_prefix='title:',
        passage_prefix='context:',
    ):
        self.data = data
        self.n_context = n_context
        self.t_n_context = t_n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            # return target + ' </s>'
            return target + ' '
        elif 'answers' in example:
            return random.choice(example['answers']) + ' '
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)
        f = self.title_prefix + " {} " + self.passage_prefix + " {}"
        keywords = example['keywords']

        return {
            'index': index,
            'question': question,
            'target': target,
            'keywords': [keywords],
        }

    def get_example(self, index):
        return self.data[index]


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation='longest_first'
            # truncation=True,
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class keyword_Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert batch[0]['target'] != None
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='max_length',
            return_tensors='pt',
            truncation='longest_first'
            # truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        # text_passages = [[example['keywords']] for example in batch]
        def append_question(example):
            if example['keywords'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['keywords']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(
            text_passages,
            self.tokenizer,
            self.text_maxlength,
        )
        
        # print(text_passages)
        # print(t_text_passages[0])
        return (
            index,
            target_ids,
            target_mask,
            passage_ids,
            passage_masks,
            None,
            None
        )

def load_data_keywords(data_path=None, global_rank=-1, world_size=-1):
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


def load_data(data_path=None, global_rank=-1, world_size=-1):
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
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples


class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            padding='max_length',
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True,
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages, self.tokenizer, self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, title_prefix='title:', passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = (
            self.title_prefix
            + " "
            + example[2]
            + " "
            + self.passage_prefix
            + " "
            + example[1]
        )
        return example[0], text


class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            padding='max_length',
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True,
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
