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

import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import transformers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning import trainer, loggers
from model.Imagemodel import ImageLitModel
from data.data_interface import ImageDataModel
import json
import numpy as np
from datasets import load_dataset


class MInterface(pl.LightningModule):
    @staticmethod
    def piplines_args(parent_args):
        total_parser = parent_args.add_argument_group("piplines args")
        # basic parameters
        # total_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/", help="models are saved here")
        total_parser.add_argument("--name", type=str, default="experiment_name", help="name of the experiment")
        # total_parser.add_argument(
        #     "--bf16",
        #     type=bool,
        #     default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        #     help="Whether to use bf16.",
        # )
        # total_parser.add_argument("--fp16", action="store_true", help="T5 overflows with fp16")
        # add training hyperparameters for epochs, batch size, learning rate, and seed
        # total_parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs to train for.")
        # total_parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size to use for training.")
        # total_parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size to use for testing.")
        # total_parser.add_argument("--generation_max_length", type=int, default=32, help="Maximum length to use for generation")
        # total_parser.add_argument("--generation_num_beams", type=int, default=2, help="Number of beams to use for generation.")
        # total_parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of accumulation_steps to use for generation.")
        total_parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
        # deepspeed
        # total_parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
        # logging & evaluation strategies
        # total_parser.add_argument("--logging_dir", type=str, default=None, help="Path to deepspeed config file.")
        # total_parser.add_argument("--logging_strategy", type=str, default="steps", help="strategy of logging")
        # total_parser.add_argument("--logging_steps", type=int, default=500, help="Steps of logging.")

        # total_parser.add_argument("--load_best_model_at_end", action="store_true", default=True)
        total_parser.add_argument("--use_checkpoint", action="store_true")
        total_parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        total_parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")
        # total_parser.add_argument("--write_results", action="store_true", help="save results")
        # total_parser.add_argument("--write_crossattention_scores", action="store_true", help="save dataset with cross-attention scores")
        
        total_parser = ImageDataModel.add_data_specific_args(total_parser)
        total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
        total_parser = ImageLitModel.add_model_specific_args(total_parser)
        total_parser = pl.Trainer.add_argparse_args(parent_args)
        return parent_args

    def __init__(self, args, model_path):
        super().__init__()
        self.args = args
        # self.save_hyperparameters(args)
        # self.lg = logger
        self.checkpoint_callback = UniversalCheckpoint(args)
        tb_logger = loggers.TensorBoardLogger(save_dir=args.logging_dir)
        self.trainer = pl.Trainer.from_argparse_args(
            args,
            logger=tb_logger,
            callbacks=[self.checkpoint_callback],
            accelerator=args.accelerator,
            strategy=args.strategy,
            devices=[i for i in range(int(args.devices))],
        )
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.load_model(args, model_path)

    def load_model(self, args, model_path):
        if args.load_checkpoints_path != "":
            self.model = ImageLitModel.load_from_checkpoint(
                args.load_checkpoints_path, args=args, model_path=model_path
            )
            print("load model from: ", args.load_checkpoints_path)
        else:
            self.model = ImageLitModel(args, model_path=model_path, tokenizer=self.tokenizer)

    def load_data(self, data_path):
        assert data_path
        if data_path.endswith('.jsonl'):
            data = open(data_path, 'r')
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as fin:
                data = json.load(fin)
        examples = []
        for k, example in enumerate(data):
            if data_path is not None and data_path.endswith('.jsonl'):
                example = json.loads(example)
            if not 'id' in example:
                example['id'] = k
            examples.append(example)
        ## egrave: is this needed?
        if data_path is not None and data_path.endswith('.jsonl'):
            data.close()

        return examples
    
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

    
    def get_features(self, sample):
        question = ['Question: Please write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Only give me the answer and do not output any other words.'\
                    'Question: ' + item + '\nAnswer:' for item in sample['question']]
        inputs = self.tokenizer(question, return_tensors='pt', padding=True)
        encoder = self.model.model.model.encoder if hasattr(self.model.model, 'model') else self.model.model.encoder
        output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
        pooled_sentence = output.last_hidden_state.cpu() # shape is [batch_size, seq_len, hidden_size]
        pooled_sentence = np.array(pooled_sentence)    
        kernel,bias = self.compute_kernel_bias(pooled_sentence,255)
        pooled_sentence = self.transform_and_normalize(pooled_sentence, kernel=kernel, bias=bias)
        sample['features'] = pooled_sentence
        print(pooled_sentence.shape())
        return sample
    
    def train(self):
        if self.args.hg_datapath is not None:
            dataset = load_dataset('hg_datapath', f"ctxs{self.args.n_c}")
        else:
            train_data = self.load_data(self.args.train_data)
            dev_data = self.load_data(self.args.eval_data)
            test_data = self.load_data(self.args.test_data)
            self.data_model = ImageDataModel(
                self.tokenizer, self.args, train_data, dev_data, test_data)
            # dataset = load_dataset("json", data_files={'train':self.args.train_data, 'eval':self.args.eval_data, 'test':self.args.test_data})
            # dataset = dataset.map(self.get_features, batched=True, desc="Features for Input")
            # self.data_model = ImageDataModel(
            #     dataset['train'], dataset['eval'], dataset['test'], self.tokenizer, self.args)
            self.model.num_data = len(train_data)
        
        self.trainer.fit(self.model, self.data_model)
    
    def test(self, data=None):
        if data is not None:
            test_data = self.load_data(self.args.test_data)
            self.data_model = ImageDataModel(self.tokenizer, self.args, test_data=test_data)
        self.trainer.test(self.model, self.data_model)
    
    def save(self, save_name):
        if self.args.hylora:
            torch.save(self.trainer.model.model.model.hypernet.state_dict(), save_name + 'hypernet.pth')
        elif self.args.lora:
            self.trainer.model.model.model.save_pretrained(save_name)
            self.tokenizer.save_pretrained(save_name)
        # if you want to save the base model to call
        # trainer.model.base_model.save_pretrained(peft_model_id)
        
    
    
class UniversalCheckpoint(ModelCheckpoint):
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group("universal checkpoint callback")

        parser.add_argument("--monitor", default="val_em", type=str)
        parser.add_argument("--mode", default="max", type=str)
        parser.add_argument("--save_ckpt_path", default="./ckpt/", type=str)
        parser.add_argument("--load_ckpt_path", default="./ckpt/", type=str)
        parser.add_argument("--filename", default="{epoch}-{step}-{val_em:.2f}", type=str)
        parser.add_argument("--save_last", action="store_true", default=True)
        parser.add_argument("--save_top_k", default=-1, type=float)
        parser.add_argument("--every_n_train_steps", default=None, type=float)
        parser.add_argument("--save_weights_only", action="store_true", default=False)
        parser.add_argument("--every_n_epochs", default=None, type=int)
        parser.add_argument("--save_on_train_epoch_end", action="store_true", default=None)
        return parent_args

    def __init__(self, args):
        super().__init__(
            monitor=args.monitor,
            save_top_k=args.save_top_k,
            mode=args.mode,
            every_n_train_steps=args.every_n_train_steps,
            save_weights_only=args.save_weights_only,
            dirpath=args.save_ckpt_path,
            filename=args.filename,
            save_last=args.save_last,
            every_n_epochs=args.every_n_epochs,
            save_on_train_epoch_end=args.save_on_train_epoch_end,
        )
