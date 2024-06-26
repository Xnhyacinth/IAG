from collections import defaultdict
import inspect
import time
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
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from src.util import load_data


class MInterface(pl.LightningModule):
    @staticmethod
    def piplines_args(parent_args):
        total_parser = parent_args.add_argument_group("piplines args")
        # basic parameters
        # total_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/", help="models are saved here")
        total_parser.add_argument(
            "--name", type=str, default="experiment_name", help="name of the experiment")
        total_parser.add_argument(
            "--seed", type=int, default=0, help="random seed for initialization")
        total_parser.add_argument("--use_checkpoint", action="store_true")
        total_parser.add_argument("--use_context", action="store_true")
        total_parser.add_argument("--test_fid", action="store_true")
        total_parser.add_argument(
            "--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        total_parser.add_argument(
            "--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")

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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.load_model(args, model_path)
        self.trainer = pl.Trainer.from_argparse_args(
            args,
            logger=tb_logger,
            callbacks=[self.checkpoint_callback],
            accelerator=args.accelerator,
            strategy=args.strategy,
            devices=[i for i in range(int(args.devices))],
        )

    def load_model(self, args, model_path):
        if args.load_checkpoints_path != "":
            print(f'model_path: {args.load_checkpoints_path}')
            self.model = ImageLitModel.load_from_checkpoint(
                args.load_checkpoints_path, args=args, model_path=model_path, tokenizer=self.tokenizer
            )
            print("load model from: ", args.load_checkpoints_path)
        else:
            self.model = ImageLitModel(
                args, model_path=model_path, tokenizer=self.tokenizer)

    def load_data(self, data_path):
        assert data_path
        if data_path.endswith('.jsonl'):
            data = open(data_path, 'r')
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as fin:
                data = json.load(fin)
        examples = defaultdict(list)
        for k, example in enumerate(data):
            if data_path is not None and data_path.endswith('.jsonl'):
                example = json.loads(example)
            if not 'id' in example:
                example['id'] = k
            try:
                example["context"] = example.pop("compressed_prompt")
            except:
                example["context"] = {"compressed_prompt":[0 * 256]}
            for key, val in example.items():
                examples[key].append(val)
        # egrave: is this needed?
        if data_path is not None and data_path.endswith('.jsonl'):
            data.close()

        return examples

    def compute_kernel_bias(self, vecs, n_components=256):
        """compute kernel and bias
        vecs.shape = [num_samples, embedding_size]
        transfer:y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :n_components], -mu

    def transform_and_normalize(self, vecs, kernel=None, bias=None):
        """ normalization
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    def get_features(self, sample):
        question = ['Question: Please write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Only give me the answer and do not output any other words.'
                    'Question: ' + item + '\nAnswer:' for item in sample['question']]
        inputs = self.tokenizer(question, return_tensors='pt', padding=True)
        encoder = self.model.encoder
        output = encoder(
            input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
        # shape is [batch_size, seq_len, hidden_size]
        pooled_sentence = output.last_hidden_state
        pooled_sentence = np.array(torch.mean(
            pooled_sentence, dim=1).cpu().detach().numpy())
        kernel, bias = self.compute_kernel_bias(pooled_sentence, 255)
        pooled_sentence = self.transform_and_normalize(
            pooled_sentence, kernel=kernel, bias=bias)
        sample['features'] = pooled_sentence
        return sample

    def load_features(self, dataset, hg_datapath):
        if 'NQ' in hg_datapath:
            data_path = 'features/NQ'
        elif 'WQ' in hg_datapath:
            data_path = 'features/WQ'
        else:
            data_path = 'features/TQA'
        if self.args.use_context:
            data_path = f"{data_path}/context-{self.args.n_c}"
        else:
            data_path = f"{data_path}/context-0"
        print(f"load data_features from {data_path}")
        dataset_features = load_dataset("json", data_files={
                                        'train': f'{data_path}/train.json', 'eval': f'{data_path}/eval.json', 'test': f'{data_path}/test.json'})
        if 'test' not in self.args.name:
            for split in ['train', 'eval', 'test']:
                if split in dataset:
                    dataset[split] = dataset[split].add_column(column=dataset_features[split]['features'], name='features')
        else:
            dataset = dataset.add_column(column=dataset_features['test']['features'], name='features')
        return dataset

    def train(self):
        if self.args.n_c != 0:
            dataset = load_data(self.args, self.args.hg_datapath)
        else:
            train_data = Dataset.from_dict(self.load_data(self.args.train_data))
            dev_data = Dataset.from_dict(self.load_data(self.args.eval_data))
            test_data = Dataset.from_dict(self.load_data(self.args.test_data))
            # dataset = load_dataset("json", data_files={'train':self.args.train_data, 'eval':self.args.eval_data, 'test':self.args.test_data})
            dataset = DatasetDict({'train': train_data, 'eval': dev_data, 'test': test_data})
        dataset = self.load_features(dataset, self.args.hg_datapath)
        print(dataset)
        self.data_model = ImageDataModel(
            self.tokenizer, self.args, dataset=dataset)
        self.model.num_data = len(dataset['train'])
        if self.args.load_checkpoints_path == "":
            del self.model.encoder
        self.trainer.fit(self.model, self.data_model)

    def test(self, model=None, data=None):
        if data is not None:
            dataset = load_data(self.args, f'{data}/test')
            dataset = self.load_features(dataset, data)
            print(dataset)
            self.data_model = ImageDataModel(
                self.tokenizer, self.args, dataset=dataset)
        start_time = time.time()
        if model is not None:
            self.trainer.test(model, self.data_model)
        self.trainer.test(self.model, self.data_model)
        with open(self.args.output_dir / 'logging.txt', 'a+') as f:
            f.write(f'Time: {time.time() - start_time:.1f}s & {(time.time() - start_time)//60}min {(time.time() - start_time)%60:.1f}s\n')
            f.close()

    def save(self, save_name):
        if self.args.hylora:
            torch.save(self.trainer.model.model.model.hypernet.state_dict(
            ), save_name + '/hypernet.pth')
        elif self.args.lora:
            self.trainer.model.model.model.save_pretrained(save_name)
            self.tokenizer.save_pretrained(save_name)
        # if you want to save the base model to call
        # trainer.model.base_model.save_pretrained(peft_model_id)
    
    def load_from_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            return self.model.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path, args=self.args, tokenizer=self.tokenizer)
        return self.model.load_from_checkpoint(checkpoint_path)


class UniversalCheckpoint(ModelCheckpoint):
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group(
            "universal checkpoint callback")

        parser.add_argument("--monitor", default="val_em", type=str)
        parser.add_argument("--mode", default="max", type=str)
        parser.add_argument("--save_ckpt_path", default="./ckpt/", type=str)
        parser.add_argument("--load_ckpt_path", default="./ckpt/", type=str)
        parser.add_argument(
            "--filename", default="{epoch}-{step}-{val_em:.2f}", type=str)
        parser.add_argument("--save_last", action="store_true", default=True)
        parser.add_argument("--save_top_k", default=-1, type=float)
        parser.add_argument("--every_n_train_steps", default=None, type=float)
        parser.add_argument("--save_weights_only",
                            action="store_true", default=False)
        parser.add_argument("--every_n_epochs", default=None, type=int)
        parser.add_argument("--save_on_train_epoch_end",
                            action="store_true", default=None)
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
