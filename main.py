import pytorch_lightning as pl
from argparse import ArgumentParser
from src import util, slurm
from model import MInterface
from pathlib import Path
from copy import deepcopy
import torch
import os
import numpy as np
import random
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CURL_CA_BUNDLE'] = ''

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main(opt):
    # set seed
    pl.seed_everything(opt.seed)
    # seed_everything(opt.seed)
    
    checkpoint_path = Path(opt.default_root_dir) / opt.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    opt.output_dir = checkpoint_path
    opt.logging_dir = checkpoint_path / 'logs'
    opt.save_ckpt_path = checkpoint_path / 'ckpt'
    opt.load_ckpt_path = checkpoint_path / 'ckpt'
    if os.path.exists(opt.save_ckpt_path / 'last.ckpt'):
        opt.load_checkpoints_path = opt.save_ckpt_path / 'last.ckpt'
    with open(checkpoint_path / 'options.txt', 'w') as o:
        for k, v in sorted(opt.__dict__.items(), key=lambda x: x[0]):
            o.write(f'{k} = {v}\n')
            
    model = MInterface(opt, opt.model_name)
    # train        
    model.train()
    # tuner = pl.tuner.tuning.Tuner(deepcopy(model.trainer))
    # # new_batch_size = tuner.scale_batch_size(model.model, datamodule=model.data_model, init_val=torch.cuda.device_count())
    # new_lr = tuner.lr_find(model.model, datamodule=model.data_model)
    # del tuner
    # import gc
    # gc.collect()
    # # model.hparams.batch_size = new_batch_size
    # # print(new_batch_size)
    # model.hparams.lr = new_lr
    # print(new_lr)
    # test
    model.trainer.checkpoint_callback.best_model_path
    model.test()
    # Save our LoRA model & tokenizer results
    # model_id=f"{opt.save_ckpt_path}/results"
    # model.save(model_id)

if __name__ == "__main__":
    total_parser = ArgumentParser("Image")
    total_parser = MInterface.piplines_args(total_parser)
    args = total_parser.parse_args()
    
    main(args)
