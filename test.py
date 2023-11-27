# Copyright 2023 Huanxuan Liao
# Contact: huanxuanliao@gmail.com

import pytorch_lightning as pl
from argparse import ArgumentParser
from src import util, slurm
from model import MInterface
from pathlib import Path
from copy import deepcopy
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CURL_CA_BUNDLE'] = ''

def main(opt):
    # set seed
    pl.seed_everything(opt.seed)
    
    checkpoint_path = Path(opt.default_root_dir) / opt.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    opt.output_dir = checkpoint_path
    opt.logging_dir = checkpoint_path / 'logs'
    opt.save_ckpt_path = checkpoint_path / 'ckpt'
    opt.load_ckpt_path = checkpoint_path / 'ckpt'
      
    with open(checkpoint_path / 'options.txt', 'w') as o:
        for k, v in sorted(opt.__dict__.items(), key=lambda x: x[0]):
            o.write(f'{k} = {v}\n')
            
    model = MInterface(opt, opt.model_name)
    # test
    model.test(data=opt.hg_datapath)

if __name__ == "__main__":
    total_parser = ArgumentParser("Image")
    total_parser = MInterface.piplines_args(total_parser)
    args = total_parser.parse_args()
    
    main(args)
