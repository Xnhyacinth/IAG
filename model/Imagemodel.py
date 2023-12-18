# Copyright (c) 2023 Huanxuan Liao
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytorch_lightning as pl
from torch import nn
import torch
import transformers
from model.model import FiDT5, T5LoraWrapper
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftConfig, PeftModel
import torch.nn.functional as F
import math
from model.losses import *
from model.evaluation import ems
from src import util
import numpy as np


class ImageModel(nn.Module):
    def __init__(self, args, pre_train_dir):
        super().__init__()
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pre_train_dir
        )
        if 'test' not in args.name and args.load_checkpoints_path == "":
            self.encoder = model.encoder
        self.model = FiDT5(model.config)
        self.model.load_t5(model.state_dict())
        if args.lora:
            # Define LoRA Config
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            # add LoRA adaptor
            self.model = get_peft_model(self.model, lora_config)
        elif args.hylora:
            self.model = T5LoraWrapper(self.model, args.lora_rank, args.load_hypernet_weights)
        # else:
        #     self.model.set_checkpoint(args.use_checkpoint)

    def forward(self, input_ids, attention_mask, labels, features=None, **kwargs):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            features=features,
                            **kwargs)
        return output

    def generate(self, input_ids, attention_mask, max_length, features=None, **kwargs):
        model = self.model.module if hasattr(
            self.model, "module") else self.model
        return model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, features=features, **kwargs)


class ImageLitModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument("--model_name", type=str, default="t5-base")
        parser.add_argument("--model_path", type=str, default=None)
        parser.add_argument("--peft_model_id", type=str, default=None)
        parser.add_argument('--load_checkpoints_path',
                            default='', type=str)
        # parser.add_argument("--warmup_steps", type=int, default=1000)
        parser.add_argument("--warmup_ratio", type=float, default=0.08)
        # parser.add_argument("--total_steps", type=int, default=1000)
        # parser.add_argument("--scheduler_steps", type=int, default=None, help="total number of step for the scheduler, if None then scheduler_total_step = total_step")
        parser.add_argument("--dropout", type=float,
                            default=0.1, help="dropout rate")
        parser.add_argument("--lr", type=float,
                            default=1e-4, help="learning rate")
        parser.add_argument("--lr_scheduler_type", type=str,
                            default="linear", help="scheduler_type of lr")

        parser.add_argument("--weight_decay", type=float, default=0.1)

        parser.add_argument('--hylora', action='store_true',
                            help='hylora or not')
        parser.add_argument('--lora', action='store_true', help='lora or not')
        parser.add_argument("--lora_rank", type=int,
                            default=16, help='rank of lora')
        parser.add_argument("--load_hypernet_weights", type=int, default=None,
                            help='Path to hypernet weights, otherwise random init')
        parser.add_argument(
            '--do_distill', action='store_true', help='distill or not')
        parser.add_argument(
            '--select', action='store_true', help='select layers for distill or not')
        # training parameters
        # parser.add_argument("--eval_steps", type=int, default=500, help="evaluate model every <eval_freq> steps during training")
        # parser.add_argument("--save_freq", type=int, default=5000, help="save model every <save_freq> steps during training")
        # parser.add_argument("--save_strategy", type=str, default="steps", help="strategy of save model")

        # kd setting
        parser.add_argument(
            "--use_lgtm", action="store_true", help="Use LGTM or not")
        parser.add_argument("--alpha_kd", type=float,
                            default=1.0, help="The weight of kd loss")
        parser.add_argument("--temperature", type=float,
                            default=1.0, help="The temperature")

        # parser.add_argument("--do_train", action="store_true", help="Train or not")
        # parser.add_argument("--do_eval", action="store_true", help="eval or not")

        # teacher setting
        parser.add_argument("--teacher_model", type=str,
                            default="none", help="Path of teacher model")
        parser.add_argument("--train_teacher",
                            action="store_true", help="Train teacher or not")
        parser.add_argument("--t_alpha_kd", type=float, default=0.4,
                            help="The weight of kd loss if train_teacher is True")
        parser.add_argument("--t_learning_rate", type=float,
                            default=3e-5, help="The learning rate of teacher")
        parser.add_argument("--n_context", type=int, default=10)
        parser.add_argument("--use_kl", action="store_true",
                            help="Whether use kl")
        parser.add_argument("--use_attn", action="store_true",
                            help="Whether use attn")
        parser.add_argument(
            "--use_hidden", action="store_true", help="Whether use hidden")
        return parent_args

    def __init__(self, args, model_path, tokenizer):
        super().__init__()
        self.args = args
        self.num_data = 100
        self.model = ImageModel(args, model_path)
        self.tokenizer = tokenizer
        self.best_em = 0.0
        if args.do_distill:
            self.load_teacher(args)
        if 'test' not in args.name and args.load_checkpoints_path == "":
            self.encoder = self.model.encoder

    def load_teacher(self, args):
        self.t_model = FiDT5.from_pretrained(args.teacher_model)
        if args.train_teacher or args.use_lgtm:
            # Define LoRA Config
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            # add LoRA adaptor
            # self.t_model = get_peft_model(self.t_model, lora_config)
            self.t_optimizer = torch.optim.AdamW(filter(
                lambda p: p.requires_grad, self.t_model.parameters()), lr=args.t_learning_rate)
            self.t_scheduler = transformers.get_linear_schedule_with_warmup(optimizer=self.t_optimizer,
                                                                            num_warmup_steps=(
                                                                                args.warmup_ratio * self.total_step),
                                                                            num_training_steps=self.total_step)

    def setup(self, stage) -> None:
        if stage == 'fit':
            if self.args.max_steps == -1:
                self.total_step = int(self.trainer.max_epochs * self.num_data /
                                      (max(1, int(self.args.devices)) * self.trainer.accumulate_grad_batches))
            else:
                self.total_step = self.args.max_steps
            if self.args.load_checkpoints_path == "":
                del self.model.encoder
            print('Total training step:', self.total_step)

    # kd loss
    def cal_loss(self, s_logits, t_logits, temperature):
        soft_labels = F.log_softmax(
            t_logits / temperature, dim=-1, dtype=torch.float32)
        log_prob = F.log_softmax(
            s_logits / temperature, dim=-1, dtype=torch.float32)
        ori_kld_loss = (
            -torch.exp(soft_labels) * log_prob +
            torch.exp(soft_labels) * soft_labels
        )
        loss = torch.mean(torch.sum(ori_kld_loss, dim=-1))

        return loss

    def training_step(self, batch, batch_idx):
        if self.args.do_distill:
            if self.args.use_lgtm:
                # fetch the batch for calculating student's feedback
                # try:
                #     held_inputs = next(held_iter)
                # except:
                #     held_iter = iter(eval_dataloader)
                #     held_inputs = next(held_iter)
                # # update the teacher
                # model_total.step(
                #     batch,
                #     held_inputs,
                #     self.argsimizer,
                #     self.args.temperature / (1 + math.log(step // 500 + 1)),
                # )
                pass
            elif self.args.train_teacher:
                (idx, labels, _, context_ids, context_mask,
                 t_context_ids, t_context_mask, features, _) = batch
                self.model.eval()
                self.t_model.train()
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=context_ids,
                        attention_mask=context_mask,
                        labels=labels,
                        features=features,
                        output_attentions=True,
                        output_hidden_states=True
                        # return_dict=False,
                    )
                    logits = outputs[1]
                teacher_outputs = self.t_model(
                    input_ids=t_context_ids,
                    attention_mask=t_context_mask,
                    labels=labels,
                    output_attentions=True,
                    output_hidden_states=True
                    # return_dict=False,
                )
                t_loss, t_logits = teacher_outputs[0], teacher_outputs[1]
                t_loss = (self.args.t_alpha_kd * self.cal_loss(t_logits, logits, self.args.temperature / (1 + math.log(self.global_step // 500 + 1)))
                          + (1 - self.args.t_alpha_kd) * t_loss
                          )

                # update the teacher
                t_loss.backward()

                if self.global_step % self.args.accumulate_grad_batches == 0:
                    self.t_optimizer.step()
                    self.t_scheduler.step()
                    self.t_optimizer.zero_grad()
            # use teacher logits as soft labels
            self.t_model.eval()
            self.model.train()
            (idx, labels, _, context_ids, context_mask,
             t_context_ids, t_context_mask, features, _) = batch
            with torch.no_grad():
                teacher_outputs = self.t_model(
                    input_ids=t_context_ids,
                    attention_mask=t_context_mask,
                    labels=labels,
                    output_attentions=True,
                    output_hidden_states=True
                    # return_dict=False,
                )
                t_logits = teacher_outputs[1]

            # outputs = model(**batch)
            outputs = self.model(
                input_ids=context_ids,
                attention_mask=context_mask,
                labels=labels,
                features=features,
                output_attentions=True,
                output_hidden_states=True
                # return_dict=False,
            )
            loss, logits = outputs[0], outputs[1]
            loss = (
                self.args.alpha_kd * self.cal_loss(logits, t_logits, self.args.temperature / (
                    1 + math.log(self.global_step // 500 + 1)))
                + (1 - self.args.alpha_kd) * loss
            )
            # print(loss)
            if self.args.use_kl:
                loss /= 2
                p_s = F.log_softmax(logits / 4.0, dim=-1)
                p_t = F.softmax(t_logits / 4.0, dim=-1)
                loss += (
                    F.kl_div(p_s, p_t, reduction='batchmean')
                ) / 100
                # print((
                #     F.kl_div(p_s, p_t, reduction='batchmean')
                # ) / 100)

            if self.args.use_attn:
                attn = outputs[-1]
                t_attn = teacher_outputs[-1]
                if self.args.select:
                    attn = [attn[0], attn[len(attn) // 2], attn[-1]]
                    t_attn = [t_attn[0], t_attn[len(t_attn) // 2], t_attn[-1]]

                loss_a = [
                    att_mse_loss(a.repeat(1, t_a.size(0) // a.size(0), 1, 1).view(-1, a.size(1), a.size(2), a.size(3)),
                                 t_a, context_mask.view(context_mask.size(0) * context_mask.size(1), -1).repeat(t_a.size(0) // a.size(0), 1))
                    for a, t_a in zip(attn, t_attn)
                ]
                loss += sum(loss_a) / len(loss_a) * 10
                # print(sum(loss_a) / len(loss_a))

                d_attn = outputs[4]
                d_t_attn = teacher_outputs[4]
                if self.args.select:
                    d_attn = [d_attn[0], d_attn[len(d_attn) // 2], d_attn[-1]]
                    d_t_attn = [d_t_attn[0],
                                d_t_attn[len(d_t_attn) // 2], d_t_attn[-1]]
                loss_a = [
                    att_ce_loss(
                        a.repeat(1, t_a.size(0) // a.size(0), 1, 1).view(-1, a.size(1), a.size(2), a.size(3)),
                        t_a,
                        context_mask.view(
                            context_mask.size(0) * context_mask.size(1), -1
                        ).repeat(t_a.size(0) // a.size(0), 1),
                    )
                    for a, t_a in zip(d_attn, d_t_attn)
                ]
                loss += sum(loss_a) / len(loss_a) / 50
                # print(sum(loss_a) / len(loss_a))

            if self.args.use_hidden:
                hd = outputs[-2]
                t_hd = teacher_outputs[-2]
                if self.args.select:
                    hd = [hd[0], hd[len(hd) // 2], hd[-1]]
                    t_hd = [t_hd[0], t_hd[len(t_hd) // 2], t_hd[-1]]

                loss_h = [
                    cos_loss(
                        h.repeat(1, t_h.size(0) // h.size(0), 1).view(-1, h.size(1), h.size(2)),
                        t_h,
                        context_mask.view(
                            context_mask.size(0) * context_mask.size(1), -1
                        ).repeat(t_h.size(0) // h.size(0), 1),
                    )
                    for h, t_h in zip(hd, t_hd)
                ]
                loss += sum(loss_h) / len(loss_h) / 5
                # print(sum(loss_h) / len(loss_h))
                d_hd = outputs[3]
                d_t_hd = teacher_outputs[3]
                if self.args.select:
                    d_hd = [d_hd[0], d_hd[len(d_hd) // 2], d_hd[-1]]
                    d_t_hd = [d_t_hd[0], d_t_hd[len(d_t_hd) // 2], d_t_hd[-1]]
                loss_h = [
                    cos_loss(
                        h.repeat(1, t_h.size(0) // h.size(0), 1).view(-1, h.size(1), h.size(2)),
                        t_h,
                        context_mask.view(
                            context_mask.size(0) * context_mask.size(1), -1
                        ),
                    )
                    for h, t_h in zip(d_hd, d_t_hd)
                ]
                loss += sum(loss_h) / len(loss_h)  # * 10
                # print(sum(loss_h) / len(loss_h))
        else:
            (idx, labels, _, context_ids, context_mask, _, _, features, _) = batch
            loss = self.model(
                input_ids=context_ids,
                attention_mask=context_mask,
                labels=labels,
                features=features,
                # return_dict=False,
            )[0]
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/lr', self.lr_schedulers().get_last_lr()[0])
        if self.args.train_teacher:
            self.log('train/t_lr', self.t_scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        (idx, labels, _, context_ids, context_mask,
         t_context_ids, t_context_mask, features, answers) = batch
        scores, t_scores = [], []
        loss = self.model(
            input_ids=context_ids,
            attention_mask=context_mask,
            labels=labels,
            features=features,
            output_attentions=True,
            output_hidden_states=True
        )[0]
        self.log('val/loss', loss, prog_bar=True)
        outputs = self.model.generate(
            input_ids=context_ids, attention_mask=context_mask, features=features, max_length=50
        )
        # t_outputs = self.model.model.model.generate(
        #     input_ids=t_context_ids, attention_mask=t_context_mask, max_length=32
        # )
        for k, (o, gold) in enumerate(zip(outputs, answers)):
            ans = self.tokenizer.decode(o, skip_special_tokens=True)
            scores.append(ems(ans, gold))
        # if self.args.do_distill:
        #     scores_teacher, t_scores_teacher = [], []
        #     t_outputs_teacher = self.t_model.generate(
        #         input_ids=t_context_ids, attention_mask=t_context_mask, max_length=50
        #     )
        #     for k, (t_o_t, gold) in enumerate(zip(t_outputs_teacher, answers)):
        #         # ans_t = self.tokenizer.decode(o_t, skip_special_tokens=True)
        #         t_ans_t = self.tokenizer.decode(
        #             t_o_t, skip_special_tokens=True)
        #         t_scores_teacher.append(ems(t_ans_t, gold))
        #     return scores, t_scores_teacher
        return scores

    def test_step(self, batch, batch_idx):
        (idx, labels, _, context_ids, context_mask,
         t_context_ids, t_context_mask, features, answers) = batch
        scores = []
        loss = self.model(
            input_ids=context_ids,
            attention_mask=context_mask,
            labels=labels,
            features=features,
            output_attentions=True,
            output_hidden_states=True
        )[0]
        self.log('test/loss', loss)
        outputs = self.model.generate(
            input_ids=context_ids, attention_mask=context_mask, features=features, max_length=50
        )
        for k, (o, gold) in enumerate(zip(outputs, answers)):
            ans = self.tokenizer.decode(o, skip_special_tokens=True)
            scores.append(ems(ans, gold))
        return scores

    def validation_epoch_end(self, validation_step_outputs):
        # compute metrics
        # exactmatch, t_exactmatch = self.compute_metrics([_[0] for _ in validation_step_outputs]), self.compute_metrics([_[1] for _ in validation_step_outputs])
        exactmatch = [_[0] for _ in validation_step_outputs]
        em = self.compute_metrics(exactmatch)
        self.log("val/em", em*100)
        self.log("val_em", em*100)
        # self.log("long context exactmatch", t_exactmatch)
        # log_em = self.compute_metrics(exactmatch, log=True)
        log = f"Eval | {self.global_step} / {self.total_step} |"
        log += (f"evaluation: {100*em:.2f}EM |")
        # teacher
        # if self.args.do_distill:
        #     t_exactmatch_teacher = [_[1] for _ in validation_step_outputs]
        #     t_em_teacher = self.compute_metrics(t_exactmatch_teacher)
        #     # self.log("teacher exactmatch", exactmatch_teacher)
        #     self.log("val/teacher long context em", t_em_teacher)
        #     # log_t_em_teacher = self.compute_metrics(t_exactmatch_teacher, log=True)
        #     log += (f"teacher evaluation long: {100*t_em_teacher:.2f}EM |")
        log += f"lr: {self.lr_schedulers().get_last_lr()[0]:.5f}\n"
        # self.lg.info(log)
        with open(self.args.output_dir / 'logging.txt', 'a+') as f:
            f.write(log)
            f.close()

    def test_epoch_end(self, test_step_outputs):
        # compute metrics
        exactmatch = [_[0] for _ in test_step_outputs]
        em = self.compute_metrics(exactmatch)
        self.log("test/em", em*100)
        log = f"Test | "
        log += (f"evaluation: {100*em:.2f}EM |")
        with open(self.args.output_dir / 'logging.txt', 'a+') as f:
            f.write(log)
            f.close()

    def compute_metrics(self, ems, log=False):
        try:
            # ems = [e.cpu().numpy() for em in ems for e in em]
            ems = [e for em in ems for e in em]
        except:
            ems = [em for em in ems]
        if log:
            return util.weighted_average(np.mean(ems), len(ems), self.args)
        return np.mean(ems)
        # ems = [e for em in ems for e in em]
        # return util.weighted_average(np.mean(ems), len(ems), self.args)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
                [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(paras, lr=self.args.lr)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, int(self.args.warmup_ratio * self.total_step),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]
