# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import torch.nn.functional as F
from torch import TensorType, nn
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers.modeling_outputs import BaseModelOutput
from torchtyping import TensorType
from typing import Optional
import math

class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, features=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, features=None):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        self.main_input_name = encoder.main_input_name
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        return BaseModelOutput(
                last_hidden_state=outputs[0],
                hidden_states=outputs[1] if len(outputs) > 1 else None,
                attentions=outputs[2] if len(outputs) > 2 else None,
            )

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output

class HyperParamNet(nn.Module):
  """
  Helper class to wrap together hypernetwork weights "linear1" and "linear2" into MLP

  Arguments:
  - linear1 : Downsampling weight (encoding dim x bottleneck)
  - linear2 : Upsampling weight (bottleneck x dim)
  - dim : output dim
  - bottleneck : bottleneck dim

  Output:
  Adapter weight generated by hypernetworks with dialect feature input
  """
  def __init__(self, linear1, linear2, dim, bottleneck, scale=False):
    super().__init__()
    self.linear1 = linear1
    self.linear2 = linear2
    self.dim = dim #Output dimension
    self.bottleneck = bottleneck #MLP bottleneck
    if scale:
      self.scale = math.sqrt(dim)
    else:
      self.scale = 1

  def set_features(self, features):
    self.features = features
    return
  
  def forward(self, features):
    output = self.linear2(F.relu(self.linear1(features))).reshape(-1, self.dim, self.bottleneck)
    return output/self.scale

class HyperLora(nn.Module):
  """
  Simple MLP Hypernet
  """
  def __init__(self, linear : nn.Module, hypernet1=None, hypernet2=None, idx=0):
    super().__init__()

    self.linear = linear
    self.hypernet1 = hypernet1
    self.hypernet2 = hypernet2
    self.dropout = nn.Dropout(p=0.1)
    # Layer idx
    self.idx = idx

  def forward(self, x):
    # Conditioning variable (either indicator or example)
    val = self.hypernet1.features
    # Layer idx is added to conditioning variable
    if self.idx is not None:
      val = nn.functional.pad(val, (0, 1), value=self.idx)
    
    # Compute hypernet weights
    weight1 = self.hypernet1(val)
    weight2 = self.hypernet2(val)
    weight1 = weight1.repeat(1, x.shape[0] // weight1.shape[0], 1).view(-1, weight1.shape[1], weight1.shape[2])
    weight2 = weight2.repeat(1, x.shape[0] // weight2.shape[0], 1).view(-1, weight2.shape[1], weight2.shape[2])
    # Apply lora
    out = self.dropout(self.linear(x))
    out = torch.matmul(torch.matmul(x, weight1), weight2) + out
    return out

class HyperNet(nn.Module):
    def __init__(self, encoding_dim, input_dim, embedding_dim, output_dim):
        super(HyperNet, self).__init__()
        self.hidden_dim = 8
        self.pre_down_linear = nn.Linear(encoding_dim+1, self.hidden_dim)
        self.pre_down_linear.weight, self.pre_down_linear.bias = self.init_layer(self.pre_down_linear)
        self.down_linear = nn.Linear(self.hidden_dim, input_dim*embedding_dim)
        self.down_linear.weight, self.down_linear.bias = self.init_layer(self.down_linear)
        self.pre_up_linear = nn.Linear(encoding_dim+1, self.hidden_dim)
        self.pre_up_linear.weight, self.pre_up_linear.bias = self.init_layer(self.pre_up_linear)
        self.up_linear = nn.Linear(self.hidden_dim, embedding_dim*output_dim)
        self.up_linear.weight, self.up_linear.bias = self.init_layer(self.up_linear)

        self.down_hypernet = HyperParamNet(self.pre_down_linear, self.down_linear, input_dim, embedding_dim)
        self.up_hypernet = HyperParamNet(self.pre_up_linear, self.up_linear, embedding_dim, output_dim, scale=True)

    def init_layer(self, layer, bias=True):
        weight = nn.Parameter(torch.normal(0, 1e-7, layer.weight.shape))
        if bias:
            bias = nn.init.zeros_(layer.bias)
        else:
            bias = None
        return weight, bias

class AdapterWrapper(nn.Module):
    """
    General Wrapper Class for Hypernet Config

    Each child class needs to implement the init hypernet method that injects hypernet weights
    """
    def __init__(self, model, embedding_dim, weights):
        super(AdapterWrapper, self).__init__()
        self.model = model
        self.down_hypernet = None
        self.up_hypernet = None
        self.embedding_dim = embedding_dim
        self.encoding_dim = 255
        down_dim = model.config.d_kv * model.config.num_heads
        input_dim = model.config.d_model

        self.hypernet = HyperNet(self.encoding_dim, input_dim, self.embedding_dim, down_dim)

        if weights is not None:
            self.hypernet.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
            print("WEIGHTS LOADED")

        # self.earth_mover_loss = SamplesLoss(loss="sinkhorn", p=2)

        self.init_hypernet()

    def init_layer(self, layer):
        weight = nn.Parameter(torch.normal(0, 1e-7, layer.weight.shape))
        bias = nn.init.zeros_(layer.bias)
        return weight, bias

    def init_hypernet(self):
        pass

    def freeze_params(self):
        # All modules in the 
        # modules_to_freeze = [self.model.encoder.encoder.block[i].layer[0] for i in range(len(self.model.encoder.encoder.block))]
        modules_to_freeze = [l.module.layer[0] if hasattr(l, "module") else l.layer[0] for l in self.model.encoder.encoder.block]
        # modules_to_freeze.extend([l.module.layer[1] if hasattr(l, "module") else l.layer[1] for l in self.model.encoder.encoder.block])
        # And the decoder modules, which has both a SelfAttention (layer[0]) 
        modules_to_freeze.extend([self.model.decoder.block[i].layer[0] for i in range(len(self.model.decoder.block))])
        # and CrossAttention (layer[1]) block
        modules_to_freeze.extend([self.model.decoder.block[i].layer[1] for i in range(len(self.model.decoder.block))])
        # modules_to_freeze.extend([self.model.decoder.block[i].layer[2] for i in range(len(self.model.decoder.block))])
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False  # Actual freezing operation
        # for layer in self.model.modules():
        #     for _, param in layer.named_parameters():
        #         param.requires_grad = False

    @torch.no_grad()
    def produce_original_embeddings(
        self,
        input_ids: TensorType["batch", "seq_len"],
        attention_mask: TensorType["batch", "seq_len"],
        features: TensorType["batch", "seq_len"],
        token_type_ids: Optional[TensorType["batch", "seq_len"]] = None,
        position_ids: Optional[TensorType["batch", "seq_len"]] = None,
        head_mask: Optional[TensorType["layers", "heads"]] = None,
    ) -> TensorType["batch", "seq_len", "hidden_size"]:
        self.train(False)
        outputs = self.last_emb(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            features=features,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            include_original=False
        )

        self.train(True)
        return outputs.last_hidden_state, attention_mask

    def last_emb(self, input_ids, attention_mask, dialect_features, original_mask=None, original_embedding=None, include_original=True,**kwargs):
        """
        forward model needs to include dialect_features parameter for Trainer to not discard this feature
        """
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        if include_original:
            inputs["original_embedding"] = original_embedding
            inputs["original_mask"] = original_mask

        self.hypernet.down_hypernet.set_dialect_features(self.emb(dialect_features))
        self.hypernet.up_hypernet.set_dialect_features(self.emb(dialect_features))
        # self.hypernet.layernorm_w_hypernet.set_dialect_features(self.emb(dialect_features))
        # self.hypernet.layernorm_b_hypernet.set_dialect_features(self.emb(dialect_features))
        outputs = self.model(**inputs)
        return outputs
    
    def get_weight(self, mask):             
        probs = torch.div(mask, mask.sum(1).reshape(-1,1))                                                                                                          
        return probs
  
    def emb(self, l):
        """
        PCA Embedding of linguistic attestation vector
        """
        feature = l
        if not isinstance(l, torch.Tensor):
            feature = torch.Tensor(l)
        return feature

    def forward(self, labels, input_ids, attention_mask, features, original_mask=None, original_embedding=None, include_original=False, **kwargs):
        """
        forward model needs to include features parameter for Trainer to not discard this feature
        """
        inputs = {"labels":labels, "input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        if include_original:
            inputs["original_embedding"] = original_embedding
            inputs["original_mask"] = original_mask

        self.hypernet.down_hypernet.set_features(self.emb(features))
        self.hypernet.up_hypernet.set_features(self.emb(features))
        outputs = self.model(**inputs)

        return outputs
    
    def generate(self, input_ids, attention_mask, features=None, **kwargs):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        self.hypernet.down_hypernet.set_features(self.emb(features))
        self.hypernet.up_hypernet.set_features(self.emb(features))
        return self.model.generate(**inputs)

class T5LoraWrapper(AdapterWrapper):
    def __init__(self, model, embedding_dim, weights):
        super().__init__(model, embedding_dim, weights)
    
    def init_hypernet(self):
        for i, l in enumerate(self.model.encoder.encoder.block):
            l = l.module if hasattr(l, "module") else l
            l.layer[0].SelfAttention.q = HyperLora(l.layer[0].SelfAttention.q, self.hypernet.down_hypernet, self.hypernet.up_hypernet, 2*i)
            l.layer[0].SelfAttention.v = HyperLora(l.layer[0].SelfAttention.v, self.hypernet.down_hypernet, self.hypernet.up_hypernet, 2*i+1)

        self.freeze_params()
        self.hypernet.pre_down_linear.weight.requires_grad = True
        self.hypernet.pre_down_linear.bias.requires_grad = True
        self.hypernet.pre_up_linear.weight.requires_grad = True
        self.hypernet.pre_up_linear.bias.requires_grad = True
        self.hypernet.down_linear.weight.requires_grad = True
        self.hypernet.down_linear.bias.requires_grad = True
        self.hypernet.up_linear.weight.requires_grad = True
        self.hypernet.up_linear.bias.requires_grad = True