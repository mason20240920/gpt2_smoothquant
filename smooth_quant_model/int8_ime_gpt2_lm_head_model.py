#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : int8_ime_gpt2_lm_head_model.py
@Author  : Barry Allen
@Date    : 2025/3/21 19:37
@Desc    : GPT-2 头的transformer
"""
from typing import Optional

import torch
from transformers import add_start_docstrings, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING, _CHECKPOINT_FOR_DOC, \
    _CONFIG_FOR_DOC
from transformers.utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings

from models.ime_gpt2_lm_head_model import IMEGPT2LMHeadModel
from models.ime_gpt2_pre_trained_model import IMEGPT2PreTrainedModel
from smooth_quant_model.int8_ime_gpt2_model import Int8ImeGPT2Model

from torch import nn


@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class Int8ImeGPT2LMHeadModel(IMEGPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = Int8ImeGPT2Model(config=config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module: IMEGPT2LMHeadModel,
                   config: GPT2Config,
                   intermediate_size: int,
                   decoder_layer_scales: dict):
        int8_module = Int8ImeGPT2LMHeadModel(config=config)
        int8_module.transformer = Int8ImeGPT2Model.from_float(module.transformer,
                                                              config=config,
                                                              intermediate_size=intermediate_size,
                                                              decoder_layer_scales=decoder_layer_scales)
        int8_module.lm_head = module.lm_head
        return int8_module

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                **kwargs):
        transformer_outputs = self.transformer(
            input_ids
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        return lm_logits