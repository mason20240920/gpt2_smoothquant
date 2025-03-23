#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : ime_gpt_2
@File    : ime_gpt2_lm_head_model.py
@Author  : Barry Allen
@Date    : 2025/3/14 14:51
@Desc    :
"""
from typing import Optional, Tuple, Union

import torch
from transformers import add_start_docstrings, GenerationMixin, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING, _CHECKPOINT_FOR_DOC, \
    _CONFIG_FOR_DOC
from transformers.utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings

from models.ime_gpt2_model import IMEGPT2Model

from models.ime_gpt2_pre_trained_model import IMEGPT2PreTrainedModel
from torch import nn


@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class IMEGPT2LMHeadModel(IMEGPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = IMEGPT2Model(config=config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs) :
        return_dict = self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return (loss, lm_logits) if loss is not None else lm_logits

