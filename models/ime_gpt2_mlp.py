#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : ime_gpt_2
@File    : ime_gpt2_mlp.py
@Author  : Barry Allen
@Date    : 2025/3/13 21:32
@Desc    :
"""
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from transformers import GPT2Config
from transformers.activations import ACT2FN


class IMEGPT2MLP(nn.Module):
    c_fc: nn.Linear

    c_proj: nn.Linear

    act: ACT2FN

    dropout: nn.Dropout

    def __init__(self,
                 config: GPT2Config,
                 intermediate_size: int):
        """
        初始化MLP层
        :param config:
        :param intermediate_size:
        """
        super().__init__()
        embed_dim: int = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self,
                hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        """
        前向传播
        :param hidden_states:
        :return:
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states