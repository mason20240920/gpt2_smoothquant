#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : ime_gpt_2
@File    : ime_gpt2_block.py
@Author  : Barry Allen
@Date    : 2025/3/13 21:40
@Desc    : 输入法GPT-2块
"""
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import GPT2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from models.ime_gpt_2_attention import IMEGPT2Attention
from models.ime_gpt2_mlp import IMEGPT2MLP


class IMEGPT2Block(nn.Module):
    """
    IME版本的GPT-2块
    """
    hidden_size: int  # 隐藏层大小

    inner_dim: int  # 内部维度

    ln_1: Qwen2RMSNorm # 第一层归一化

    attn: IMEGPT2Attention # 注意力机制

    ln_2: Qwen2RMSNorm # 第二层归一化

    mlp: IMEGPT2MLP # 全连接层


    def __init__(self,
                 config: GPT2Config,
                 layer_idx: int):
        """
        初始化GPT-2块
        :param config:
        """
        super(IMEGPT2Block, self).__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else hidden_size * 4

        self.ln_1 = Qwen2RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = IMEGPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = Qwen2RMSNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = IMEGPT2MLP(intermediate_size=inner_dim, config=config)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]):
        """
        前向传播
        :param hidden_states: 前向传播
        :return:
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        # 归一化
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 归一化
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states # hidden_states

