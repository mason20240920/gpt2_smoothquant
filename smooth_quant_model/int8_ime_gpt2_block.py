#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : int8_ime_gpt2_block.py
@Author  : Barry Allen
@Date    : 2025/3/21 19:05
@Desc    : GPT-2 Attention和MLP的Block
"""
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import GPT2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from models.ime_gpt2_block import IMEGPT2Block
from smooth_quant_model.int8_ime_gpt2_attention import Int8IMEGPT2Attention
from smooth_quant_model.int8_ime_gpt2_mlp import Int8IMEGPT2MLP


class Int8IMEGPT2Block(nn.Module):
    """
    Int8量化版本的GPT-2模块
    """

    attn: Int8IMEGPT2Attention

    mlp: Int8IMEGPT2MLP

    def __init__(self,
                 config: GPT2Config):
        super(Int8IMEGPT2Block, self).__init__()
        hidden_size: int = config.hidden_size

        self.ln_1: Qwen2RMSNorm = Qwen2RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2: Qwen2RMSNorm = Qwen2RMSNorm(hidden_size, eps=config.layer_norm_epsilon)

    @staticmethod
    def from_float(
            module: IMEGPT2Block,
            config: GPT2Config,
            intermediate_size: int,
            attn_input_scale: float,
            attn_output_scale: float,
            out_input_scale: float,
            q_output_scale: float,
            k_output_scale: float,
            v_output_scale: float,
            attn_weight_input_scale: float,
            attn_weight_input_zp: torch.int8,
            q_output_zp: torch.int8,
            k_output_zp: torch.int8,
            v_output_zp: torch.int8,
            attn_input_zp: torch.int8,
            attn_output_zp: torch.int8,
            out_input_zp: torch.int8,
            fc1_input_scale: float,
            fc1_input_zp: torch.int8,
            fc1_output_scale: float,
            fc1_output_zp: torch.int8,
            fc2_input_scale: float,
            fc2_input_zp: torch.int8,
            fc2_output_scale: float,
            fc2_output_zp: torch.int8):
        int8_module: Int8IMEGPT2Block = Int8IMEGPT2Block(config=config)
        int8_module.ln_1 = module.ln_1
        int8_module.ln_2 = module.ln_2
        # int8_module.attn = module.attn
        int8_module.attn = Int8IMEGPT2Attention.from_float(
            module=module.attn,
            input_scale=attn_input_scale,
            output_scale=attn_output_scale,
            out_input_scale=out_input_scale,
            q_input_scale=q_output_scale,
            k_input_scale=k_output_scale,
            v_input_scale=v_output_scale,
            attn_weight_input_scale=attn_weight_input_scale,
            attn_weight_input_zp=attn_weight_input_zp,
            q_input_zp=q_output_zp,
            k_input_zp=k_output_zp,
            v_input_zp=v_output_zp,
            input_zp=attn_input_zp,
            output_zp=attn_output_zp,
            out_input_zp=out_input_zp,
        )
        int8_module.mlp = Int8IMEGPT2MLP.from_float(
            module=module.mlp,
            config=config,
            intermediate_size=intermediate_size,
            c_fc_input_scale=fc1_input_scale,
            c_fc_input_zp=fc1_input_zp,
            c_gelu_input_scale=fc1_output_scale,
            c_gelu_input_zp=fc1_output_zp,
            c_gelu_output_scale=fc2_input_scale,
            c_gelu_output_zp=fc2_input_zp,
            c_proj_output_scale=fc2_output_scale,
            c_proj_output_zp=fc2_output_zp,
        )
        return int8_module

    def forward(self,
                hidden_states: Optional[Tuple[torch.FloatTensor]]):
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

        return hidden_states  # hidden_states
