#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : int8_ime_gpt2_attention.py
@Author  : Barry Allen
@Date    : 2025/3/21 16:39
@Desc    : INT8的IME-GPT2-Attention
"""
from typing import Tuple, Optional

import torch
from torch import nn
from transformers import GPT2Config

from models.ime_gpt_2_attention import IMEGPT2Attention
from quant_ops.quant_linear import W8A8B32O8LinearForMac, W8A8BFP32OFP32LinearForMac
from quant_ops.quant_bmm import BMMS8TS8NF32T, BMMS8TS8NS8T
from supports.quant_base_method import quant_per_tensor_asymmetric_int8_with_val, quantize_tensor, dequantize_tensor


class Int8IMEGPT2Attention(nn.Module):


    input_scale: float  # 输入Tensor的scale

    input_zero_point: torch.int8 # 输入的zero point

    q_output_scale: float

    q_output_zero_point: torch.int8

    k_output_scale: float

    k_output_zero_point: torch.int8

    v_output_scale: float

    v_output_zero_point: torch.int8

    proj_input_scale: float

    proj_input_zero_point: torch.int8

    attn_weights_scale: float

    attn_weights_zero_point: torch.int8

    def __init__(self,
                 config: GPT2Config):
        super(Int8IMEGPT2Attention, self).__init__()
        self.config = config
        max_positions: int = config.max_position_embeddings

        self.embed_dim: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.embed_dim // self.num_heads
        self.split_size: int = self.embed_dim

        # 输入注意力
        self.c_attn: W8A8B32O8LinearForMac = W8A8B32O8LinearForMac(in_features=self.embed_dim,
                                                                   out_features=3 * self.embed_dim)
        self.c_proj: W8A8BFP32OFP32LinearForMac = W8A8BFP32OFP32LinearForMac(in_features=self.embed_dim,
                                                                             out_features=self.embed_dim)

        self.qk_bmm: BMMS8TS8NF32T = BMMS8TS8NF32T(alpha=1.0,
                                                   tensor_a_zp=1.0,
                                                   tensor_b_zp=1.0,)
        self.pv_bmm: BMMS8TS8NS8T = BMMS8TS8NS8T(alpha=1.0,
                                                 tensor_a_zp=1.0,
                                                 tensor_b_zp=1.0,
                                                 tensor_o_zp=1.0,
                                                 tensor_o_alpha=1.0)

        # 不需要梯度，不会被优化器更新 bias
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 不需要梯度, 不会被优化器更新
        self.register_buffer("masked_bias", torch.tensor(0.0), persistent=False)


    @staticmethod
    @torch.no_grad()
    def from_float(
            module: IMEGPT2Attention,
            input_scale: float,
            output_scale: float,
            out_input_scale: float,
            q_input_scale: float,
            k_input_scale: float,
            v_input_scale: float,
            attn_weight_input_scale: float,
            attn_weight_input_zp: torch.int8,
            q_input_zp: torch.int8,
            k_input_zp: torch.int8,
            v_input_zp: torch.int8,
            input_zp: torch.int8,
            output_zp: torch.int8,
            out_input_zp: torch.int8):
        """
        生成GPT-2的结果
        :param attn_weight_input_zp:
        :param attn_weight_input_scale:
        :param v_input_zp:
        :param v_input_scale:
        :param k_input_zp:
        :param k_input_scale:
        :param q_input_zp: QK输入的0点
        :param q_input_scale: QK输入的scale
        :param module:
        :param input_scale:
        :param output_scale:
        :param out_input_scale:
        :param input_zp:
        :param output_zp:
        :param out_input_zp:
        :return:
        """
        int8_module: Int8IMEGPT2Attention = Int8IMEGPT2Attention(config=module.config)
        int8_module.input_zero_point = input_zp
        int8_module.input_scale = input_scale
        int8_module.q_output_scale = q_input_scale
        int8_module.k_output_scale = k_input_scale
        int8_module.v_output_scale = v_input_scale
        int8_module.q_output_zero_point = q_input_zp
        int8_module.k_output_zero_point = k_input_zp
        int8_module.v_output_zero_point = v_input_zp
        int8_module.proj_input_scale = out_input_scale
        int8_module.proj_input_zero_point = out_input_zp
        int8_module.attn_weights_scale = attn_weight_input_scale
        int8_module.attn_weights_zero_point = attn_weight_input_zp
        # Fuse the scaling into the q_proj output scale
        int8_module.c_attn = W8A8B32O8LinearForMac.from_float(
            module=module.c_attn,
            input_scale=input_scale,
            output_scale=output_scale,
            input_zp=input_zp,
            output_zp=output_zp
        )
        int8_module.c_proj = W8A8BFP32OFP32LinearForMac.from_float(
            module=module.c_proj,
            input_scale=out_input_scale,
            input_zp=out_input_zp,
        )
        int8_module.qk_bmm = BMMS8TS8NF32T.from_a_and_b(
            a_scale=q_input_scale,
            b_scale=k_input_scale,
            a_zp=q_input_zp,
            b_zp=k_input_zp,
        )
        int8_module.pv_bmm = BMMS8TS8NS8T.from_a_and_b(
            a_scale=attn_weight_input_scale,
            b_scale=v_input_scale,
            a_zp=attn_weight_input_zp,
            b_zp=v_input_zp,
            o_zp=out_input_zp,
            o_scale=out_input_scale,
        )
        return int8_module

    @torch.no_grad()
    def forward(self,
                hidden_states:Optional[Tuple[torch.Tensor]]):
        """
        前向推理
        :param hidden_states:
        :return:
        """
        # 量化的hidden size
        hidden_states = quantize_tensor(input_tensor=hidden_states,
                                       scale=self.input_scale,
                                       zero_point=self.input_zero_point)

        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)  # [batch_size, seq_len, num_heads, head_dim]
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)  # [batch_size, seq_len, num_heads, head_dim]

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        # ============================QKV的Self-Attention计算============================
        attn_weights: torch.Tensor = self.qk_bmm(query_states, key_states.transpose(-1, -2))

        query_length, key_length = query_states.size(-2), key_states.size(-2)
        causal_mask: torch.Tensor = self.bias[:, :, key_length - query_length: key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 1. 对attn_weights进行再次的量化
        attn_weights = quantize_tensor(input_tensor=attn_weights,
                                       scale=self.attn_weights_scale,
                                       zero_point=self.attn_weights_zero_point)
        attn_output: torch.Tensor = self.pv_bmm(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        # ==============================================================================
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)

        return attn_output

