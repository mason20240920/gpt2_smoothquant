#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : int8_ime_gpt2_mlp.py
@Author  : Barry Allen
@Date    : 2025/3/21 18:43
@Desc    : 量化的MLP层
"""
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from transformers import GPT2Config
from transformers.activations import ACT2FN

from torch import nn

from models.ime_gpt2_mlp import IMEGPT2MLP
from quant_ops.quant_linear import W8A8BFP32OFP32LinearForMac, W8A8B32O8LinearForMac
from supports.quant_base_method import quantize_tensor, dequantize_asymmetric_quantize, map_act_to_255_lst, map_tensor_values


class Int8IMEGPT2MLP(nn.Module):
    """
    Int8 量化的 GPT-2
    """
    c_fc: W8A8BFP32OFP32LinearForMac

    c_proj: W8A8BFP32OFP32LinearForMac

    act: ACT2FN

    c_fc_input_scale: float

    c_fc_input_zero_point: torch.int8

    c_proj_output_scale: float

    c_proj_output_zero_point: torch.int8

    c_gelu_output_scale: float

    c_gelu_output_zero_point: torch.int8

    c_gelu_input_scale: float

    c_gelu_input_zero_point: torch.int8

    gelu_map: Dict[int, int] = {}

    def __init__(self, config: GPT2Config, intermediate_size: int):
        super(Int8IMEGPT2MLP, self).__init__()
        self.config = config
        embed_dim: int = config.hidden_size
        self.c_fc = W8A8BFP32OFP32LinearForMac(in_features=embed_dim, out_features=intermediate_size)
        self.c_proj = W8A8BFP32OFP32LinearForMac(in_features=intermediate_size, out_features=embed_dim)
        self.act = ACT2FN[config.activation_function]

    @staticmethod
    @torch.no_grad()
    def from_float(
            module: IMEGPT2MLP,
            config: GPT2Config,
            intermediate_size: int,
            c_fc_input_scale: float,
            c_fc_input_zp: float,
            c_gelu_input_scale: float,
            c_gelu_input_zp: float,
            c_gelu_output_scale: float,
            c_gelu_output_zp: float,
            c_proj_output_scale: float,
            c_proj_output_zp: float):
        int8_module: Int8IMEGPT2MLP = Int8IMEGPT2MLP(config=config,
                                                     intermediate_size=intermediate_size)
        int8_module.c_fc_input_zero_point = c_fc_input_zp
        int8_module.c_proj_output_zero_point = c_proj_output_zp
        int8_module.c_proj_output_scale = c_proj_output_scale
        int8_module.c_fc_input_scale = c_fc_input_scale
        int8_module.c_gelu_output_scale = c_gelu_output_scale
        int8_module.c_gelu_output_zero_point = c_gelu_output_zp
        int8_module.c_gelu_input_scale = c_gelu_input_scale
        int8_module.c_gelu_input_zero_point = c_gelu_input_zp
        int8_module.c_fc = W8A8B32O8LinearForMac.from_float(
            module=module.c_fc,
            input_scale=c_fc_input_scale,
            input_zp=c_fc_input_zp,
            output_scale=c_gelu_input_scale,
            output_zp=c_gelu_input_zp,
        )
        int8_module.gelu_map = map_act_to_255_lst(act_func=module.act,
                                                  scale=c_gelu_input_scale,
                                                  zp=c_gelu_input_zp,
                                                  o_scale=c_gelu_output_scale,
                                                  o_zp=c_gelu_output_zp)
        int8_module.c_proj = W8A8BFP32OFP32LinearForMac.from_float(
            module=module.c_proj,
            input_scale=c_gelu_output_scale,
            input_zp=c_gelu_output_zp
        )
        return int8_module


    def forward(self,
                hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        """
        前向推理
        :param hidden_states:
        :return:
        """
        # np.save("mlp_input.npy",hidden_states.detach().cpu().numpy())
        # 对输入进行一次量化
        hidden_states = quantize_tensor(input_tensor=hidden_states,
                                        scale=self.c_fc_input_scale,
                                        zero_point=self.c_fc_input_zero_point)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = map_tensor_values(tensor=hidden_states,
                                          mapping=self.gelu_map)
        # hidden_states = dequantize_asymmetric_quantize(q_A=hidden_states,
        #                                                scale=self.c_gelu_output_scale,
        #                                                zero_point=self.c_gelu_output_zero_point)
        hidden_states = self.c_proj(hidden_states)
        # hidden_states = dequantize_asymmetric_quantize(q_A=hidden_states,
        #                                                scale=self.c_proj_output_scale,
        #                                                zero_point=self.c_proj_output_zero_point)
        return hidden_states

