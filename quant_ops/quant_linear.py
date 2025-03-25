#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : quant_linear.py
@Author  : Barry Allen
@Date    : 2025/3/20 20:09
@Desc    :
"""
import torch
from torch import nn
from supports.quant_base_method import channel_wise_symmetric_quantize, quant_bias


class W8A8B32O8LinearForMac(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 alpha: float =1.0,
                 beta: float =1.0):
        """
        线性层
        :param in_features: 输入维度
        :param out_features: 输出维度
        :param alpha:
        :param beta:
        """
        super(W8A8B32O8LinearForMac, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features

        self.register_buffer(
            'weight',
            torch.randint(-127,127, (self.out_features,self.in_features),
                          dtype=torch.int8,
                          requires_grad=False))

        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int32, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))
        self.zero_point: torch.int8 = 0  # 零点值
        self.out_zp: torch.int8 = 0 # 输出零点值
        self.out_scale: float = 0  # 输出缩放因子
        self.gemm_scales: torch.Tensor

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @staticmethod
    def from_float(module: nn.Linear,
                   input_scale: float,
                   input_zp: torch.int8,
                   output_scale: float,
                   output_zp: torch.int8):
        int8_module: W8A8B32O8LinearForMac = W8A8B32O8LinearForMac(in_features=module.in_features,
                                                                 out_features=module.out_features)
        int8_weights, weight_scale = channel_wise_symmetric_quantize(module.weight.t()) # 权重量化
        s32_bias: torch.Tensor = quant_bias(input_tensor=module.bias,
                                            input_scale=input_scale,
                                            weight_scale=weight_scale)  # 偏置量化
        int8_module.weight = int8_weights
        int8_module.bias = s32_bias
        int8_module.zero_point = input_zp  # 输入的zero point
        int8_module.out_zp = output_zp  # 输出的zero point
        int8_module.out_scale = output_scale
        weight_scale = weight_scale.view(1, -1)  # 扩展维度
        int8_module.gemm_scales = weight_scale * input_scale
        return int8_module

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        前向传播
        :param x: 输入
        :return:
        """
        # 1. 将输入转换为int32
        input_q: torch.Tensor = x.to(torch.int32)
        weight_q: torch.Tensor = self.weight.to(torch.int32)

        adjust_a: torch.Tensor = input_q - self.zero_point

        adjust_a_float = adjust_a.float()
        weight_q_float = weight_q.float()

        # 2. 进行矩阵计算推理
        gemm_o: torch.Tensor = ((torch.matmul(adjust_a_float,  weight_q_float) + self.bias) * self.gemm_scales) / self.out_scale

        # origin_gemm_o: torch.Tensor = (torch.matmul(adjust_a, weight_q) + self.bias)

        # 3. 获取的是S32的结果, 进行转为int 8 的结果
        output: torch.Tensor = torch.clamp(gemm_o + self.out_zp, -128, 127).to(torch.int8)
        return output

class W8A8BFP32OFP32LinearForMac(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 alpha: float =1.0,
                 beta: float =1.0):
        """
        线性层
        :param in_features: 输入维度
        :param out_features: 输出维度
        :param alpha:
        :param beta:
        """
        super(W8A8BFP32OFP32LinearForMac, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features

        self.register_buffer(
            'weight',
            torch.randint(-127,127, (self.out_features,self.in_features),
                          dtype=torch.int8,
                          requires_grad=False))

        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int32, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))
        self.zero_point: torch.int8 = 0  # 零点值
        self.out_zp: torch.int8 = 0 # 输出零点值
        self.out_scale: float = 0  # 输出缩放因子
        self.gemm_scales: torch.Tensor

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @staticmethod
    def from_float(module: nn.Linear,
                   input_scale: float,
                   input_zp: torch.int8):
        int8_module: W8A8BFP32OFP32LinearForMac = W8A8BFP32OFP32LinearForMac(in_features=module.in_features,
                                                                             out_features=module.out_features)
        int8_weights, weight_scale = channel_wise_symmetric_quantize(module.weight.t()) # 权重量化
        s32_bias: torch.Tensor = quant_bias(input_tensor=module.bias,
                                            input_scale=input_scale,
                                            weight_scale=weight_scale)  # 偏置量化
        int8_module.weight = int8_weights
        int8_module.bias = s32_bias
        int8_module.zero_point = input_zp  # 输入的zero point
        weight_scale = weight_scale.view(1, -1)  # 扩展维度
        int8_module.gemm_scales = weight_scale * input_scale
        return int8_module

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        前向传播
        :param x: 输入
        :return:
        """
        # 1. 将输入转换为int32
        input_q: torch.Tensor = x.to(torch.int32)
        weight_q: torch.Tensor = self.weight.to(torch.int32)

        adjust_a: torch.Tensor = input_q - self.zero_point


        adjust_a_float = adjust_a.float()
        weight_q_float = weight_q.float()

        # 2. 进行矩阵计算推理
        gemm_o: torch.Tensor = ((torch.matmul(adjust_a_float, weight_q_float) + self.bias) * self.gemm_scales)
        return gemm_o




