#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : quant_base_method.py
@Author  : Barry Allen
@Date    : 2025/3/21 09:46
@Desc    : 基本的量化方法
"""
import torch
from typing import Tuple

from torch.onnx.symbolic_opset11 import clamp


@torch.no_grad()
def quant_per_tensor_asymmetric(input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    非对称量化整个tensor
    :param input_tensor: 输入张量
    :return: 返回量化(量化后的uint8张量, scale和zero point)
    """
    # 获取最大最小值
    max_val:torch.Tensor = input_tensor.max()
    min_val:torch.Tensor = input_tensor.min()

    # 计算scale和zero point
    scale: torch.Tensor = (max_val - min_val) / 255
    zero_point: torch.Tensor = torch.round(-min_val / scale).to(torch.int8)

    # 在CPU上需要转为float进行运算
    if not input_tensor.is_cuda:
        input_tensor = input_tensor.float()

    # 量化过程: (x - min_val) / scale
    input_tensor = input_tensor.sub(min_val).div(scale).round_()

    # 限制在[0, 255]范围内并转为uint8
    input_tensor = torch.clamp(input_tensor, 0, 255)
    quantized_tensor: torch.Tensor = input_tensor.to(torch.uint8)
    return quantized_tensor, scale, zero_point


@torch.no_grad()
def quant_per_tensor_asymmetric_int8(input_tensor: torch.Tensor):
    """
    非对称量化整个tensor (有符号int8版本)
    :param input_tensor: 输入张量
    :return: 返回量化后的张量、scale和zero point
    """
    # 获取最大最小值
    max_val = input_tensor.max()
    min_val = input_tensor.min()

    # 计算scale和zero point
    scale = (max_val - min_val) / 255  # 范围仍然是255，因为总共有256个数值

    # 计算zero point并限制在int8范围内
    zero_point = torch.round(-128-min_val / scale).clamp(-128, 127).to(torch.int8)

    if not input_tensor.is_cuda:
        input_tensor = input_tensor.float()

    # 量化过程
    input_tensor = input_tensor.sub(min_val).div(scale).round_()

    # 限制在[-128, 127]范围内并转换为int8
    input_tensor = torch.clamp(input_tensor - 128, -128, 127)
    quantized_tensor = input_tensor.to(torch.int8)

    return quantized_tensor, scale, zero_point

@torch.no_grad()
def quant_per_tensor_asymmetric_int8_with_val(input_tensor: torch.Tensor,
                                              max_val: float = 2.0,
                                              min_val: float = -1.0):
    """
    非对称量化整个tensor (有符号int8版本)
    :param input_tensor: 输入张量
    :param max_val: 最大值
    :param min_val: 最小值
    :return: 返回量化后的张量、scale和zero point
    """
    # 计算scale和zero point
    scale: torch.Tensor = torch.tensor((max_val - min_val) / 255)  # 范围仍然是255，因为总共有256个数值

    # 计算zero point并限制在int8范围内
    zero_point = torch.clamp(torch.round(-128- min_val / scale), -128, 127).to(torch.int8)
    print(zero_point)

    if not input_tensor.is_cuda:
        input_tensor = input_tensor.float()

    # 量化过程
    input_tensor = input_tensor.sub(min_val).div(scale).round_()

    # 限制在[-128, 127]范围内并转换为int8
    input_tensor = torch.clamp(input_tensor - 128, -128, 127)
    quantized_tensor = input_tensor.to(torch.int8)

    return quantized_tensor, scale, zero_point

@torch.no_grad()
def channel_wise_symmetric_quantize(weight_tensor: torch.Tensor,
                                    quant_bit: int = 8,
                                    channel_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对称per channel矩阵量化
    :param weight_tensor: 权重张量
    :param quant_bit:
    :param channel_dim:
    :return:
    """
    # 1. 计算量化参数 (对称int8范围为[-127, 127])
    offset = 1 << (quant_bit - 1)
    clip_max = offset - 1
    clip_min = -offset

    # 2. 获取通道数量
    num_channels: int = weight_tensor.shape[channel_dim]

    # 3. 初始化缩放因子和量化后的张量
    q_weight_tensor: torch.Tensor = torch.zeros_like(weight_tensor)
    scale = torch.zeros(num_channels, dtype=torch.float32)

    # 4. 对每个通道单独进行量化
    for c in range(num_channels):
        # 5. 创建索引以选择特定通道
        indices = [slice(None)] * len(weight_tensor.shape)
        indices[channel_dim] = c

        # 获取当前通道的数据
        channel_data = weight_tensor[tuple(indices)]

        # 计算当前通道的最大绝对值
        max_abs = torch.max(torch.abs(channel_data)).item()

        # 计算当前通道的scale
        channel_scale = max_abs / clip_max
        # 避免除零错误
        if channel_scale == 0:
            channel_scale = 1.0

        # 存储scale
        scale[c] = channel_scale

        # 对当前通道进行量化
        q_channel = torch.round(channel_data / channel_scale)
        q_channel = torch.clamp(q_channel, clip_min, clip_max)

        # 将量化后的通道数据放回结果张量
        q_weight_tensor[tuple(indices)] = q_channel

    return q_weight_tensor.to(torch.int8), scale


@torch.no_grad()
def quant_bias(input_tensor: torch.Tensor,
               input_scale: float,
               weight_scale: torch.Tensor,
               quant_bit: int = 8) -> torch.Tensor:
    """
    量化偏置值
    :param input_tensor:
    :param input_scale:
    :param weight_scale:
    :param quant_bit:
    :return:
    """
    offset = 1 << (quant_bit - 1)
    clip_max = offset - 1
    clip_min = -offset

    q_bias: torch.Tensor = torch.round(input_tensor / (input_scale * weight_scale))
    q_bias = torch.clamp(q_bias, clip_min, clip_max)

    return q_bias.to(torch.int32)


@torch.no_grad()
def dequantize_gemm_output(q_c: torch.Tensor, input_scale: float, weight_scale: torch.Tensor):
    """
    将GEMM的int32输出反量化为float32
    :param q_c: GEMM的int32输出 [M, N]
    :param input_scale: 输入的缩放因子 (标量)
    :param weight_scale: 权重的每通道缩放因子 [N]
    :return:
    """
    float_output = q_c.float()

    # 2. 将weight_scale调整维度以便广播 [N] -> [1, N]
    weight_scale = weight_scale.reshape(1, -1)

    # 3. 直接应用scale因子
    float_output = float_output * input_scale * weight_scale

    return float_output

@torch.no_grad()
def dequantize_asymmetric_quantize(q_A: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    return (q_A.to(torch.float32) - zero_point) * scale


@torch.no_grad()
def quantize_tensor(input_tensor: torch.Tensor,
                    scale: float,
                    zero_point: torch.int8,
                    num_bits: int = 8) -> torch.Tensor:
    """
    量化一个张量并返回量化后的张量
    :param input_tensor: 输入张量
    :param scale:
    :param zero_point:
    :param num_bits:
    :return:
    """
    offset = 1 << (num_bits - 1)
    clip_max = offset - 1
    clip_min = -offset

    # 量化公式
    output = torch.round(input_tensor / scale + zero_point)

    # 裁剪到有效范围
    output = torch.clamp(output, clip_min, clip_max)

    return output.to(torch.int8)

@torch.no_grad()
def dequantize_tensor(input_tensor: torch.Tensor,
                      scale: float,
                      zero_point: torch.int8) -> torch.Tensor:
    return (input_tensor.float() - zero_point) * scale



