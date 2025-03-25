#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : quant_bmm.py
@Author  : Barry Allen
@Date    : 2025/3/21 15:29
@Desc    : 量化的BMM操作
"""
import torch


class BMMS8TS8NS8T(torch.nn.Module):
    """
    传入一个张量S8和S8,输出S8的矩阵
    """

    def __init__(self,
                 alpha: float,
                 tensor_a_zp: float,
                 tensor_b_zp: float,
                 tensor_o_zp: float,
                 tensor_o_alpha: float):
        """
        初始化获取的BMM算子
        :param alpha:
        :param tensor_a_zp:
        :param tensor_b_zp:
        """
        super(BMMS8TS8NS8T, self).__init__()
        self.register_buffer('a_zp', torch.tensor(tensor_a_zp))
        self.register_buffer('b_zp', torch.tensor(tensor_b_zp))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('out_zp', torch.tensor(tensor_o_zp))
        self.register_buffer('o_alpha', torch.tensor(tensor_o_alpha))

    @staticmethod
    def from_a_and_b(a_zp: float,
                     b_zp: float,
                     o_zp: float,
                     a_scale: float,
                     b_scale: float,
                     o_scale: float):
        bmm_module: BMMS8TS8NS8T = BMMS8TS8NS8T(alpha=1.0,
                                                tensor_a_zp=1.0,
                                                tensor_b_zp=1.0,
                                                tensor_o_zp=1.0,
                                                tensor_o_alpha=1.0)
        alpha_scale: float = a_scale * b_scale
        if not torch.is_tensor(alpha_scale):
            alpha_scale: torch.Tensor = torch.tensor(alpha_scale)
        if not torch.is_tensor(a_zp):
            a_zp: torch.Tensor = torch.tensor(a_zp)
        if not torch.is_tensor(b_zp):
            b_zp: torch.Tensor = torch.tensor(b_zp)
        if not torch.is_tensor(o_zp):
            o_zp: torch.Tensor = torch.tensor(o_zp)
        if not torch.is_tensor(o_scale):
            o_scale: torch.Tensor = torch.tensor(o_scale)
        bmm_module.alpha = alpha_scale
        bmm_module.a_zp = a_zp
        bmm_module.b_zp = b_zp
        bmm_module.out_zp = o_zp
        bmm_module.o_alpha = o_scale
        return bmm_module

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param y:
        :param x:
        :return:
        """
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int32
        input_a: torch.Tensor = (x.to(torch.int32) - self.a_zp).float()
        input_b: torch.Tensor = (y.to(torch.int32) - self.b_zp).float()
        quant_o: torch.Tensor = torch.matmul(input_a,
                                             input_b) * self.alpha / self.o_alpha

        output: torch.Tensor = torch.clamp(quant_o + self.out_zp, -128, 127).to(torch.int8)

        return output


class BMMS8TS8NF32T(torch.nn.Module):
    def __init__(self,
                 alpha: float,
                 tensor_a_zp: float,
                 tensor_b_zp: float):
        """
        初始化获取的BMM算子
        :param alpha:
        :param tensor_a_zp:
        :param tensor_b_zp:
        """
        super(BMMS8TS8NF32T, self).__init__()
        self.register_buffer('a_zp', torch.tensor(tensor_a_zp))
        self.register_buffer('b_zp', torch.tensor(tensor_b_zp))
        self.register_buffer('alpha', torch.tensor(alpha))

    @staticmethod
    def from_a_and_b(a_zp: float,
                     b_zp: float,
                     a_scale: float,
                     b_scale: float):
        bmm_module: BMMS8TS8NF32T = BMMS8TS8NF32T(alpha=1.0,
                                                  tensor_a_zp=1.0,
                                                  tensor_b_zp=1.0)
        alpha_scale: float = a_scale * b_scale
        if not torch.is_tensor(alpha_scale):
            alpha_scale: torch.Tensor = torch.tensor(alpha_scale)
        if not torch.is_tensor(a_zp):
            a_zp: torch.Tensor = torch.tensor(a_zp)
        if not torch.is_tensor(b_zp):
            b_zp: torch.Tensor = torch.tensor(b_zp)
        bmm_module.alpha = alpha_scale
        bmm_module.a_zp = a_zp
        bmm_module.b_zp = b_zp
        return bmm_module

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param y:
        :param x:
        :return:
        """
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] float32
        input_a: torch.Tensor = (x.to(torch.int32) - self.a_zp).float()
        input_b: torch.Tensor = (y.to(torch.int32) - self.b_zp).float()
        quant_o: torch.Tensor = torch.matmul(input_a,
                                             input_b) * self.alpha

        return quant_o
