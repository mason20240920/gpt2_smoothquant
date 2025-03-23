#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : smooth_quant_matmul.py
@Author  : Barry Allen
@Date    : 2025/3/22 14:35
@Desc    :
"""
from torch import nn
import torch as TF

# 保存未被修改的原始矩阵乘法函数
# original_matmul = TF.matmul

class SmoothQuantMatMul(nn.Module):
    def __init__(self):
        super(SmoothQuantMatMul, self).__init__()

    def forward(self, input1, input2):
        # 自定义操作
        print("自定义的 matmul 被调用")
        print(f"Input1 的前 10 个元素：\n{input1.flatten()[:10]}")
        print(f"Input2 的前 10 个元素：\n{input2.flatten()[:10]}")

        # 调用原始的矩阵乘法函数
        return original_matmul(input1, input2)

    def __call__(self, input1, input2):
        return self.forward(input1, input2)