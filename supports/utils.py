#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : utils.py
@Author  : Barry Allen
@Date    : 2025/3/20 15:05
@Desc    : GPT-2的activate-scales
"""
import functools
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.ime_gpt2_lm_head_model import IMEGPT2LMHeadModel
from supports.constant import TRAIN_DATA_PATH


def collect_ime_gpt2_act_scales(model: IMEGPT2LMHeadModel,
                                dataset: TensorDataset,
                                num_samples: int = 2,
                                seq_len: int = 16):
    """
    拦截输入的激活函数值
    :param model: GPT2的LMHead模型
    :param dataset:
    :param num_samples: 序列样本数量
    :param seq_len: 序列长度
    :return:
    """
    act_scales: Dict[str, torch.Tensor] = {}  # 激活函数范围
    device: torch.device = next(model.parameters()).device

    def stat_tensor(tensor_name: str,
                    tensor_data: torch.Tensor):
        hidden_dim: int = tensor_data.shape[-1] # 隐藏层大小
        tensor_data = tensor_data.view(-1, hidden_dim).abs().detach() # 隐藏层绝对值的大小
        comming_max = torch.max(tensor_data, dim=0)[0].float().cpu() # 获取最大值
        if tensor_name in act_scales:
            act_scales[tensor_name] = torch.max(act_scales[tensor_name], comming_max)
        else:
            act_scales[tensor_name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    # 用于钩子函数
    hooks = []

    # 注册钩子到所有需要的层
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        input_ids = dataset[i][0].to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


def read_samples_from(file_path: str) -> torch.Tensor:
    ret: List[torch.Tensor] = []

    with open(file=file_path, mode='rt', encoding='utf-8') as read_file:
        while True:
            sample_line: str = read_file.readline()
            if not sample_line:
                break
            sample_lst: List[int] = list(map(int, sample_line.split()))
            ret.append(torch.tensor(sample_lst, dtype=torch.long))

    final_ret: torch.Tensor = torch.cat(ret, dim=0).view(len(ret), -1)

    return final_ret


def create_dataloader(input_tensor: torch.Tensor,
                      batch_size: int = 1) -> TensorDataset:
    # 假设输入形状是[512, 16]
    # 创建数据集
    dataset: TensorDataset = TensorDataset(input_tensor)

    # # 创建DataLoader
    # dataloader: DataLoader = DataLoader(dataset=dataset,
    #                                     batch_size=batch_size,
    #                                     shuffle=False) # 校准时通常不需要打乱数据

    return dataset


def find_best_scales(model: IMEGPT2LMHeadModel):
    """
    寻找最好的scales
    :param model: 输入法特定模型
    :return:
    """
    input_tensor: torch.Tensor = read_samples_from(file_path=TRAIN_DATA_PATH)
    data_loader: TensorDataset = create_dataloader(input_tensor=input_tensor)

    model.eval()

    # 收集激活值统计信息
    act_scales: Dict[str, torch.Tensor] = collect_ime_gpt2_act_scales(model=model, dataset=data_loader)

    # 保存统计信息
    torch.save(act_scales, "./act_scales/ime_gpt2.pt")


def create_dataset() -> TensorDataset:
    """
    无参数创建验证集
    :return:
    """
    input_tensor: torch.Tensor = read_samples_from(file_path=TRAIN_DATA_PATH)
    data_loader: TensorDataset = create_dataloader(input_tensor=input_tensor)
    return data_loader


def export_int8_ime_gpt2_to_onnx(model: nn.Module,
                                 save_path: str,
                                 batch_size: int = 1,
                                 sequence_length: int = 16):
    """
    导出GPT-2的onnx模型
    :param model:
    :param save_path:
    :param batch_size:
    :param sequence_length:
    :return:
    """
    # 1. 设置为评估模式
    model.eval()

    # 2. 准备示例输入
    dummy_input_ids: torch.Tensor = torch.randint(0, 6003, (batch_size, sequence_length))

    # 3. 定义输入输出名称
    input_names: List[str] = ["input_ids"]
    output_names: List[str] = ["hidden_states"]

    # 4. 导出模型
    torch.onnx.export(
        model,
        (dummy_input_ids),  # 模型输入
        save_path,  # 保存路径
        input_names=input_names,  # 输入名称
        output_names=output_names,  # 输出名称
        do_constant_folding=True,  # 进行常量折叠优化
        opset_version=20,  # ONNX算子版本
        verbose=False,
        export_params=True,  # 导出模型参数
    )
