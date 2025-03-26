#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : smooth.py
@Author  : Barry Allen
@Date    : 2025/3/20 11:28
@Desc    : GPT-2版本的平滑量化SmoothQuant
"""
from collections import defaultdict
from functools import partial
from typing import List, Dict, Any

import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from models.ime_gpt2_block import IMEGPT2Block
from models.ime_gpt2_lm_head_model import IMEGPT2LMHeadModel

import torch
from torch import nn


@torch.no_grad()
def smooth_ln_fcs(ln: Qwen2RMSNorm,
                  fcs: List[nn.Linear],
                  act_scales: torch.Tensor,
                  alpha=0.5):
    """
    平滑fcs
    :param ln: 父算子
    :param fcs: 子算子列表(里面的Linear线性层)
    :param act_scales: scale张量信息
    :param alpha:
    :return:
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (nn.LayerNorm, Qwen2RMSNorm))
    # 确定是不是线性层
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    # 计算所有Linear层权重的最大绝对值
    # 对每个fc层，先在输出维度上取最大值[1, in_features]
    # 然后将所有fc层的结果拼接[num_fcs, in_features]
    # 最后在fc层维度上取最大值[in_features]
    weight_scales: torch.Tensor = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # 计算最终的缩放因子：scales = (A^α / W^(1-α))
    # 其中A是激活值缩放，W是权重缩放
    # 这样可以平衡激活值和权重的量化误差
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # 调整LayerNorm层的参数
    # 除以scales相当于减小输出范围
    if isinstance(ln, Qwen2RMSNorm):
        ln.weight.div_(scales)
    elif isinstance(ln, nn.LayerNorm):
        ln.weight.div_(scales)
        ln.bias.div_(scales)

    # 调整每个Linear层的权重
    # 乘以scales相当于补偿LayerNorm的缩放
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ime_gpt2(model: IMEGPT2LMHeadModel,
                    scales: Dict[str, torch.Tensor],
                    alpha: float = 0.5):
    """
    平滑
    :param model: 需要进行SmoothQuant的模型
    :param scales:  存储激活值统计信息
    :param alpha:
    :return:
    """
    for name, module in model.named_modules():
        # 用前置的LayerNorm来补偿后续的Linear层的scales
        if isinstance(module, IMEGPT2Block):
            attn_ln: Qwen2RMSNorm = module.ln_1
            qkv: nn.Linear = module.attn.c_attn
            qkv_input_scales: torch.Tensor = scales[name + ".attn.c_attn"]
            smooth_ln_fcs(ln=attn_ln, fcs=[qkv], act_scales=qkv_input_scales, alpha=alpha)

            ffn_ln: Qwen2RMSNorm = module.ln_2
            fc1: nn.Linear = module.mlp.c_fc
            fc1_input_scales: torch.Tensor = scales[name + ".mlp.c_fc"]
            smooth_ln_fcs(ln=ffn_ln, fcs=[fc1], act_scales=fc1_input_scales, alpha=alpha)
        # else:
        #     raise NotADirectoryError("Module '{}' not found.".format(name))


@torch.no_grad()
def get_static_decoder_layer_scales(
        model: IMEGPT2LMHeadModel,
        dataset: TensorDataset,
        num_samples:int = 2,
        seq_len: int = 16):
    """
    获取解码器层的激活值缩放因子, 用于量化准备
    :param model: 输入法定制模型
    :param dataset: 验证的数据集
    :param num_samples: 样本数量
    :param seq_len: 序列长度
    :return:
    """
    # 对torch.matmul进行替换拦截

    device: torch.device = next(model.parameters()).device

    act_dict = defaultdict(dict) # 存储每个模块的输入输出最大值
    model.eval()

    def stat_io_hook(m: nn.Module,
                     x: Any,
                     y: torch.Tensor,
                     name: str):
        """
        钩子函数: 记录模块的输入输出最大值
        :param m: 模块
        :param x: 输入
        :param y: 输出
        :param name: 模块名称
        :return:
        """
        # 处理输入, 如果是元祖则取第一个元素
        if isinstance(x, tuple):
            x = x[0]
        # 处理输出, 如果是元祖则取第一个元素
        if isinstance(y, tuple):
            y = y[0]
        # 先判断是不是注意力头

        if name.endswith("attn.c_attn"):
            # 进行QKV形状变化
            query_states, key_states, value_states = y.split(m.in_features, dim=2)
            head_dim = m.in_features // model.config.n_head
            shape_q = (*query_states.shape[:-1], -1, head_dim)  # [batch_size, seq_len, num_heads, head_dim]
            shape_kv = (*key_states.shape[:-1], -1, head_dim)  # [batch_size, seq_len, num_heads, head_dim]

            query_states = query_states.view(shape_q).transpose(1, 2)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

            sub_y_output = [query_states, key_states, value_states]

            prefix_name: str = name.replace("c_attn", "")
            sub_names: List[str] = [f"{prefix_name}q_proj",
                                    f"{prefix_name}k_proj",
                                    f"{prefix_name}v_proj"]
            for sub_index, sub_name in enumerate(sub_names):
                if sub_name not in act_dict or 'input' not in act_dict[sub_name]:
                    act_dict[sub_name]['input_max'] = x.detach().max().item()
                    act_dict[sub_name]['input_min'] = x.detach().min().item()
                else:
                    act_dict[sub_name]['input_max'] = max(act_dict[sub_name]['input_max'], x.detach().max().item())
                    act_dict[sub_name]['input_min'] = min(act_dict[sub_name]['input_min'], x.detach().min().item())

                if sub_name not in act_dict or 'output' not in act_dict[sub_name]:
                    act_dict[sub_name]['output_max'] = sub_y_output[sub_index].detach().max().item()
                    act_dict[sub_name]['output_min'] = sub_y_output[sub_index].detach().min().item()
                else:
                    act_dict[sub_name]['output_max'] = max(act_dict[sub_name]['output_max'],
                                                           sub_y_output[sub_index].detach().max().item())
                    act_dict[sub_name]['output_min'] = max(act_dict[sub_name]['output_min'],
                                                           sub_y_output[sub_index].detach().min().item())



        # 更新输入最大值
        if name not in act_dict or 'input' not in act_dict[name]:
            act_dict[name]['input_max'] = x.detach().max().item()
            act_dict[name]['input_min'] = x.detach().min().item()
        else:
            act_dict[name]['input_max'] = max(
                act_dict[name]['input_max'], x.detach().max().item()
            )
            act_dict[name]['input_min'] = min(
                act_dict[name]['input_min'], x.detach().min().item()
            )

        # 更新输出最大值
        if name not in act_dict or 'output' not in act_dict[name]:
            act_dict[name]['output_max'] = y.detach().max().item()
            act_dict[name]['output_min'] = y.detach().min().item()
        else:
            act_dict[name]['output_max'] = max(
                act_dict[name]['output_max'], y.detach().max().item()
            )
            act_dict[name]['output_min'] = min(
                act_dict[name]['output_min'], y.detach().min().item()
            )

    # 注册钩子
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))

    # 遍历样本收集激活值统计信息
    for i in pbar:
        # 对文本进行分词
        input_ids = dataset[i][0].to(device)

        model(input_ids)  # 前向传播
        # 显示当前平均输入范围
        mean_range = np.mean([
            v["input_max"] - v["input_min"] for v in act_dict.values()
        ])
        pbar.set_description(f"Mean input scale: {mean_range:.2f}")

    # 清理钩子
    for hook in hooks:
        hook.remove()

    # 收集每一层的缩放因子
    decoder_layer_scales: List = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}

        # 计算每层的scale和zero point
        def compute_scale_zp(module_name, io_type):
            max_val = act_dict[module_name][f"{io_type}_max"]
            min_val = act_dict[module_name][f"{io_type}_min"]
            scale = (max_val - min_val) / 255
            zp = torch.round(torch.tensor(-128-min_val / scale)).clamp(-128, 127).to(torch.int8).item()
            return scale, zp

        # 计算每层的scale和zero point
        def compute_scale_zp_without_name():
            max_val = 1.0
            min_val = 0.0
            scale = (max_val - min_val) / 255
            zp = torch.round(torch.tensor(-128 - min_val / scale)).clamp(-128, 127).to(torch.int8).item()
            return scale, zp
        # 注意力层的输入和qkv输出缩放
        # 注意力层的输入和输出
        attn_name = f"transformer.h.{idx}.attn.c_attn"
        scale_dict["attn_input_scale"], scale_dict["attn_input_zp"] = compute_scale_zp(attn_name, "input")
        scale_dict["attn_output_scale"], scale_dict["attn_output_zp"] = compute_scale_zp(attn_name, "output")

        scale_dict["q_output_scale"], scale_dict["q_output_zp"] = compute_scale_zp(f"transformer.h.{idx}.attn.q_proj", "output")
        scale_dict["k_output_scale"], scale_dict["k_output_zp"] = compute_scale_zp(f"transformer.h.{idx}.attn.k_proj", "output")
        scale_dict["v_output_scale"], scale_dict["v_output_zp"] = compute_scale_zp(f"transformer.h.{idx}.attn.v_proj", "output")

        scale_dict["attn_weight_input_scale"], scale_dict["attn_weight_input_zp"] = compute_scale_zp_without_name()

        # 投影层
        proj_name = f"transformer.h.{idx}.attn.c_proj"
        scale_dict["out_input_scale"], scale_dict["out_input_zp"] = compute_scale_zp(proj_name, "input")

        # FFN层
        fc1_name = f"transformer.h.{idx}.mlp.c_fc"
        fc2_name = f"transformer.h.{idx}.mlp.c_proj"
        scale_dict["fc1_input_scale"], scale_dict["fc1_input_zp"] = compute_scale_zp(fc1_name, "input")
        scale_dict["fc2_output_scale"], scale_dict["fc2_output_zp"] = compute_scale_zp(fc2_name, "input")

        decoder_layer_scales.append(scale_dict)
        decoder_layer_scales.append(scale_dict)

    # # 恢复原始的 torch.matmul
    # torch.matmul = original_matmul

    return decoder_layer_scales, act_dict


