#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : ime_gpt_2
@File    : ime_gpt_2_attention.py
@Author  : Barry Allen
@Date    : 2025/3/13 20:58
@Desc    : IME版本的GPT-2注意力层
"""
from torch import nn
from transformers import GPT2Config
from typing import Optional, Tuple, Callable
import torch

def eager_attention_forward(module: nn.Module,
                            query: torch.Tensor,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            **kwargs) -> torch.Tensor:
    attn_weights: torch.Tensor = torch.matmul(query, key.transpose(-1, -2))

    # if only "normal" attention layer implements causal mask
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask: torch.Tensor = module.bias[:, :, key_length - query_length: key_length, :key_length]
    mask_value = torch.finfo(attn_weights.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output


class IMEGPT2Attention(nn.Module):
    """
    IME版本的GPT-2注意力层
    """
    config: GPT2Config

    embed_dim: int # 嵌入维度

    num_heads: int # 注意力头数量

    head_dim: int # 每个注意力头的维度

    split_size: int # 分割大小

    c_attn: nn.Linear # 输入注意力

    c_proj: nn.Linear # 输出注意力

    attn_dropout: nn.Dropout # 注意力正则化

    resid_dropout: nn.Dropout # 残差正则化

    is_casual: bool # 是否是因果模型

    pruned_heads: set # 修剪的注意力头

    def __init__(self, config: GPT2Config, layer_idx: Optional[int] = None):
        """
        初始化GPT-2的自注意力
        :param config:
        :param layer_idx:
        """
        super(IMEGPT2Attention, self).__init__()
        self.config = config
        max_positions:int = config.max_position_embeddings
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

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        # 判断注意力头和维度是否能被整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 输入注意力
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # 正则化
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_casual = True # 是否是因果模型

        self.pruned_heads = set()

    def forward(self,
                hidden_states: Optional[Tuple[torch.Tensor]]):
        """
        前向推理
        :param hidden_states:
        :return:
        """
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim) # [batch_size, seq_len, num_heads, head_dim]
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)  # [batch_size, seq_len, num_heads, head_dim]

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        present = None

        attention_interface: Callable = eager_attention_forward

        attn_output = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            dropout=self.attn_dropout.p if self.training else 0.0
        )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output





