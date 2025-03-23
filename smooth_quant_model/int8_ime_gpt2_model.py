#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : int8_ime_gpt2_model.py
@Author  : Barry Allen
@Date    : 2025/3/21 19:25
@Desc    : Int8量化的GPT2模型
"""
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import GPT2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from models.ime_gpt2_model import IMEGPT2Model
from models.ime_gpt2_pre_trained_model import IMEGPT2PreTrainedModel
from smooth_quant_model.int8_ime_gpt2_block import Int8IMEGPT2Block


class Int8ImeGPT2Model(IMEGPT2PreTrainedModel):

    def __init__(self, config: GPT2Config):
        super(Int8ImeGPT2Model, self).__init__(config)

        self.embed_dim: int = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.h: nn.ModuleList[Int8IMEGPT2Block] = nn.ModuleList(
            Int8IMEGPT2Block(config=config) for i in range(config.num_hidden_layers))
        self.ln_f = Qwen2RMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module: IMEGPT2Model,
                   config: GPT2Config,
                   intermediate_size: int,
                   decoder_layer_scales: dict):
        int8_module = Int8ImeGPT2Model(config=config)
        for i, h in enumerate(module.h):
            int8_module.h[i] = Int8IMEGPT2Block.from_float(
                h, config, intermediate_size, **decoder_layer_scales[i]
            )
        int8_module.wte = module.wte
        int8_module.wpe = module.wpe
        int8_module.ln_f = module.ln_f
        return int8_module

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None) -> Tuple:
        """
        前向传播
        :param input_ids:
        :param attention_mask:
        :return:
        """
        # 1. 先判断输入的input ids是否合法
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()  # [batch size, sequence length]
            input_ids = input_ids.view(-1, input_shape[-1])  # 将输入张量统一转换为二维形式 [batch size, sequence length]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 2. 确定输入的设备
        device: torch.device = input_ids.device if input_ids is not None else "cpu"

        # 3. 键值缓存（KV Cache）- 暂时没有使用
        past_length = 0
        past_key_values = tuple([None] * len(self.h))  # 创建一个列表，长度等于(block)层数，每个元素都是None

        # 4. 确定输入的位置编码
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long,
                                    device=device)  # [sequence length + 0]
        position_ids = position_ids.unsqueeze(0)

        # 5. 输入词的embedding
        inputs_embeds = self.wte(input_ids)  #

        # 6. 位置编码的embedding
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)  # wpe + wte

        # 8. 将原始的二维张量重塑为三维张量
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)  # [-1, sequence length, hidden size]

        # 9. 循环遍历每个GPT2Block模块
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]

            outputs = block(
                hidden_states
            )

            # 得到输出结果
            hidden_states = outputs

        # 10. RMSNorm层进行正则化
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        return tuple(
            v
            for v in [hidden_states]
            if v is not None
        )
