#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : ime_gpt_2
@File    : ime_gpt2_model.py
@Author  : Barry Allen
@Date    : 2025/3/14 09:22
@Desc    : 预训练的IME定制版GPT-2模型
"""
import warnings
from typing import Callable, Optional, Tuple

import torch
from transformers import add_start_docstrings, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2_START_DOCSTRING, GPT2_INPUTS_DOCSTRING, _CHECKPOINT_FOR_DOC, \
    _CONFIG_FOR_DOC, PARALLELIZE_DOCSTRING, DEPARALLELIZE_DOCSTRING
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from models.ime_gpt2_pre_trained_model import IMEGPT2PreTrainedModel
from models.ime_gpt2_block import IMEGPT2Block
from torch import nn


@add_start_docstrings(
"The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class IMEGPT2Model(IMEGPT2PreTrainedModel):
    """
    预训练模型
    """
    _supports_param_buffer_assignment: bool = False
    embed_dim: int # 嵌入维度

    wte: nn.Embedding # Word Token Embeddings（词元嵌入）
    wpe: nn.Embedding # Word Position Embeddings（词位嵌入）

    drop: nn.Dropout # 正则化

    h: nn.ModuleList # 隐藏层

    ln_f: Qwen2RMSNorm # 归一化层

    gradient_checkpointing: bool # 梯度检查点

    _attn_implementation: str # 注意力实现

    device_map: dict

    first_device: str

    last_device: str

    model_parallel: bool = False # 模型并行

    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(IMEGPT2Block(config=config, layer_idx=i) for i in range(config.num_hidden_layers))
        self.ln_f = Qwen2RMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        warnings.warn(
            "`GPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple:
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

        # 7. 使用Dropout层进行正则化
        hidden_states = self.drop(hidden_states)

        # 8. 将原始的二维张量重塑为三维张量
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)  # [-1, sequence length, hidden size]

        # 9. 循环遍历每个GPT2Block模块
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block=block.__call__,
                    hidden_states=hidden_states,
                )
            else:
                outputs = block(
                    hidden_states
                )

            # 得到输出结果
            hidden_states = outputs

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # 10. RMSNorm层进行正则化
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        return tuple(
            v
            for v in [hidden_states]
            if v is not None
        )




