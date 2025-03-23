#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : ime_gpt_2
@File    : ime_gpt2_pre_trained_model.py
@Author  : Barry Allen
@Date    : 2025/3/14 09:13
@Desc    : 预训练的权重初始化类
"""
import math
from typing import Callable, List

from transformers import PreTrainedModel, GPT2Config, load_tf_weights_in_gpt2
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class IMEGPT2PreTrainedModel(PreTrainedModel):
    """
    预训练的权重初始化类, a simple interface for downloading and loading pretrained models.
    """

    config_class: GPT2Config = GPT2Config
    load_tf_weights: Callable = load_tf_weights_in_gpt2
    base_model_prefix: str = "transformer"
    is_parallelizable: bool = True
    supports_gradient_checkpointing: bool = True
    _no_split_modules: List[str] = ["GPT2Block"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True
    _supports_sdpa: bool = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2RMSNorm):
            # module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 RMS Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))
