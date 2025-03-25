#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : quantize_gpt_qat
@File    : main.py
@Author  : Barry Allen
@Date    : 2025/3/19 14:04
@Desc    : QAT量化GPT-2
"""
import os

import torch
from torch.utils.data import TensorDataset
from transformers import GPT2Config

from models.ime_gpt2_lm_head_model import IMEGPT2LMHeadModel
from smooth_quant.smooth import smooth_ime_gpt2, get_static_decoder_layer_scales
from smooth_quant_model.int8_ime_gpt2_lm_head_model import Int8ImeGPT2LMHeadModel
from supports.utils import collect_ime_gpt2_act_scales, find_best_scales, create_dataset, export_int8_ime_gpt2_to_onnx

if __name__ == '__main__':
    purpose: str = "training"
    if purpose == "scales":
        ime_gpt2_path: str = os.path.join("/Users/mason/Desktop/Desktop/PythonProjects/ime_gpt_2/output", 'final_model')
        model: IMEGPT2LMHeadModel = IMEGPT2LMHeadModel.from_pretrained(ime_gpt2_path)
        find_best_scales(model=model)
    else:
        act_scales = torch.load("./act_scales/ime_gpt2.pt")
        ime_gpt2_path: str = os.path.join("/Users/mason/Desktop/Desktop/PythonProjects/ime_gpt_2/output", 'final_model')
        model: IMEGPT2LMHeadModel = IMEGPT2LMHeadModel.from_pretrained(ime_gpt2_path)
        print(model.transformer.h[0].ln_1.weight[:10])
        smooth_ime_gpt2(model=model, scales=act_scales, alpha=0.5)
        print(model.transformer.h[0].ln_1.weight[:10])
        dataset: TensorDataset = create_dataset()
        decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model=model,
                                                                           dataset=dataset)
        print(decoder_layer_scales)
        int8_model: Int8ImeGPT2LMHeadModel = Int8ImeGPT2LMHeadModel.from_float(module=model,
                                                                               config=model.config,
                                                                               intermediate_size=3072,
                                                                               decoder_layer_scales=decoder_layer_scales)
        input_ids: torch.LongTensor = torch.tensor([0, 3, 4, 5, 6, 7], dtype=torch.long)
        # 5. 进行推理
        output = model(input_ids)
        int8_output = int8_model(input_ids)
        values, indices = torch.max(output, dim=1)
        int8_values, int8_indices = torch.max(int8_output, dim=1)
        # export_int8_ime_gpt2_to_onnx(model=int8_model,
        #                              save_path="./int8_ime_gpt2.onnx",)
        print(f"scales: {indices} \n values: {values}")
        print(f"int8 scales: {int8_indices} \n values: {int8_values}")
