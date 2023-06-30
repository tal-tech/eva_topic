#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-17
'''

import torch
import numpy as np


class Infer(object):

    def __init__(self):
        pass

    @staticmethod
    def _inference(model, inference_data):
        model.eval()
        question_input = inference_data["question"]
        answer_input = inference_data["answer"]
        (question_input, answer_input) = (
            Infer.to_batch_tensor(question_input),
            Infer.to_batch_tensor(answer_input)
        )
        inference_result = model.forward(question_input, answer_input)
        return inference_result

    @staticmethod
    def to_batch_tensor(data):
        monitor_types = [np.ndarray, list]
        if type(data) in monitor_types:
            tensor_data = torch.LongTensor(data).unsqueeze(dim=0)
        elif isinstance(data, dict):
            tensor_data = dict()
            for (k, v) in data.items():
                if type(data) in monitor_types:
                    tensor_data[k] = torch.LongTensor(v).unsqueeze(dim=0)
                else:
                    tensor_data[k] = Infer.to_batch_tensor(v)
        else:
            raise TypeError("Infer can't transform {} to batch tensor!".format(type(data)))
        return tensor_data

