#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-15
'''


# API Implementation
import torch
import os
import sys
project_root_directory = os.path.dirname(
            os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))  # Project root directory
sys.path.append(project_root_directory)
from eva_model.QAREToolkit.qare.api.load_resource import get_api_config
from eva_model.QAREToolkit.qare.api.load_resource import load_resource
from eva_model.QAREToolkit.qare.utils.utils import Utils
import pandas as pd


MODEL_NAME, DATASET_NAME = get_api_config()
(config, vocab, model, transform_instance, process_tokens, reader) = load_resource(MODEL_NAME, DATASET_NAME)


class QareAPI(object):

    def __init__(self,
                 config=config,
                 vocab=vocab,
                 model=model,
                 transform_instance=transform_instance,
                 process_tokens=process_tokens,
                 reader=reader):
        self.model = model
        self.config = config
        self.vocab = vocab
        self.transform = transform_instance
        self.process_tokens = process_tokens
        self.reader = reader

    def _preprocess(self, question, answer):
        if isinstance(question, pd.Series):
            question = question.item()
        if isinstance(answer, pd.Series):
            answer = answer.item()
        request_data = {"question": str(question), "answer": str(answer)}
        read_data = self.reader.read_json_data(request_data)

        transform_kwargs = {"vocab": self.vocab,
                            "question_truncate_len": self.config.question_truncate_len,
                            "answer_truncate_len": self.config.answer_truncate_len,
                            "do_lowercase": self.config.do_lowercase,
                            "process_tokens_func": self.process_tokens
                            }
        if self.config.__dict__.get("word_truncate_len", None) is not None:
            transform_kwargs["word_truncate_len"] = self.config.word_truncate_len

        question_token_idxs, answer_token_idxs, _ = self.transform(
            instance = read_data,
            **transform_kwargs)
        wrapper_data = {"question": question_token_idxs, "answer": answer_token_idxs}
        return wrapper_data
    
    # get relevance confidence (return float)
    def infer(self, question, answer):
        model_input = self._preprocess(question, answer)
        model_output = self.model.infer(model_input)
        relevance_confidence = model_output[0][1].item()
        return relevance_confidence
    
    # get class id (return int)
    def infer_class(self, question, answer):
        model_input = self._preprocess(question, answer)
        model_output = self.model.infer(model_input)
        _, infer_output = torch.max(model_output, 1)
        infer_output = infer_output[0].item()
        return infer_output
    
    # get probability belong to different class (return list)
    def infer_proba(self, question, answer):
        model_input = self._preprocess(question, answer)
        model_output = self.model.infer(model_input)
        infer_output = model_output.tolist()[0]
        return infer_output

