#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-08
'''

import json
import logging
from collections import OrderedDict
from eva_model.QAREToolkit.qare.data_factory.reader.tokenizer import ChineseTokenizer
from eva_model.QAREToolkit.qare.data_factory.reader.base_reader import BaseReader
from eva_model.QAREToolkit.qare.utils.utils import Utils


LABEL_PLACEHOLDER = 9

class GeneralReader(BaseReader):

    def __init__(self, need_segment, tokenizer = ChineseTokenizer()):
        self.tokenizer = tokenizer
        self.need_segment = need_segment

    @Utils.timeit
    def read(self, file_path):
        '''
         Read data from json file (demand json format as follows)
            general json file format:
                { "data":
                    [
                        {
                          "id": int,
                          "question": string,
                          "answer": string,
                          "label": int
                        }, ...
                    ]
                }
        '''
        logging.info("Reading file at {}".format(file_path))
        instances = self._read_json_file(file_path)
        return instances

    def _read_json_file(self, file_path):
        with open(file_path, encoding="utf-8") as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
        instances = []
        for data in dataset:
            if {"id", "question", "answer", "label"} > set(data.keys()):
                raise AttributeError(('''dataset lack of attribute, 
                    should include ["id", "question", "answer", "label"]'''))
            answer = data["answer"].strip()
            answer_tokens = self.tokenizer.word_tokenizer(answer, self.need_segment)

            question = data["question"].strip()
            question_tokens = self.tokenizer.word_tokenizer(question, self.need_segment)

            label = int(data["label"])
            id = data["id"]

            instances.append(self._make_instance(id,
                                      question, question_tokens,
                                      answer, answer_tokens,
                                      label)
                             )
        return instances

    def _make_instance(self,
                       id,
                       question,
                       question_tokens,
                       answer,
                       answer_tokens,
                       label):
        return OrderedDict({
            "id": id,
            "question": question,
            "question_tokens": question_tokens,
            "answer": answer,
            "answer_tokens": answer_tokens,
            "label": label
        })

    def read_json_data(self, data):
        '''
         Read data from json format data (demand json format as follows)
            json data format:
                {
                  "question": string,
                  "answer": string,
                }
        '''
        if {"question", "answer"} > set(data.keys()):
            raise AttributeError(('''request data lack of attribute, 
                should include ["question", "answer"]'''))
        answer = data["answer"].strip()
        answer_tokens = self.tokenizer.word_tokenizer(answer, self.need_segment)

        question = data["question"].strip()
        question_tokens = self.tokenizer.word_tokenizer(question, self.need_segment)
        return self._make_request_instance(question, question_tokens,
                                           answer, answer_tokens)

    def _make_request_instance(self,
                               question,
                               question_tokens,
                               answer,
                               answer_tokens):

        return OrderedDict({
            "question": question,
            "question_tokens": question_tokens,
            "answer": answer,
            "answer_tokens": answer_tokens,
            "label": LABEL_PLACEHOLDER
        })