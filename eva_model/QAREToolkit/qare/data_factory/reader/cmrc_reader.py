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


class CmrcReader(BaseReader):

    def __init__(self):
        self.tokenizer = ChineseTokenizer()

    @Utils.timeit
    def read(self, file_path):
        logging.info("Reading file at %s", file_path)
        instances = self._read_json(file_path)
        return instances

    def _read_json(self, file_path):
        with open(file_path, encoding="utf-8") as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        instances = []
        article_num = len(dataset)
        attack_interval = article_num // 3
        for article_id, article in enumerate(dataset):
            for paragraph in article['paragraphs']:
                answer = paragraph["context"]
                answer_tokens = self.tokenizer.word_tokenizer(answer)
                attack_paragraph = dataset[(article_id + attack_interval) % article_num]['paragraphs'][0]
                attack_answer = attack_paragraph["context"]
                attack_answer_tokens = self.tokenizer.word_tokenizer(attack_answer)
                label = 1
                attack_label = 0

                for question_answer in paragraph['qas']:
                    question = question_answer["question"].strip()
                    question_tokens = self.tokenizer.word_tokenizer(question)
                    id = question_answer['id']

                    instances.append(
                        self._make_instance(answer, answer_tokens,
                                              question, question_tokens,
                                              label, id)
                    )
                    instances.append(
                        self._make_instance(attack_answer, attack_answer_tokens,
                                            question, question_tokens,
                                            attack_label, id)
                    )
        return instances

    def _make_instance(self, answer, answer_tokens, question, question_tokens, label, id):

        return OrderedDict({
            "id": id,
            "question": question,
            "question_tokens": question_tokens,
            "answer": answer,
            "answer_tokens": answer_tokens,
            "label": label
        })