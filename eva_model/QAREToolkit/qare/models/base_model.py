#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-09
'''

import os
import torch
from eva_model.QAREToolkit.qare.utils.utils import Utils
from eva_model.QAREToolkit.qare.train.trainer import Trainer
from eva_model.QAREToolkit.qare.train.evaluator import Evaluator
from eva_model.QAREToolkit.qare.train.infer import Infer


class BaseModel(torch.nn.Module):

    def __init__(self, vocab, config):
        super(BaseModel, self).__init__()
        self.vocab = vocab
        self.config = config

    def loss_function(self):
        raise NotImplementedError

    def optimizer(self):
        raise NotImplementedError

    def forward(self, answer, question):
        '''
        :param answer: (batch, answer_len)
        :param question: (batch, question_len)
        :return:
        '''
        raise NotImplementedError

    def save_model(self, save_model_path):
        torch.save(self.cpu().state_dict(), save_model_path)

    def load_model(self, save_model_path):
        if os.path.exists(save_model_path):
            self.load_state_dict(torch.load(save_model_path))
            print("load model from {}.".format(save_model_path))
        else:
            print("fail to load model from {}.".format(save_model_path))

    @Utils.timeit
    def train_and_evaluate(self,
                           train_data,
                           eval_data,
                           device,
                           save_model_path,
                           epochs,
                           monitor = "accuracy",
                           log_verbose = False):
        '''
        :param train_data:
        :param eval_data:
        :param device:
        :param save_model_path:
        :param epochs:
        :param monitor: accuracy, precision, recall, f1, auc
        :param log_verbose: True, Fasle(defalt)
        :return: train_loss_list, train_detail_loss_list
        '''
        return Trainer._train_and_evaluate(self,
                                           train_data,
                                           eval_data,
                                           device,
                                           save_model_path,
                                           epochs,
                                           monitor = monitor,
                                           log_verbose = log_verbose)

    @Utils.timeit
    def evaluate(self, eval_data, device, log_result = False):
        '''
        :param eval_data:
        :param device:
        :param log_result: True, Fasle(defalt)
        :return: eval_metrics, eval_loss, eval_pred_class_id, eval_pred_prob
        '''
        return Evaluator._evaluate(self, eval_data, device, log_result = log_result)

    def infer(self, inference_data):
        '''
        :param inference_data:
        :return: inference_result
        '''
        return Infer._inference(self, inference_data)