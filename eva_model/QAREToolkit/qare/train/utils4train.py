#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-29
'''

import torch
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import logging


class Utils4Train(object):

    def __init__(self):
        pass

    ''' Get metrics to evaluate model '''
    @staticmethod
    def get_metrics(hat_labels, hat_labels_prob, true_labels, average_method = "weighted"):
        hat_labels_prob = np.array(hat_labels_prob)
        n_classes = hat_labels_prob.shape[1]
        true_label_mat = label_binarize(true_labels, classes=list(range(n_classes)))
        hat_labels_prob = hat_labels_prob[:, 1] if n_classes == 2 else hat_labels_prob

        if n_classes == 2:
            precision = precision_score(true_labels, hat_labels, average=average_method)
            recall = recall_score(true_labels, hat_labels, average=average_method)
            f1 = f1_score(true_labels, hat_labels, average=average_method)
            accuracy = accuracy_score(true_labels, hat_labels)
            auc = roc_auc_score(true_label_mat, hat_labels_prob, average=average_method)
        else:
            precision = precision_score(true_labels, hat_labels, average='macro')
            recall = recall_score(true_labels, hat_labels, average='macro')
            f1 = f1_score(true_labels, hat_labels, average='macro')
            accuracy = accuracy_score(true_labels, hat_labels)
            auc = roc_auc_score(true_label_mat, hat_labels_prob, average='macro')

        return accuracy, precision, recall, f1, auc

    ''' Check model parameters,write to log '''
    @staticmethod
    def check_param(model, switch=False):

        def deep_trace(np_array, n_deep):
            if n_deep == 0:
                return np_array
            if len(np_array.shape) < 1:
                raise ValueError
            return deep_trace(np_array[0], n_deep - 1)

        if not switch:
            return

        params = list(model.named_parameters())
        logging.info("*" * 40)
        logging.info("params num:" + str(len(params)))
        n_sample = 4
        for name, param in model.named_parameters():
            if param.requires_grad:
                logging.info(name)
                data = param.data.cpu().numpy()
                data_dim = len(data.shape)
                logging.info(
                    "data:" + str(list(deep_trace(data, data_dim - 1)[0:n_sample])) + " - shape:" + str(data.shape))
                grad = param.grad
                if grad is not None:
                    grad = grad.cpu().numpy()
                    grad_dim = len(grad.shape)
                    logging.info(
                        "grad:" + str(list(deep_trace(grad, grad_dim - 1)[0:n_sample])) + " - shape:" + str(grad.shape))
        logging.info("*" * 40)

    ''' Put data to device '''
    @staticmethod
    def to_device(data, device):
        if torch.is_tensor(data):
            data2device = data.to(device)
        elif isinstance(data, dict):
            data2device = dict()
            for (k, v) in data.items():
                if torch.is_tensor(v):
                    data2device[k] = v.to(device)
                else:
                    data2device[k] = Utils4Train.to_device(v, device)
        else:
            raise TypeError("Can't push data to device!")
        return data2device
