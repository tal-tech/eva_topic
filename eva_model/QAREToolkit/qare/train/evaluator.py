#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-17
'''

import logging
import torch
from eva_model.QAREToolkit.qare.train.utils4train import Utils4Train


class Evaluator(object):

    def __init__(self):
        pass

    @staticmethod
    def _evaluate(model, eval_batch_data, device, log_result=False):
        eval_pred_class_id = []
        eval_pred_prob = []
        eval_true_labels = []
        eval_loss = 0
        loss_function = model.loss_function()
        with torch.no_grad():
            model.eval()
            Utils4Train.check_param(model)
            for eval_iter, (eval_batch_question, eval_batch_answer, eval_batch_label) in enumerate(
                    eval_batch_data.get_batch_generator()):
                model = model.to(device)
                model.eval()
                eval_true_labels += eval_batch_label.cpu().tolist()
                eval_batch_question, eval_batch_answer, eval_batch_label = (
                    Utils4Train.to_device(eval_batch_question, device),
                    Utils4Train.to_device(eval_batch_answer, device),
                    Utils4Train.to_device(eval_batch_label, device)
                )

                eval_batch_pred = model(eval_batch_question, eval_batch_answer)
                iter_eval_loss = loss_function(eval_batch_pred, eval_batch_label)
                eval_loss += iter_eval_loss.item()
                _, eval_batch_pred_class_id = torch.max(eval_batch_pred, 1)
                eval_pred_class_id += eval_batch_pred_class_id.cpu().tolist()
                eval_pred_prob += eval_batch_pred.cpu().tolist()

        eval_metrics = dict()
        (eval_metrics["accuracy"], eval_metrics["precision"], eval_metrics["recall"],
         eval_metrics["f1"], eval_metrics["auc"]) = Evaluator._get_eval_metrics(eval_pred_class_id, eval_pred_prob, eval_true_labels)

        if log_result:
            Evaluator._log_eval_result(eval_metrics["accuracy"], eval_metrics["precision"], eval_metrics["recall"],
                                    eval_metrics["f1"], eval_metrics["auc"])
        return eval_metrics, eval_loss, eval_pred_class_id, eval_pred_prob, eval_true_labels

    @staticmethod
    def _get_eval_metrics(eval_pred_class_id, eval_pred_prob, eval_true_labels):

        accuracy, precision, recall, f1, auc = Utils4Train.get_metrics(eval_pred_class_id,
                                                                 eval_pred_prob, eval_true_labels)
        return accuracy, precision, recall, f1, auc

    @staticmethod
    def _log_eval_result(accuracy, precision, recall, f1, auc):
        logging.info('\tAcc:{0:.4f}, Precision:{1:.4f}, Recall:{2:.4f}, F1:{3:.4f}, AUC:{4:.4f} '.format(
            accuracy, precision, recall, f1, auc))