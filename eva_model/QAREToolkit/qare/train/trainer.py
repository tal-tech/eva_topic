#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-11
'''

import logging
from eva_model.QAREToolkit.qare.train.evaluator import Evaluator
from eva_model.QAREToolkit.qare.utils.utils import Utils
from eva_model.QAREToolkit.qare.train.utils4train import Utils4Train


class Trainer(object):

    def __init__(self):
        pass

    @staticmethod
    def _train_and_evaluate(model,
                            train_batch_data,
                            eval_batch_data,
                            device,
                            save_model_path,
                            epochs,
                            monitor,
                            log_verbose):
        '''
        :param model:
        :param train_batch_data:
        :param eval_batch_data:
        :param device:
        :param save_model_path:
        :param epochs:
        :param monitor: accuracy, precision, recall, f1, auc
        :param log_verbose: True, Fasle
        :return:
        '''
        logging.info("Begin training model.")

        loss_function = model.loss_function()
        optimizer_, optimizer_kwargs = model.optimizer()
        optimizer = optimizer_(model.parameters(), **optimizer_kwargs)


        train_loss_list = []
        train_detail_loss_list = []

        # Measure initial model
        logging.info('Initial evaluation result.')
        (eval_metrics, eval_loss, eval_pred_class_id, eval_pred_prob, eval_true_labels
         ) = Evaluator._evaluate(model, eval_batch_data, device)
        accuracy, precision, recall, f1, auc = Evaluator._get_eval_metrics(
            eval_pred_class_id, eval_pred_prob, eval_true_labels)
        Evaluator._log_eval_result(accuracy, precision, recall, f1, auc)
        best_eval_monitor = eval_metrics[monitor]
        model.save_model(save_model_path)

        for epoch in range(epochs):
            train_loss, train_iter_loss_list = Trainer._train(model, epoch, train_batch_data,
               optimizer, loss_function, device, log_verbose)

            train_loss_list.append(train_loss)
            train_detail_loss_list += train_iter_loss_list

            if not log_verbose:
                logging.info('Epoch:{0:d}, Train Loss:{1}'.format(epoch + 1, train_loss))

            (eval_metrics, eval_loss, eval_pred_class_id, eval_pred_prob, eval_true_labels
             ) = Evaluator._evaluate(model, eval_batch_data, device)
            logging.info('Epoch:{0:d}, Eval Loss:{1}'.format(epoch + 1, eval_loss))

            if eval_metrics[monitor] > best_eval_monitor:
                best_eval_monitor = eval_metrics[monitor]
                model.save_model(save_model_path)
                logging.info('Best eval {0} now become{1}. '.format(monitor, best_eval_monitor))
                accuracy, precision, recall, f1, auc = Evaluator._get_eval_metrics(
                    eval_pred_class_id, eval_pred_prob, eval_true_labels)
                Evaluator._log_eval_result(accuracy, precision, recall, f1, auc)

        return train_loss_list, train_detail_loss_list

    @staticmethod
    @Utils.timeit
    def _train(model, epoch, train_batch_data,
               optimizer, loss_function, device, log_verbose):

        train_loss = 0
        train_size = len(train_batch_data)
        train_iter_loss_list = []

        if log_verbose:
            log_times_per_epoch = 5
            log_interval_per_epoch = max(1, train_size // (log_times_per_epoch * train_batch_data.get_batch_size()))

        for train_iter, (train_batch_question, train_batch_answer, train_batch_label) in enumerate(
                train_batch_data.get_batch_generator()):
            optimizer.zero_grad()

            model = model.to(device)
            model.train()
            train_batch_answer, train_batch_question, train_batch_label = (
                Utils4Train.to_device(train_batch_answer, device),
                Utils4Train.to_device(train_batch_question, device),
                Utils4Train.to_device(train_batch_label, device)
            )
            train_batch_pred = model(train_batch_question, train_batch_answer)
            train_iter_loss = loss_function(train_batch_pred, train_batch_label)
            train_iter_loss.backward()
            Utils4Train.check_param(model)
            optimizer.step()
            train_iter_loss = train_iter_loss.item()

            train_iter_loss_list.append(train_iter_loss)
            train_loss += train_iter_loss

            if log_verbose and (train_iter + 1) % log_interval_per_epoch == 0:
                logging.info('Epoch:{0:d}, Iteration:{1:d}, Train Loss:{2:.4f}'.format(
                    epoch + 1, train_iter + 1, train_iter_loss))

        return train_loss, train_iter_loss_list



