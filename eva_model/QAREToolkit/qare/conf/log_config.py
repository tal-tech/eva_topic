#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-27
'''

import os
import logging


class LogConfig(object):

    def __init__(self):
        pass

    @staticmethod
    def config_log(model_config):
        # Define Log Directory
        model_name = "_".join(model_config.model_name.strip().split())
        log_directory = os.path.join(model_config.project_root_directory, "log", model_name, model_config.dataset_name)
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        # Define Log Filename
        log_filename = os.path.join(log_directory, model_config.save_model_name)
        logging.basicConfig(filename=log_filename, level=logging.INFO, filemode='w',
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        # Log out model parameters
        attri_names = ""
        for idx, attibute_value in enumerate(model_config.attribute_values):
            attri_names += "\n" + str(idx) + ". " + str(attibute_value[0]) + ": " + str(attibute_value[1])
        logging.info("\nmodel parameters: " + attri_names)