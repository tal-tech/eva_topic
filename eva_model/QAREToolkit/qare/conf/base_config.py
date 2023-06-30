#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-09
'''

import os
import sys
import torch
import logging
from collections import OrderedDict


class BaseConfig(object):

    def __init__(self):

        ''' Common config items '''
        self.set_save_model_path()

    def set_save_model_path(self):
        # Project root directory
        project_root_directory = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))

        attribute_values = []
        values = []
        for attibute in OrderedDict(sorted(self.__dict__.items(), key=lambda t: t[0])):
            if str(attibute) in ["epochs", "learning_rate", "batch_size"]:
                # Pass model unrelated attributes
                continue
            attibute = str(attibute)
            value = "".join(str(self.__dict__[attibute]).split())
            attribute_values.append((attibute, value))
            values.append(value)

        self.attribute_values = attribute_values

        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("model name should be string and can not be empty!")
        if not self.dataset_name or not isinstance(self.dataset_name, str):
            raise ValueError("dataset name should be string and can not be empty!")

        # Append project directory to system
        self.project_root_directory = project_root_directory
        sys.path.append(os.path.dirname(project_root_directory))

        # Define save model path
        self.save_model_name = "_".join(values)[0:200]   # File name length limited
        save_model_directory = os.path.join(project_root_directory, "models", "save_model",
                                            "_".join(self.model_name.split() + self.dataset_name.split()))
        if not os.path.exists(save_model_directory):
            os.makedirs(save_model_directory)
        self.save_model_path = os.path.join(save_model_directory, self.save_model_name)

        # Define save vocab path
        save_vocab_directory = os.path.join(project_root_directory, "data_factory", "save_vocab",
                                            "_".join(self.model_name.split() + self.dataset_name.split()))
        if not os.path.exists(save_vocab_directory):
            os.makedirs(save_vocab_directory)
        self.save_vocab_path = os.path.join(save_vocab_directory,
                                            self.save_model_name + ".pkl")

        # Select computing resources to use ( GPU or CPU )
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")


    class ConstError(TypeError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't modify config const (%s)" % name)
        self.__dict__[name] = value
