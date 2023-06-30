#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-26
'''

from eva_model.QAREToolkit.qare.conf.base_config import BaseConfig


class ConfigBiDAF(BaseConfig):

    def __init__(self):
        ''' Custom model related config items '''
        self.model_name = "bidaf"
        self.dataset_name = "kst_math_plus_chinese"
        self.dropout = 0.4   # Dropout prob across the layers
        self.word_truncate_len = 8 # Words truncation length
        self.char_embedding_size = 50    # Single word embedding size(ch) or Char embedding size(en)
        self.char_channel_size = 100    # Char channel size for CNN
        self.char_channel_width = 5 # Char channel width for CNN
        self.word_embedding_size = 200  # Word embedding size (follow pretrained embedding size if pretrained embedding exists)
        self.hidden_size = (self.word_embedding_size + self.char_channel_size) // 2  # Hidden units number

        ''' Common config items '''
        self.epochs = 500    # Epochs of training model (not recorded in model name)
        self.learning_rate = 0.01    # Learning rate based on gradient descent (not recorded in model name)
        self.batch_size = 100   # Number of single input samples
        self.n_classes = 2  # Number of classes to be classified
        self.need_segment = True  # Chinese word segmentation
        self.do_lowercase = True    # Word to lowercase
        self.question_truncate_len = 90  # Question truncation length
        self.answer_truncate_len = 90  # Answer truncation length
        super(ConfigBiDAF, self).__init__()


config = ConfigBiDAF()