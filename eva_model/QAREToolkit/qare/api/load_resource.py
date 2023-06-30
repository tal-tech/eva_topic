#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-08-08
'''

import os
from configparser import ConfigParser
from eva_model.QAREToolkit.qare.data_factory.vocabulary import Vocabulary
from eva_model.QAREToolkit.qare.data_factory.reader.general_reader import GeneralReader
from eva_model.QAREToolkit.qare.data_factory.reader.libu_reader import LibuReader

base_dir = os.path.dirname(os.path.abspath(__file__))
conf_file_path = os.path.join(base_dir, "conf.ini")

def get_api_config():
    config = ConfigParser()
    config.read(conf_file_path)
    model_name = config["DEFAULT"]["MODEL_NAME"]
    dataset_name = config["DEFAULT"]["DATASET_NAME"]

    return model_name, dataset_name

    
def load_resource(model_name, dataset_name):

    # import config and model
    if model_name == "MatchLSTM":
        from eva_model.QAREToolkit.qare.models.match_lstm import MatchLSTM as api_model

        if dataset_name == "kst_math":
            from eva_model.QAREToolkit.qare.conf.match_lstm.kst.math import config
        elif dataset_name == "kst_chinese":
            from eva_model.QAREToolkit.qare.conf.match_lstm.kst.chinese import config
        elif dataset_name == "kst_chinese_grade123":
            from eva_model.QAREToolkit.qare.conf.match_lstm.kst.chinese_grade123 import config
        elif dataset_name == "kst_math_plus_chinese":
            from eva_model.QAREToolkit.qare.conf.match_lstm.kst.chinese import config
        elif dataset_name == "cmrc":
            from eva_model.QAREToolkit.qare.conf.match_lstm.cmrc import config
        elif dataset_name == "squadv2":
            from eva_model.QAREToolkit.qare.conf.match_lstm.squadv2 import config

    elif model_name == "MWAN":
        from eva_model.QAREToolkit.qare.models.mwan import MWAN as api_model

        if dataset_name == "kst_math":
            from eva_model.QAREToolkit.qare.conf.mwan.kst.math import config
        elif dataset_name == "kst_chinese":
            from eva_model.QAREToolkit.qare.conf.mwan.kst.chinese import config
        elif dataset_name == "kst_chinese_grade123":
            from eva_model.QAREToolkit.qare.conf.mwan.kst.chinese_grade123 import config
        elif dataset_name == "kst_math_plus_chinese":
            from eva_model.QAREToolkit.qare.conf.mwan.kst.chinese import config
        elif dataset_name == "cmrc":
            from eva_model.QAREToolkit.qare.conf.mwan.cmrc import config
        elif dataset_name == "squadv2":
            from eva_model.QAREToolkit.qare.conf.mwan.squadv2 import config

    elif model_name == "StructuredSelfAttentiveNet":
        from eva_model.QAREToolkit.qare.models.ssan import StructuredSelfAttentiveNet as api_model

        if dataset_name == "kst_math":
            from eva_model.QAREToolkit.qare.conf.ssan.kst.math import config
        elif dataset_name == "kst_chinese":
            from eva_model.QAREToolkit.qare.conf.ssan.kst.chinese import config
        elif dataset_name == "kst_chinese_grade123":
            from eva_model.QAREToolkit.qare.conf.ssan.kst.chinese_grade123 import config
        elif dataset_name == "kst_math_plus_chinese":
            from eva_model.QAREToolkit.qare.conf.ssan.kst.chinese import config
        elif dataset_name == "cmrc":
            from eva_model.QAREToolkit.qare.conf.ssan.cmrc import config
        elif dataset_name == "squadv2":
            from eva_model.QAREToolkit.qare.conf.ssan.squadv2 import config

    elif model_name == "BiDAF":
        from eva_model.QAREToolkit.qare.models.bidaf import BiDAF as api_model

        if dataset_name == "kst_math":
            from eva_model.QAREToolkit.qare.conf.bidaf.kst.math import config
        elif dataset_name == "kst_chinese":
            from eva_model.QAREToolkit.qare.conf.bidaf.kst.chinese import config
        elif dataset_name == "kst_chinese_grade123":
            from eva_model.QAREToolkit.qare.conf.bidaf.kst.chinese_grade123 import config
        elif dataset_name == "kst_math_plus_chinese":
            from eva_model.QAREToolkit.qare.conf.bidaf.kst.chinese import config
        elif dataset_name == "cmrc":
            from eva_model.QAREToolkit.qare.conf.bidaf.cmrc import config
        elif dataset_name == "squadv2":
            from eva_model.QAREToolkit.qare.conf.bidaf.squadv2 import config
        elif dataset_name == "libu":
            from eva_model.QAREToolkit.qare.conf.bidaf.libu.libu import config

    elif model_name == "QANet":
        from eva_model.QAREToolkit.qare.models.qanet import QANet as api_model

        if dataset_name == "kst_math":
            from eva_model.QAREToolkit.qare.conf.qanet.kst.math import config
        elif dataset_name == "kst_chinese":
            from eva_model.QAREToolkit.qare.conf.qanet.kst.chinese import config
        elif dataset_name == "kst_chinese_grade123":
            from eva_model.QAREToolkit.qare.conf.qanet.kst.chinese_grade123 import config
        elif dataset_name == "kst_math_plus_chinese":
            from eva_model.QAREToolkit.qare.conf.qanet.kst.chinese import config
        elif dataset_name == "cmrc":
            from eva_model.QAREToolkit.qare.conf.qanet.cmrc import config
        elif dataset_name == "squadv2":
            from eva_model.QAREToolkit.qare.conf.qanet.squadv2 import config
    
    elif model_name == "RNet":
        from eva_model.QAREToolkit.qare.models.rnet import RNet as api_model

        if dataset_name == "kst_math":
            from eva_model.QAREToolkit.qare.conf.rnet.kst.math import config
        elif dataset_name == "kst_chinese":
            from eva_model.QAREToolkit.qare.conf.rnet.kst.chinese import config
        elif dataset_name == "kst_chinese_grade123":
            from eva_model.QAREToolkit.qare.conf.rnet.kst.chinese_grade123 import config
        elif dataset_name == "kst_math_plus_chinese":
            from eva_model.QAREToolkit.qare.conf.rnet.kst.chinese import config
        elif dataset_name == "cmrc":
            from eva_model.QAREToolkit.qare.conf.rnet.cmrc import config
        elif dataset_name == "squadv2":
            from eva_model.QAREToolkit.qare.conf.rnet.squadv2 import config
    
    else:
        raise ValueError("model_name:{} error!".format(model_name))

    # import transform method
    if model_name in ["MatchLSTM", "MWAN", "StructuredSelfAttentiveNet"]:
        from eva_model.QAREToolkit.qare.data_factory.transform_instance import transform_instance
    elif model_name in ["BiDAF", "QANet", "RNet"]:
        from eva_model.QAREToolkit.qare.data_factory.transform_instance import transform_instance_to_cw as transform_instance
    else:
        raise ValueError("model_name:{} error!".format(model_name))

    # import preprocess method
    if dataset_name in ["kst_math", "kst_chinese", "kst_chinese_grade123", "kst_math_plus_chinese", "cmrc"]:
        from eva_model.QAREToolkit.qare.data_factory.data_process_tools import process_chinese_tokens as process_tokens
    elif dataset_name in ["squadv2", "libu"]:
        from eva_model.QAREToolkit.qare.data_factory.data_process_tools import filter_special_tokens as process_tokens
    else:
        raise ValueError("dataset_name:{} error!".format(dataset_name))

    save_vocab_path = config.save_vocab_path
    vocab = Vocabulary(config.device)
    vocab.load(save_vocab_path)

    model = api_model(vocab, config)
    model.load_model(config.save_model_path)
    if dataset_name in ["libu"]:
        reader = LibuReader(config.need_segment)
    else:
        reader = GeneralReader(config.need_segment)
    
    return config, vocab, model, transform_instance, process_tokens, reader