#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-10
'''

import numpy as np
from eva_model.QAREToolkit.qare.data_factory.data_process_tools import filter_empty_tokens
from eva_model.QAREToolkit.qare.data_factory.data_process_tools import convert_tokens_to_padded_idxs
from eva_model.QAREToolkit.qare.data_factory.data_process_tools import convert_tokens_to_char_padded_idxs


''' 
Preprocessing of instance. 
'''
def transform_instance(instance, 
                       vocab,
                       process_tokens_func,
                       do_lowercase,
                       question_truncate_len, 
                       answer_truncate_len):

    question_tokens = instance['question_tokens']
    answer_tokens = instance['answer_tokens']

    if do_lowercase:
        question_tokens = list(map(str.lower, question_tokens))
        answer_tokens = list(map(str.lower, answer_tokens))

    question_tokens = process_tokens_func(question_tokens)
    answer_tokens = process_tokens_func(answer_tokens)

    question_tokens = filter_empty_tokens(question_tokens)[0:question_truncate_len]
    answer_tokens = filter_empty_tokens(answer_tokens)[0:answer_truncate_len]
    pad_token_idx = vocab.get_pad_idx()
    unk_token_idx = vocab.get_unk_idx()

    question_token_idxs = convert_tokens_to_padded_idxs(question_tokens, vocab.word2idx, question_truncate_len,
                                                        pad_token_idx, unk_token_idx)
    answer_token_idxs = convert_tokens_to_padded_idxs(answer_tokens, vocab.word2idx, answer_truncate_len,
                                                      pad_token_idx, unk_token_idx)
    label = instance["label"]

    return np.array(question_token_idxs), np.array(answer_token_idxs), np.array(label)


''' 
Preprocessing of instance. 
Different levels of granularity: char & word(en); single word & words(chinese)
'''
def transform_instance_to_cw(instance,
                             vocab,
                             process_tokens_func,
                             do_lowercase,
                             question_truncate_len,
                             answer_truncate_len,
                             word_truncate_len):

    question_tokens = instance['question_tokens']
    answer_tokens = instance['answer_tokens']

    if do_lowercase:
        question_tokens = list(map(str.lower, question_tokens))
        answer_tokens = list(map(str.lower, answer_tokens))

    question_tokens = process_tokens_func(question_tokens)
    answer_tokens = process_tokens_func(answer_tokens)

    question_tokens = filter_empty_tokens(question_tokens)[0:question_truncate_len]
    answer_tokens = filter_empty_tokens(answer_tokens)[0:answer_truncate_len]

    pad_token_idx = vocab.get_pad_idx()
    unk_token_idx = vocab.get_unk_idx()

    question_w_idxs = convert_tokens_to_padded_idxs(question_tokens, vocab.word2idx, question_truncate_len,
                                                        pad_token_idx, unk_token_idx)

    answer_w_idxs = convert_tokens_to_padded_idxs(answer_tokens, vocab.word2idx, answer_truncate_len,
                                                      pad_token_idx, unk_token_idx)

    question_chars_idxs = convert_tokens_to_char_padded_idxs(question_tokens, vocab.char2idx,
                                                             question_truncate_len, word_truncate_len,
                                                        pad_token_idx, unk_token_idx)
    answer_chars_idxs = convert_tokens_to_char_padded_idxs(answer_tokens, vocab.char2idx,
                                                             answer_truncate_len, word_truncate_len,
                                                        pad_token_idx, unk_token_idx)

    question_token_idxs = {"word": np.array(question_w_idxs),
                           "char": np.array(question_chars_idxs)}
    answer_token_idxs = {"word": np.array(answer_w_idxs),
                         "char": np.array(answer_chars_idxs)}
    label = np.array(instance["label"])

    return question_token_idxs, answer_token_idxs, label