#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-29
'''

import re


''' Filter empty token in token list. '''
def filter_empty_tokens(tokens):

    def not_empty(s):
        return s and s.strip()

    return list(filter(not_empty, tokens))

''' Filter special token. '''
def filter_special_token(token):
    token = re.sub("[\_\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%]", "", token)
    return token

''' Filter special token in token list. '''
def filter_special_tokens(tokens):
    tokens = list(map(filter_special_token, tokens))
    return filter_empty_tokens(tokens)

''' Mapping tokens to token idx sequence. '''
def convert_tokens_to_padded_idxs(tokens,
                                  word2idx,
                                  truncate_len,
                                  pad_idx,
                                  unk_idx):

    token_idxs = [word2idx.get(token, unk_idx) for token in tokens]
    token_num = len(token_idxs)
    if token_num < truncate_len:
        token_idxs = token_idxs + [pad_idx for _ in range(truncate_len - token_num)]
    else:
        token_idxs = token_idxs[0:truncate_len]
    if token_num == 0:
        token_idxs[0] = 1
    return token_idxs


''' Mapping tokens to char idx sequence. '''
def convert_tokens_to_char_padded_idxs(tokens,
                                       char2idx,
                                       tokens_truncate_len,
                                       token_tuncate_len,
                                       pad_idx,
                                       unk_idx):
    token_num = len(tokens)
    char_idxs = []
    for i in range(tokens_truncate_len):
        if i < token_num:
            token = tokens[i]
            chars = list(token)
            _char_idxs = convert_tokens_to_padded_idxs(chars,
                                                       char2idx,
                                                       token_tuncate_len,
                                                       pad_idx,
                                                       unk_idx)
        else:
            _char_idxs = [pad_idx] * token_tuncate_len
        char_idxs.append(_char_idxs)
    return char_idxs



# Mapper from Arabic to Chinese
Arabic2Chinese = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}

''' Process special Chinese token. '''
def process_chinese_token(token):

    token = re.sub("[a-zA-Z\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%]", "", token)
    trans_token = ""
    for char in token:
        char = Arabic2Chinese[char] if char in Arabic2Chinese else char
        trans_token += char
    return trans_token

''' Process special Chinese tokens. '''
def process_chinese_tokens(tokens):
    tokens = list(map(process_chinese_token, tokens))
    return filter_empty_tokens(tokens)
