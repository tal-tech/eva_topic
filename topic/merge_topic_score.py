#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2020-03-03
'''

from .get_relevance_score import get_relevance_score
from .get_hit_key_score import get_hit_key_score

def merge_topic_score(merge_que_struct, structure, answer):
    qare_score = get_relevance_score(merge_que_struct, answer)
    hit_key_score = get_hit_key_score(structure, answer)
    if qare_score == 0 or hit_key_score < 0.15:
        return 1
    if hit_key_score < 0.35:
        return 2
    elif hit_key_score > 0.6:
        return 5
    elif hit_key_score > 0.55:
        return 4
    else:
        return 3
