#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2020-03-11
'''

import os
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
import re
from .get_free_talk_type import get_free_talk_type
from .merge_topic_score import merge_topic_score


def get_topic_score(asr_result_list, question, structure):
    question, structure = question.lower(), structure.lower()
    merge_que = ""
    eng_pattern = re.compile("[a-zA-Z]+")
    word_list = re.findall(eng_pattern, question)

    for w in word_list:
        if w != "topic" and w not in structure:
            merge_que += w + " "
    merge_que += structure

    stu_ans = " ".join([asr_result["result"] for asr_result in asr_result_list])

    FreeTalkType = get_free_talk_type(structure, stu_ans)
    ThemeScore = merge_topic_score(merge_que, structure, stu_ans)
    ThemeProbability = ThemeScore / 5

    TopicRes = {
        "ThemeProbability": ThemeProbability,
        "ThemeScore": ThemeScore,
        "FreeTalkType": FreeTalkType
    }
    return TopicRes
