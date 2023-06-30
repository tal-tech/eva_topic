#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2020-02-25
'''

import re
from .common import calc_coverage


def get_coverage_score(question, answer):

    clean_question = []
    for token in question.split():
        if token.isalpha():
            clean_question.append(token)
    question = " ".join(clean_question)
    coverage = calc_coverage(answer, question)
    return coverage


def get_free_talk_type(question, answer):
    short_answer_threshold = 5
    question, answer = question.lower(), answer.lower()
    question, answer = question.replace("i'm", "i am"), answer.replace("i'm", "i am")
    question, answer = question.replace("'s", ""), answer.replace("'s", "")

    new_que = question
    if "what your name" in question:
        new_que += "my name "
    if "how are you" in question or "how're you" in question:
        new_que += "i am fine"
    if new_que != "":
        question = new_que
    question = question.replace("_", " ")
    question = question.replace(".", "")
    filter_pattern = re.compile("\([a-zA-Z]+\)")
    question = re.sub(filter_pattern, "", question)
    question = " ".join(question.split("/"))

    ans_len = len(answer.split())
    if ans_len < short_answer_threshold:
        # 无效数据（过短），free talk值为-1
        free_talk_type = -1
        return free_talk_type

    que_len = len(question.split())

    coverage = get_coverage_score(question, answer)

    pure_free_talk_threshold = 0.2
    if coverage < pure_free_talk_threshold:
        # 学生回答脱离模版，完全进行自由发挥，free talk值为2
        free_talk_type = 2
        return free_talk_type

    alpha = ans_len / que_len / coverage
    median_alpha = 3.0

    if alpha < median_alpha:
        # 学生回答依赖题干给的模板，且空白位置为较短的填词，free talk值为0
        free_talk_type = 0
        return free_talk_type
    else:
        # 学生回答依赖题干给的模板，且空白位置为正常的填词，free talk值为 1
        free_talk_type = 1
        return free_talk_type
