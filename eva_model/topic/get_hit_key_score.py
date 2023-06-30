#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2020-02-26
'''

import re
from .common import calc_coverage


def get_hit_key_score(question, answer):

    question, answer = question.lower(), answer.lower()
    question, answer = question.replace("i'm", "i am"), answer.replace("i'm", "i am")
    question, answer = question.replace("'s", ""), answer.replace("'s", "")

    new_que = ""
    state_words = ["fine", "happy", "good", "fun", "tired", "sad", "seek"]
    if "what your name" in question:
        new_que += "my name "
    if new_que != "":
        if "i am" in answer:
            for state_word in state_words:
                answer = answer.replace("i am " + state_word, state_word)
            answer = answer.replace("i am", "my name")
    if new_que != "" and ("how are you" in question or "how're you" in question):
        max_coverage = 0
        for state_word in state_words:
            question = new_que + state_word
            question = question.replace("_", " ")
            question = question.replace(".", "")
            filter_pattern = re.compile("\([a-zA-Z]+\)")
            question = re.sub(filter_pattern, "", question)
            question = " ".join(question.split("/"))
            clean_question = []
            for token in question.split():
                if token.isalpha():
                    clean_question.append(token)
            question = " ".join(clean_question)
            coverage = calc_coverage(answer, question)
            max_coverage = max(coverage, max_coverage)
            # print(coverage)
            # print(question, answer)
        coverage = max_coverage
    else:
        if new_que != "":
            question = new_que
        question = question.replace("_", " ")
        question = question.replace(".", "")
        filter_pattern = re.compile("\([a-zA-Z]+\)")
        question = re.sub(filter_pattern, "", question)
        question = " ".join(question.split("/"))
        clean_question = []
        for token in question.split():
            if token.isalpha():
                clean_question.append(token)
        question = " ".join(clean_question)
        coverage = calc_coverage(answer, question)
    return coverage

