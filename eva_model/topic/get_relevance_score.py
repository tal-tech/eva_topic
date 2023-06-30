#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2020-02-25
'''

import os
import sys
project_root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory
sys.path.append(project_root_directory)
from eva_model.QAREToolkit.qare.api.qare_api import QareAPI
import re
from .common import calc_coverage


def get_coverage_score(question, answer):
    question, answer = question.lower(), answer.lower()
    question, answer = question.replace("i'm", "i am"), answer.replace("i'm", "i am")

    new_que = ""
    if "what's your name" in question:
        new_que = "my name"
    if "how are you" in question or "how're you" in question:
        new_que += "i am fine"
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

qare_api = QareAPI()

def get_relevance_score(question, answer):
    relevance_prob = qare_api.infer_proba(question, answer)[1]

    if relevance_prob < 0.5:
        coverage = get_coverage_score(question, answer)
        if coverage >= 0.35:
            relevance_prob = min(1.0, coverage * 2)
    elif relevance_prob >= 0.5 and relevance_prob < 0.8:
        relevance_prob *= 0.1
    if answer.lower().strip() == "null":
        relevance_prob = 0

    is_relevance = 0 if relevance_prob <= 0.5 else 1

    return is_relevance

