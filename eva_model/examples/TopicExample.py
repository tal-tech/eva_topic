#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2020-03-05
'''

import os
import json
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)
import sys
sys.path.append(project_dir)
from topic.get_topic_score import get_topic_score
import json

def example():
    with open(os.path.join(base_dir, "./eva_model/examples/asr_result.json"), "r") as fr:
        asr_res = json.load(fr)
    question = "能够根据图片和提示描述自己的日常活动。"
    structure = "I usually ______ (do) at ______.(time)\n\nI can ______ at ______.\n\nST6-S80-E53-11-内容图"
    TopicRes = get_topic_score(asr_res, question, structure)

    return asr_res

if __name__ == "__main__":

    res = example()
    print(res)
