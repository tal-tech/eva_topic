#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-08-06
'''
import os
import sys
project_root_directory = os.path.dirname(
            os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))  # Project root directory
sys.path.append(project_root_directory)
from eva_model.QAREToolkit.qare.api.qare_api import QareAPI


# Load model based on config file
def example(sample_num = 20):

    import time
    qare_api = QareAPI()
    # print("start sleep")
    # time.sleep(50)
    # print("finish sleep")
    t0 = time.time()
    question = '''My birthday is on _______.
I usually have a birthday party with my friends and family.
We eat cake, play games and _______ together.
We always have fun together.

ST6-S122-E82-12-内容图'''
    answer = '''he is very thin funny three is drinks three scary and together we always have fun together'''
    for i in range(sample_num):
        if i % 10000 == 0:
            time.sleep(10)
        infer_result = qare_api.infer(question, answer)
    t1 = time.time()
    print(infer_result)
    print("average response time: %0.4f ms" % ((t1-t0)*1000 / sample_num))


# Dynamically load model
def example_dynamic(sample_num = 20):
    from eva_model.QAREToolkit.qare.api.qare_api import QareAPI
    from eva_model.QAREToolkit.qare.api.load_resource import load_resource

    model_name = "MWAN"
    dataset_name = "kst_chinese_grade123"

    (config, vocab, model, transform_instance,
     process_tokens, reader) = load_resource(
        model_name=model_name, dataset_name=dataset_name)

    import time
    qare_api = QareAPI(config=config,
                             vocab=vocab,
                             model=model,
                             transform_instance=transform_instance,
                             process_tokens=process_tokens,
                             reader=reader)


    t0 = time.time()
    question = "说一说长方体的侧面积怎么求解"
    answer = "长方体的侧面积是底面周长乘高。"
    for _ in range(sample_num):
        infer_result = qare_api.infer(question, answer)
    t1 = time.time()
    print(infer_result)
    print("average response time: %0.4f ms" % ((t1 - t0) * 1000 / sample_num))

if __name__ == "__main__":

    example(1)
