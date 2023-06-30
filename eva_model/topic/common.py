#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2020-02-17
'''

import re
from nltk.stem import WordNetLemmatizer

word_lemma = WordNetLemmatizer()

def regular_str2words(instr):

    instr = instr.replace("o'clock", "oclock")
    lemma_ans = []
    for w in instr.split():
        w = w.lower()
        lemma_w = word_lemma.lemmatize(w)
        lemma_ans.append(lemma_w)

    match_res = re.findall('[a-zA-Z]+', " ".join(lemma_ans))
    match_res = list(match_res)
    for i in range(len(match_res)):
        cur_str = match_res[i].lower()
        # abbr -> word
        if cur_str == "s":
            match_res[i] = "is"
        elif cur_str == "m":
            match_res[i] = "am"
        elif cur_str == "re":
            match_res[i] = "are"
    return match_res

# 计算source text 相对于 target text 覆盖率
def calc_coverage(source_text, target_text):
    target_text, source_text = target_text.strip().split(), source_text.strip().split()

    n1 = len(source_text)
    n2 = len(target_text)
    if target_text == source_text:
        return 1
    elif n1 == 0 or n2 == 0:
        return 0

    v0 = [0] * (n2 + 1)
    v1 = [0] * (n2 + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(n1):
        v1[0] = 0
        for j in range(n2):
            cost = 0 if target_text[j] == source_text[i] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1], v0[j] + cost)

        for j in range(len(v0)):
            v0[j] = v1[j]

    return 1.0 - v1[n2] / n2



def get_error(target, prediction):
    error = []
    relErro = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
        relErro.append(1 - prediction[i]/target[i])

    # print("Errors: ", error)
    # print(error)

    squaredError = []
    absError = []
    apeErro = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    for val in relErro:
        apeErro.append(abs(val))

    # print("Square Error: ", squaredError)
    # print("Absolute Value of Error: ", absError)
    #
    print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE

    from math import sqrt

    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE
    print("MAE = ", sum(absError) / len(absError))  # 平均绝对误差MAE
    print("MAPE = ", sum(apeErro) / len(absError))  # 平均相对误差MAPE