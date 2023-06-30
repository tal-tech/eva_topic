#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-08
'''

import time
import logging
from functools import wraps
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


class Utils(object):

    def __init__(self):
        pass

    ''' Record function running time '''
    @staticmethod
    def timeit(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = function(*args, **kwargs)
            t1 = time.time()
            logging.info(
                'Function - {0} running time : {1} m - {2} s - {3:.3f} ms'.format(
                    function.__name__,
                    int((t1 - t0) // 60),
                    int((t1 - t0) % 60),
                    (t1 - t0 - int(t1 - t0)) * 1000))
            return result
        return wrapper
