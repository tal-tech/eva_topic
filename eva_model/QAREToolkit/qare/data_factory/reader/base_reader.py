#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-08
'''


class BaseReader(object):

    def __init__(self, *args, **kwargs):
        pass

    def read(self, *args, **kwargs):
        raise NotImplementedError
