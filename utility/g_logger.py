#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import logging
from config import Config

g_logger = logging.getLogger(__name__)


def init_logger():
    h = logging.StreamHandler()
    h.setLevel(Config.LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s [%(thread)d] %(levelname)s %(name)s - %(message)s')
    h.setFormatter(formatter)
    g_logger.setLevel(Config.LOG_LEVEL)
    g_logger.addHandler(h)


init_logger()


if __name__ == '__main__':
    print(Config.LOG_LEVEL)
