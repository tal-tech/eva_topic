#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import logging


class Config:
    BASE_FOLDER = os.path.abspath(os.path.dirname(__file__))

    LOG_LEVEL = logging.DEBUG
    if os.environ.get('LOG_LEVEL') != "DEBUG":
        LOG_LEVEL = logging.INFO

    TEMP_FOLDER = os.path.join(BASE_FOLDER, 'tmp')
    if not os.path.exists(TEMP_FOLDER):
        os.mkdir(TEMP_FOLDER)

    DEPLOY_ENV = os.environ.get('DEPLOY_ENV') or 'local'

    APP_NAME = os.environ.get('APP_NAME')
    SERVER_HOST = os.environ.get('SERVER_HOST')


if __name__ == '__main__':
    print(Config.TEMP_FOLDER)
