#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import json
import base64
import traceback
from flask import Response
from utility.g_logger import g_logger


class Status:
    SUCCESS = 'success'
    PAR_ERROR = 'parameters error'
    BAD_REQ = 'bad request'
    MODEL_ERR = 'model error'


error_code = {
    'success': 20000,
    'parameters error': 0,
    'bad request': 0,
    'model error': 0,
}


def make_response(msg, data=None):
    if data is None:
        data = {}
    return Response(json.dumps({'code': error_code.get(msg), 'msg': msg, 'data': data}), status=200,
                    content_type='application/json')


def log_error(idx, error, detail=''):
    return g_logger.error("Request: {} -- {} -- : {}".format(idx, error, detail))


