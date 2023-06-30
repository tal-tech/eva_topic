#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import socket
from config import Config
from utility.g_logger import g_logger
import py_eureka_client.eureka_client as eureka_client

SERVER_PORT = 8004
HEART_BEAT_INTERVAL = 3

app_name = Config.APP_NAME
SERVER_HOST = Config.SERVER_HOST


def eureka_register():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        hostname = socket.gethostname()

        instance_id = "{}:{}:{}".format(hostname, app_name.lower(), SERVER_PORT)
        a, b = eureka_client.init(eureka_server=os.environ.get("EUREKA_URL") or
                                  "http://AILab:PaaS@eureka-dev.facethink.com/eureka/",
                                  app_name=app_name,
                                  # 当前组件的主机名，可选参数，如果不填写会自动计算一个，
                                  # 如果服务和 eureka 服务器部署在同一台机器，请必须填写，否则会计算出 127.0.0.1
                                  instance_id=instance_id,
                                  instance_port=SERVER_PORT,
                                  instance_host=SERVER_HOST,
                                  renewal_interval_in_secs=HEART_BEAT_INTERVAL,
                                  # 调用其他服务时的高可用策略，可选，默认为随机
                                  ha_strategy=eureka_client.HA_STRATEGY_RANDOM)
        return a, b
    except Exception as e:
        g_logger.error('eureka error:{}'.format(e))


def eureka_stop():
    eureka_client.stop()
