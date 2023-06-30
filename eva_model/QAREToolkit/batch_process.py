#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-07-25
'''

import os
import re


def change(top_dir):
    for dir_path,subpaths,files in os.walk(top_dir, False):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(dir_path, file)
                print(file_path)
                with open(file_path, "r") as fr:
                    content = fr.read()
                    content = content.replace("TAL-AI", "TAL-AI")
                with open(file_path, "w") as fw:
                    fw.write(content)


if __name__ == "__main__":
    html_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__))
    )
    change(html_path)