# -*- coding: utf-8 -*-
# @Time    : 2021/3/13 16:57
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 读取.py
# @Software: PyCharm
import json
access = []
with open('numbers.json') as f_obj:
    access = json.load(f_obj)
    print(access)