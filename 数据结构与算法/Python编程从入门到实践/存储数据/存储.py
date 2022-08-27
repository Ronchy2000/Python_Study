# -*- coding: utf-8 -*-
# @Time    : 2021/3/13 16:48
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 存储.py
# @Software: PyCharm
import json

numbers = [2,3,5,7,11,13]

filename = 'numbers.json'
with open(filename,'w') as f_obj:
    json.dump(numbers,f_obj)