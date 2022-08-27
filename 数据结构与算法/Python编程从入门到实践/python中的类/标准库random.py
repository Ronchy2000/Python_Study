# -*- coding: utf-8 -*-
# @Time    : 2021/3/13 13:40
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 标准库random.py
# @Software: PyCharm
from random import randint

x = 0
while x < 9:
    x = randint(1,10)
    print(x)