# -*- coding: utf-8 -*-
# @Time    : 2022/2/27 13:59
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 求和.py
# @Software: PyCharm
def sum(n):
    if n == 0:
        return 0
    else:
        return n+sum(n-1)

print(sum(3))