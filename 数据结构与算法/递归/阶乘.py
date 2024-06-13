# -*- coding: utf-8 -*-
# @Time    : 2022/2/27 13:56
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 阶乘.py
# @Software: PyCharm

def factorial_recursive(n):
    if n == 0:
        return 1
    else:
        return n * factorial_recursive(n-1)

print(factorial_recursive(5))