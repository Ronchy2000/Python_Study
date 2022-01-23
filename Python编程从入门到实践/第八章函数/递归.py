# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 11:47
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 递归.py
# @Software: PyCharm

def countdown(i):
    print(i)
    if i<=0:
        return
    else:
        countdown(i+1)

countdown(1)