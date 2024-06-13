# -*- coding: utf-8 -*-
# @Time    : 2022/8/9 13:38
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : test.py.py
# @Software: PyCharm
import numpy as np

# u = [1,2,3,4]
# v = np.tile(u ,(3,1))
# print("v",v)
# print("u",u) # u 不变
#-------------------------

a = np.array([[1,2],[3,4]])
print(a)
b = a.flatten('F')
print(b)
