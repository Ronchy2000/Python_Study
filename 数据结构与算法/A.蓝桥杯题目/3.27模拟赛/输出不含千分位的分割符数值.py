# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 23:45
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 输出不含千分位的分割符数值.py
# @Software: PyCharm
str1 = input().split(',')
print(str1)
ls =''
for i in range(len(str1)):
    ls = ls + str(str1[i])
print(ls)