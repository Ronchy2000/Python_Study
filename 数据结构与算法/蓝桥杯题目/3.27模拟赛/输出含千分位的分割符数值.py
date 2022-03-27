# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 23:12
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 输出含千分位的分割符数值.py
# @Software: PyCharm
num = float(input())
#保留两位小数
print('%.2f'%num)
str = str(num)
print('len(str):',len(str))
flag1 = 0
flag2 = 0
if len(str) >11:
    str = str[0:len(str) - 11] + ',' + str[-11:]
    flag1 = 1
if len(str)-flag1 >8:
    str = str[0:len(str)- 8] + ',' + str[-8:]
    flag2 = 1
if len(str)-flag1-flag2 >5:
    str = str[0:len(str) -5] + ',' + str[-5:]
print(str)
print('len(str):',len(str))
