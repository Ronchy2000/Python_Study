# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 16:49
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 2019数字.py
# @Software: PyCharm
'''
1-2019中含有2 0 1 9 的数字  求和
'''
sum = 0
for i in range(1,2020):
    s = str(i)
    if '2' in s or '0' in s or '1' in s or '9' in s:
        sum += i
print(sum)