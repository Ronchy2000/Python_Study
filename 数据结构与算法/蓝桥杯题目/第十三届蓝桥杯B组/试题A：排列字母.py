# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 14:09
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 试题A：排列字母.py
# @Software: PyCharm
str = input()
str = list(str)
tmp = [ord(i) for i in str]
# print(tmp)
tmp = sorted(tmp)
ans = [chr(i) for i in tmp]
print(ans)