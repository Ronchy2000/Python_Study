# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 22:15
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 最2数字.py
# @Software: PyCharm
ls = range(1,2022)
ls = map(str,ls)
cnt = 0
for i in ls:
    if '2' in i:
        cnt += 1
print(cnt)
