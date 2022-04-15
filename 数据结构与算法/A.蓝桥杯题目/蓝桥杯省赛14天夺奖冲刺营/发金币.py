# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 16:24
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 发金币.py
# @Software: PyCharm
n = int(input())
j,k=0,0
sum = 0
for i in range(1,n+1):
    if j == k:
        j =0
        k+=1
    j += 1
    k += 1
    sum+=k

print(sum)
