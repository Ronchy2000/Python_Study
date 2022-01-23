# -*- coding: utf-8 -*-
# @Time    : 2021/4/13 19:21
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 二分查找.py
# @Software: PyCharm
import random
#lst = [random.randint(1,10) for i in range(10)]

lst = range(0,10)

def locate(li,val):
    left = 0
    right = len(lst)-1
    while left <= right: #候选区有值
        mid = (left + right)//2
        if li[mid] == val:
            return mid
        elif li[mid] > val: #val在mid左侧
            right = mid - 1
        elif li[mid] < val: #val在mid右侧
            left = mid + 1
        else:
            return None
for i in lst:
    print(i)
print(locate(lst,6))