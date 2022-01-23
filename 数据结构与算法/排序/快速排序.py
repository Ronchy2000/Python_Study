# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 12:08
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 快速排序.py
# @Software: PyCharm

list = [33,4,65,45,15,10]

def quicksort(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        less = [i for i in array[1:] if i<=pivot]
        greater = [i for i in array[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)  #列表类型：[pivot]; 类似于"apple"str类型

print(quicksort(list))