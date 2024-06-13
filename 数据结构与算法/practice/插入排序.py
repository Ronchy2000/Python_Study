# -*- coding: utf-8 -*-
# @Time    : 2022/2/25 13:11
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 插入排序.py
# @Software: PyCharm

def insertion_sort(arr):
    for index in range(1,len(arr)):
        #从第二个元素开始作为空隙，并把该位置的值保存 position为空隙的位置
        position = index
        temp_val = arr[index]
        #############
        while position>0 and arr[position-1]>temp_val:
            #temp_val 与左边的值比较,如果左边大，就把左边的值右移，空隙就变成了 temp_val左侧，索引为position-1
            arr[position] = arr[position-1]
            position = position-1
        #把保存的值放入到空隙中
        arr[position] = temp_val
