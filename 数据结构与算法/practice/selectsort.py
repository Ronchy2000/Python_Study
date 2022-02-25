# -*- coding: utf-8 -*-
# @Time    : 2022/2/25 12:35
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : selectsort.py
# @Software: PyCharm
#终于学会了选择排序，也搞清楚了与冒泡排序的区别

def selectionSort(arr):
    for i in range(len(arr)-1):
        minIndex = i
        for j in range(i+1,len(arr)):
            if arr[j] <arr[minIndex]:
                minIndex = j
        #交换是在内层循环结束后交换 ，如果放在内层循环里交换则是冒泡排序！
        if i !=minIndex:
            arr[i],arr[minIndex] = arr[minIndex],arr[i]
            print(arr)
    return arr
print('finished:',selectionSort([5,4,3,2,1]))
