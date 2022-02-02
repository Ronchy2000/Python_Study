# -*- coding: utf-8 -*-
# @Time    : 2021/4/13 18:56
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 冒泡排序.py
# @Software: PyCharm
import random
'''
def sort(lst):
    for i in range(len(lst)-1):
        exchange = False  #标志位
        for j in range(len(lst)-i-1):
            if lst[j] > lst[j+1]:
                lst[j],lst[j+1] = lst[j+1],lst[j]  #同时交换两个数
                exchange = True
        print(lst)
        if (exchange == False):  #若未交换，则返回
            return
# lst =[random.randint(0,1000) for i in range(50)]
lst = [9,8,7,0,1,2,3,4,5,6]
print(lst)
sort(lst)
'''

def bubble_sort(list):
    unsorted_until_index = len(list) - 1 #该索引之前的数据都没排过序
    sorted = False #用来记录数组是否已完全排好序

    while not sorted:
        sorted = True
        for i in range(unsorted_until_index):
            if list[i]> list[i+1]:
                sorted = False
                list[i],list[i+1] = list[i+1],list[i]
        unsorted_until_index = unsorted_until_index - 1
list = [65,55,45,35,25,15,10]
bubble_sort(list)
print(list)