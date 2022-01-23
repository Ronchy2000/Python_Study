# -*- coding: utf-8 -*-
# @Time    : 2021/4/13 18:56
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 冒泡排序.py
# @Software: PyCharm
import random
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
