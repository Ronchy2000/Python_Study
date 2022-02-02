# -*- coding: utf-8 -*-
# @Time    : 2021/4/13 19:53
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 选择排序.py
# @Software: PyCharm
# 从小到大排序
'''
def select_sort_simple(li):
    li_new = []
    for i in range(len(li)):
        min_val = min(li)
        li_new.append(min_val)
        li.remove(min_val)
    return li_new

li = [3,2,4,1,5,6,8,7,9]
print(li)
# print(select_sort_simple(li))



def select_sort(li):
     for i in range( len(li)-1 ):   #i是第几趟
        #认为i即是最小的数
         for j in range(i+1,len(li)): #无序区
             if li[j] > li[i]:
                 li[j],li[i] = li[i],li[j]
print(select_sort(li))
'''


