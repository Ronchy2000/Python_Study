# -*- coding: utf-8 -*-
# @Time    : 2021/4/12 22:22
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 顺序查找.py
# @Software: PyCharm

#Linear Search

lst = ['Apple','iphone11','iphone 11 pro','ipad2019']
# print(list(enumerate(li)))

def linear_search(li,val):
    for i,v in enumerate(li):
        if v ==val:
            return i
    else:
        return None

print(linear_search(lst,'ipad2019'))