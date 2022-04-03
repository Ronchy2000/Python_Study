# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 15:49
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 集合操作.py
# @Software: PyCharm
'''
求 A B 的公共元素

'''
A = [1,2,3,4,5,6,7]
B = [1,3,5,7]

#利用set

print('set(A) & set(B):',set(A) & set(B))

print('set(A) | set(B):',set(A) | set(B))

print('set(A) - set(B):',set(A) - set(B))

print('set(B) - set(A):',set(B) - set(A))

print('set(A) ^ set(B):',set(A) ^ set(B))

print('set(B) ^ set(A):',set(B) ^ set(A))