# -*- coding: utf-8 -*-
# @Time    : 2022/2/27 14:04
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 斐波那契数列.py
# @Software: PyCharm

#时间复杂度 O(2^n)
# def fibonacci(n):
#     if n <= 2:
#         return 1
#     return fibonacci(n-2)+fibonacci(n-1)
#
# print(fibonacci(40))

#斐波那契额数列用用递归并不好



#O（N）
def fibonacci2(n):
    a , b = 1, 1
    for i in range(1,n+1):
        a,b = b,a+b
        print(a)
fibonacci2(5)
