# -*- coding: utf-8 -*-
# @Time    : 2022/2/27 14:04
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 快速指数.py
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
data = []
def fibonacci2(n):
    a , b = 0, 1
    for i in range(0,n):
        #print(a)
        data.append(a)
        a,b = b,a+b

fibonacci2(50)
if __name__ == '__main__':

    m = int(input())
    while m:
        m -= 1
        n = int(input())
        print(data[n])
