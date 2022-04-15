# -*- coding: utf-8 -*-
# @Time    : 2022/2/21 19:41
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 快速指数.py
# @Software: PyCharm
'''
1 1 2 3 5 8 13 21 34 .....
求100以内的斐波那契数列
'''
a ,b = 1,1
print(a,end='\n')
print(b,end='\n')
while(1):
    #第三位
    c = a+b
    a = b
    b = c
    if c >=100 :
        break
    print(c,end='\n')

