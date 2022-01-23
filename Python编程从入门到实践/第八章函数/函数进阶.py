# -*- coding: utf-8 -*-
# @Time    : 2021/3/21 16:26
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 函数进阶.py
# @Software: PyCharm
def fun2(*args):
    print(args)
def fun3(**args2):
    print(args2)

fun2(10,20,30,40,50,60)
fun3(a=11,b=22,c=33,d=44,e=55)