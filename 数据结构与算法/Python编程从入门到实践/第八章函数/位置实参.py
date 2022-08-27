# -*- coding: utf-8 -*-
# @Time    : 2021/3/21 16:07
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 位置实参.py
# @Software: PyCharm
def fun(a,b,c):
    print("a= ",a)
    print("b= ",b)
    print("c= ",c)

# fun(10,20,30)
# lst=[11,22,33]
# fun(*lst)
# fun (c= 7,b=9,a =6)
dic = {'b':111,'a':222,'c':333}
fun(**dic)