# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 20:58
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : python异常处理机制.py
# @Software: PyCharm
while True:
    try:
        a = int(input("Please input an integer:"))
        b = int(input("Please input an integer:"))
        answer = a/b
    except BaseException as error:
        print("Error！")
        print(error)
    else:
        print("the answer is",answer)
