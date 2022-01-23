# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 14:01
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : test.py
# @Software: PyCharm
# s = input('birth:')
# print(s+'5')
# from 函数 import apple
# apple()

from turtle import *

def drawStar(x, y):
    pu()
    goto(x, y)
    pd()
    # set heading: 0
    seth(0)
    for i in range(5):
        fd(40)
        rt(144)

for x in range(0, 350, 50):
    drawStar(x, 0)

done()