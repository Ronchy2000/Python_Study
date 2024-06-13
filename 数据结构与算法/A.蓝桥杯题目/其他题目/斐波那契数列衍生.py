# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 16:52
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 斐波那契数列衍生.py
# @Software: PyCharm
'''
给定数列1，1，1，3，5，9，17，....
从第四项开始，每项都是前3项的和。求第20190324项的最后4位数字
'''
#不用递归：O（2^n）
#用线性数组
a,b,c = 1,1,1
for i in range(1,20190325):
    #第四位数
    d = (a+b+c)%10000
    #依次‘推’
    a = b
    b = c
    c = d
    # print(a,b,c,d)
    print('i:{}'.format(i))
print(d)

###
#>>>