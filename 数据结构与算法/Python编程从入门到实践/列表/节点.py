# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 10:45
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 节点.py
# @Software: PyCharm

'''在有些情况下，存储数据的内存分配不能位于连续的内存块中。
所以接受指针的帮助，其中数据和数据元素的下一个位置的地址也被存储。
所以从当前数据元素的值中知道下一个数据元素的地址。通常这样的结构被称为指针。
但在Python中，将它们称为节点'''

class daynames:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None

e1 = daynames('Mon')
e2 = daynames('Tue')
e3 = daynames('Wed')
e4 = daynames("Thu")

e1.nextval = e2
e2.nextval = e3
e3.nextval = e4

thisvalue = e1

while thisvalue:
    print(thisvalue.dataval)
    thisvalue = thisvalue.nextval


