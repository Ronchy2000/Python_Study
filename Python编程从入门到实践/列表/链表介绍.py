# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 22:28
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 链表介绍.py
# @Software: PyCharm
class Node:
    def __init__(self,item):
        self.item = item
        self.next =None
a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
a.next = b
b.next = c
c.next = d

print(a.next.next.item)