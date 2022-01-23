# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 22:37
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 创建链表.py
# @Software: PyCharm
class Node:
    def __init__(self,item):
        self.item = item
        self.next = None

    def creat_linklist(self,li):
        self.head = Node(li[0])
        for element in li[1:]:
            node = Node(element)
            node.next = head  #原来的头交给next
            head = node        #新的head为node本身
        return head
lis = [1,2,3]
n = Node()
n.creat_linklist(lis)
print()