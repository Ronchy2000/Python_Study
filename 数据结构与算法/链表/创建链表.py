# -*- coding: utf-8 -*-
# @Time    : 2022/2/27 15:34
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 创建链表.py
# @Software: PyCharm
class Node():
    def __init__(self,val = None,next = None):
        self.val = val
        self.next = next

class SingleLinkedList():
    def __init__(self):
        self.head = Node()
        self.size = 0
#接着插入
    def add_first(self,val):
        node = Node(val,None)
        node.next = self.head.next
        self.head.next = node #存储
        self.size += 1
    def len(self):
        return self.size

#实例对象
ll = SingleLinkedList()
#实例对象访问类方法
ll.add_first(5)
ll.add_first(8)
print(ll.head.next)#存储地址