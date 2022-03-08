# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 10:26
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 封装_链表.py
# @Software: PyCharm
#万门大学教育，python 算法课
class Node:
    #value
    #next
    def __init__(self,val = None,next = None):
        self.value = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = Node() #dummy node
        self.size = 0

    def add_first(self,value):#插入   dummy-> 插 -> 口 -> 口 -> 口
        node = Node(value)
        node.next = self.head.next
        self.head.next = node
        self.size += 1

    def add_last(self,val):
        node = Node(val)
        r = self.head
        while r.next != None:
            r = r.next #dummy node ，所以 先跳  后再打印
        r.next = node
        self.size += 1

    def remove(self):
        #if remove_first:

        #else:
        pass

    def peek(self,val):#查找
        pass

    def print_linklist(self):
        r = self.head
        while r.next != None:
            r = r.next
            print(r.value,end=' ')

    def len(self):
        return self.size

ll = LinkedList()
ll.add_first(5)
ll.add_first(8)
ll.add_first(11)
print(ll.len())
ll.add_last(5)
ll.print_linklist()
