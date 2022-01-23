# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 10:57
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 链表.py
# @Software: PyCharm


#创建链表
#创建Node对象并创建另一个类来使用这个节点对象
class Node:
    def __init__(self,dataval=None):
        self.dataval = dataval
        self.nextval = None

class SLinkedList:
    def __init__(self):
        self.headval = None

    #链表只能向前遍历
    def listprint(self):

        while self.headval is not None:
            print(self.headval.dataval)
            self.headval = self.headval.nextval
# 链表列表开头插入
    def AtBegining(self,newdata):
        NewNode = Node(newdata)

        NewNode.nextval = self.headval
        self.headval = NewNode
#链表末尾插入
    def AtEnd(self,newdata):
        NewNode = Node(newdata)
        if self.headval is None:
            self.headval = NewNode
            return
        laste = self.headval
        while(laste.nextval):
            laste = laste.nextval
        laste.nextval = NewNode

list1 =  SLinkedList()
list1.headval = Node("Mon")
e2 = Node("Tue")
e3 = Node("Wed")

list1.headval.nextval = e2

e2.nextval = e3

list1.AtBegining("Sun")
list1.AtEnd("LAST")
list1.listprint()