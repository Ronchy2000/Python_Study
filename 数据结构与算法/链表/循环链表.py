# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 16:02
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 循环链表.py
# @Software: PyCharm
'''
设有 n 个人围坐在圆桌周围，现从某个位置 k 上的人开始报数，报数到 m 的人就站出来。
下一个人，即原来的第 m+1 个位置上的人，又从 1 开始报数，再报数到 m 的人站出来。
依次重复下去，直到全部的人都站出来为止。试设计一个程序求出这 n 个人的出列顺序。
'''
class Node:
    def __init__(self,val = None,next = None):
        self.value = val
        self.next = next
        #
        #self.prev = prev

def creat_ll(n):
    if n <= 0:
        return False
    elif n == 1:
        return Node(1)
    else:
        head = Node(1)  #没有Dummy node 起始时刻就是链表的初值
        r = head
        for i in range(2,n+1):
            r.next = Node(i)
            r = r.next #迭代
        #循环链表的核心
        r.next = head
        return head
#不能这么打印，这是个循环链表！
# def print_ll(l):
#     while l.next != None:
#         print(l.value,end=' ')
#         l = l.next

if __name__ =='__main__':
    n , m ,k  = map(int,input().split(' '))
    # print(n,m,k,end=' ')
    head = creat_ll(n)

    r = head
    #print_ll(head)
    #放在k上,从0开始：
    for i in range(0,k-1):
        r = r.next
    #不是只剩一个元素，开始剔除
    while r.next != r:
        #开始报数
        for i in range(1,m-1):
            r = r.next
        print(r.next.value,' ')
        #删除被选择的元素  r.next
        r.next = r.next.next
        # r = r.next
    print(r.value)

