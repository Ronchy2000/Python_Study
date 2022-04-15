# -*- coding: utf-8 -*-
# @Time    : 2022/3/5 14:55
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 我的第一个链表.py
# @Software: PyCharm
class Node():
    def __init__(self,value=None,next=None):
        self.value = value
        self.next = next
def creat_linklist():
    head = Node()
    for i in range(1,11):
        node = Node(i)
        node.next = head.next
        head.next = node
    return head

def add_last(ll,val):
    new_node = Node(val)
    # node = ll
    while ll.next != None:
        ll = ll.next
    ll.next = new_node
    print(ll.value,end=',')#表示查找到了1
    print('ll.next:',ll.next)
    print('add OK!')



def print_linklist(ll):
    while ll:
        print(ll.value,end=';')
        ll = ll.next

head = creat_linklist()
print('head:',head)
print_linklist(head)
#>>>   None,10,9,8,7,6,5,4,3,2,1,
add_last(head,666)
print('\n')
print_linklist(head)
