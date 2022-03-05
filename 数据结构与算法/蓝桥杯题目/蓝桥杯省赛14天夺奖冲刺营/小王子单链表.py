# -*- coding: utf-8 -*-
# @Time    : 2022/3/5 12:59
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 小王子单链表.py
# @Software: PyCharm
'''
小王子有一天迷上了排队的游戏，桌子上有标号为 1-101−10 的 1010 个玩具，现在小王子将他们排成一列，可小王子还是太小了，
他不确定他到底想把那个玩具摆在哪里，直到最后才能排成一条直线，求玩具的编号。已知他排了 MM 次，每次都是选取标号为 XX 个放到最前面，
求每次排完后玩具的编号序列。

要求一：采用单链表解决

##############
输入描述
第一行是一个整数 MM，表示小王子排玩具的次数。

随后 MM 行每行包含一个整数 XX，表示小王子要把编号为 XX 的玩具放在最前面。

输出描述
共 MM 行，第 ii 行输出小王子第 ii 次排完序后玩具的编号序列。

输入输出样例:
输入    输出
5
3       3 1 2 4 5 6 7 8 9 10
2       2 3 1 4 5 6 7 8 9 10
3       3 2 1 4 5 6 7 8 9 10
4       4 3 2 1 5 6 7 8 9 10
2       2 4 3 1 5 6 7 8 9 10
'''
#利用单链表
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
def print_linklist(ll):
    while ll:
        print(ll.value,end='\n')
        ll = ll.next

head = creat_linklist()
print('head:',head)
print_linklist(head)
