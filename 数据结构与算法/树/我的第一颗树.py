# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 8:48
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 我的第一颗树.py
# @Software: PyCharm
class Node(object):
    def __init__(self,item):
        self.item = item
        self.left = None
        self.right = None
    def __str__(self):
        return str(self.item)  #print 一个 Node 类时会打印 __str__ 的返回值

class Tree(object):
    def __init__(self):
        self.root = Node('root')
    def add(self,item):
        node = Node(item)
        if self.root is None:
            self.root = node
        else:
            q = [self.root]
            print('q:',q)
            while 1:
                #找根结点
                pop_node = q.pop(0)
                print('pop_node:',pop_node)
                if pop_node.left ==None:
                    pop_node.left = node
                    return
                elif pop_node.right == None:
                    pop_node.right = node
                    return
                else:
                    q.append(pop_node.left)
                    q.append(pop_node.right)

t = Tree()
t.add('child1')
