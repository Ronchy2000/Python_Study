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
            # 从根结点入手
            q = [self.root]

            while 1:
                for i in q:
                    print("i:", i)
                pop_node = q.pop(0)
                print('q:',q)
                print('pop_node:',pop_node)
                if pop_node.left ==None:
                    pop_node.left = node
                    print('left add!')
                    print('----------------------------')
                    return
                elif pop_node.right == None:
                    pop_node.right = node
                    print('right add!')
                    print('----------------------------')
                    return
                else:
                    q.append(pop_node.left)
                    q.append(pop_node.right)
#########################################################################################################3
    def get_parent(self, item):
        if self.root.item == item:
            return None  # 根节点没有父节点
        tmp = [self.root]  # 将tmp列表，添加二叉树的根节点
        while tmp:
            pop_node = tmp.pop(0)
            if pop_node.left and pop_node.left.item == item:  # 某点的左子树为寻找的点
                return pop_node  # 返回某点，即为寻找点的父节点
            if pop_node.right and pop_node.right.item == item:  # 某点的右子树为寻找的点
                return pop_node  # 返回某点，即为寻找点的父节点
            if pop_node.left is not None:  # 添加tmp 元素
                tmp.append(pop_node.left)
            if pop_node.right is not None:
                tmp.append(pop_node.right)
        return None

    def delete(self, item):
        if self.root is None:  # 如果根为空，就什么也不做
            return False

        parent = self.get_parent(item)
        if parent:
            del_node = parent.left if parent.left.item == item else parent.right  # 待删除节点
            if del_node.left is None:
                if parent.left.item == item:
                    parent.left = del_node.right
                else:
                    parent.right = del_node.right
                del del_node
                return True
            elif del_node.right is None:
                if parent.left.item == item:
                    parent.left = del_node.left
                else:
                    parent.right = del_node.left
                del del_node
                return True
            else:  # 左右子树都不为空
                tmp_pre = del_node
                tmp_next = del_node.right
                if tmp_next.left is None:
                    # 替代
                    tmp_pre.right = tmp_next.right
                    tmp_next.left = del_node.left
                    tmp_next.right = del_node.right

                else:
                    while tmp_next.left:  # 让tmp指向右子树的最后一个叶子
                        tmp_pre = tmp_next
                        tmp_next = tmp_next.left
                    # 替代
                    tmp_pre.left = tmp_next.right
                    tmp_next.left = del_node.left
                    tmp_next.right = del_node.right
                if parent.left.item == item:
                    parent.left = tmp_next
                else:
                    parent.right = tmp_next
                del del_node
                return True
        else:
            return False
t = Tree()
t.add('child1')
t.add('child2')
t.add('child11')
