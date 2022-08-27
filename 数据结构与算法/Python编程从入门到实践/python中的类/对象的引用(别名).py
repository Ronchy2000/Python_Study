# -*- coding: utf-8 -*-
# @Time    : 2022/3/5 14:22
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 对象的引用(别名).py
# @Software: PyCharm

class Node():
    def __init__(self,value=None,next=None):
        self.value = value
        self.next = next


n = Node(66)
print(n.value)
nn = n #对象的引用（其实就是对象的别名）
print(nn.value)
nn.value = 666


print('nn.value：',nn.value)
nn.value = 666
print('>>>>>>>>>修改了nn<<<<<<<')
print('n.value；',n.value)