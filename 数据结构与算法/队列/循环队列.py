# -*- coding: utf-8 -*-
# @Time    : 2022/3/13 10:24
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 循环队列.py
# @Software: PyCharm

#python 不需要循环队列，因为python本身就是动态的list，很方便操作

#使用deque


# head == tail 的时候，循环队列为空
# 循环队列满的情况 ；tail > head   (tail+1)%QeueueSize ==head
from collections import deque
Vqueue = deque([])
Nqueue = deque([])
def Enqueue(name,type):
    global Vqueue,Nqueue
    if type == 'V':
        Vqueue.append(name)
    elif type == 'N':
        Nqueue.append(name)

def Dequeue(type):
    global Vqueue,Nqueue
    if type == 'V':
        if len(Vqueue) == 0:
            return None
        # s = Vqueue[0] #这是用的list
        # Vqueue.remove(Vqueue[0])
        # return s
        else:
            Vqueue.popleft() #这是用的deque
    if type == "N":
        if len(Nqueue) == 0:
            return None
        else:
            Nqueue.popleft()

#
# Enqueue('lu','V')
# Enqueue('ronchy','V')
# Dequeue("V")
# print('len:',len(Vqueue))

if __name__ == "__main__":
    M = int(input())
    while M:
        M -= 1
        operation = input('input:').split() #输入存储为list
        # print(operation)
        if operation[0] == "IN":
            Enqueue(operation[1],operation[-1])
        elif operation[0] == "OUT":
            Dequeue(operation[-1])
    # print(Vqueue)
    # print(Nqueue)
    for i in Vqueue:
        print(i)
    for ii in Nqueue:
        print(ii)

