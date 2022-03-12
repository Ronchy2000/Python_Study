# -*- coding: utf-8 -*-
# @Time    : 2022/3/12 15:04
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 银行普通队列.py
# @Software: PyCharm
import os
import sys

Vqueue, Nqueue = [], []
Vhead, Nhead = 0, 0
Vtail, Ntail = 0, 0


def Enqueue(name, type):
    global Vqueue, Nqueue,Vhead, Nhead,Vtail, Ntail
    if type =="V":
        Vqueue.append(name)
        Vtail += 1
    elif type == "N":
        Nqueue.append(name)
        Ntail += 1

def Dequeue(type):
    global Vqueue, Nqueue, Vhead, Nhead, Vtail, Ntail
    if type == "V":
        Vhead += 1
    elif type == "N":
        Nhead += 1

def gethead(type):
    if type == "V":
        return Vqueue[Vhead]
    elif type == "N":
        return Nqueue[Nhead]
if __name__ == "__main__":
    M = int(input())
    while M:
        M = M - 1

        operation = input().split()
        print(operation)
        if operation[0] == "IN":
            Enqueue(operation[1],operation[-1])
        elif operation[0] == "OUT":
            Dequeue(operation[-1])

    #print("Vqueue:",Vqueue[Vhead:])
    #print("Nqueue:",Nqueue[Nhead:])
    for i in  Vqueue[Vhead:]:
        print(i)
    for ii in Nqueue[Nhead:]:
        print(ii)




