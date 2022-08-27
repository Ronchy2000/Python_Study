# -*- coding: utf-8 -*-
# @Time    : 2021/4/11 19:56
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : ATM等待时间.py
# @Software: PyCharm
import random

class ATM:
    def __init__(self,maxtime = 10):
        self.maxtime = maxtime

    #模拟服务时间
    def getServCompleteTime(self,start = 0):
        return start + random.randint(1,self.maxtime)

class Customers:
    def __init__(self,n):
        self.count = n
        self.left = n
#下一个客户到达的时间
    def getNextArrvTime(self,start = 0,arrvtime = 10):
        if self.left != 0:
            self.left -= 1
            return start + random.randint(1, arrvtime) #1-10产生随机数
        else:
            return 0
    def isOver(self):
        return True if self.left == 0 else False

c = Customers(100)
a= ATM()

wait_list = [] #等待时间节点
wait_time = 0  #等待总时间
cur_time = 0
cur_time += c.getNextArrvTime()

wait_list.append(cur_time)

while len(wait_list) != 0 or not c.isOver():
    if wait_list[0] <= cur_time:
        next_time = a.getServCompleteTime(cur_time)
        del wait_list[0]
    else:
        next_time = cur_time + 1

    if not c.isOver() and len(wait_list) == 0:
        next_arrv = c.getNextArrvTime(cur_time)
        wait_list.append(next_arrv)

    if not c.isOver() and wait_list[-1] < next_time:
        next_arrv = c.getNextArrvTime(wait_list[-1])
        wait_list.append(next_arrv)
        while next_arrv < next_time and not c.isOver():
            next_arrv = c.getNextArrvTime(next_arrv)
            wait_list.append(next_arrv)

    for i in wait_list:
        if i <= cur_time:
            wait_time += next_time - cur_time
        elif cur_time < i < next_time:
            wait_time += next_time - i
        else:
            pass

    cur_time = next_time

print(wait_time/c.count)
