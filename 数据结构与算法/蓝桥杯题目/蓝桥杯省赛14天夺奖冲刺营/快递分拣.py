# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 15:31
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 快递分拣.py
# @Software: PyCharm

#本题难点，巧用下标。
    #city 和 delivery对应 (下标)

city = []
#使用生成器生成列表
#二维列表，用来存放一个城市的 多个快递
delivery = [[] for i in range(1000)]

def find(s):
    for i in range(0,len(city)):
        if city[i] == s:
            return i
    return -1

if __name__ =="__main__":
    n = int(input())
    for i in range(0,n):
        t = input().split()
        if find(t[1]) == -1: #查找城市是否出现
            city.append(t[1])
            delivery[len(city)-1].append(t[0])
        else:
            delivery[find(t[1])].append(t[0])
    for i in range(0,len(city)):
        print(city[i],len(delivery[i]))
        for j in  delivery[i]:
            print(j)
