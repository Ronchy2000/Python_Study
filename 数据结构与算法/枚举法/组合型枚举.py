# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 15:05
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 组合型枚举.py
# @Software: PyCharm
'''
组合型枚举
n 个里面挑选 m 个,

'''
n = 0
m = 0
chosen =[]
def calc(x):
    if len(chosen) >m:
        return 0
    if len(chosen) + n -x +1 <m: #剪枝
        return 0
    if x == n+1:
        for i in chosen:
            print(i,end=' ')
        print()
        return 0
    calc(x+1)
    chosen.append(x)

    calc(x+1)
    chosen.pop()

tem = input().split()
n = int(tem[0])
m = int(tem[1])
calc(1)
