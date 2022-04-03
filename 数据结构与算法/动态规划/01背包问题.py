# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 18:22
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 01背包问题.py
# @Software: PyCharm
# 置零，表示初始状态
def bag(n,c,w,v):
    value = [[0 for j in range(c + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, c + 1):
            value[i][j] = value[i - 1][j]
            # 背包总容量够放当前物体，遍历前一个状态考虑是否置换
            if j >= w[i - 1] and value[i][j] < value[i - 1][j - w[i - 1]] + v[i - 1]:
                value[i][j] = value[i - 1][j - w[i - 1]] + v[i - 1]
    # for x in value:
    #     print(x)
    return value

answer_list = bag(6,10,[2,2,3,1,5,2],[2,3,1,5,4,3])
print('----------------------------------')
for i in answer_list:
    print(i,end='\n')
