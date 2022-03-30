# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 15:57
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 杨辉三角.py
# @Software: PyCharm

n = int(input())
#定义一个二维矩阵，抽象思考
'''
存放数据
口口口口口口口口...
口口口口口口口口...
口口口口口口口口...
.
.
.

'''
a = [[0]*100] *100

#print(a)
#输入杨辉三角
for i in range(0,n):
    a[i] = input().split()
    a[i] = list(map(int,a[i]))

# for i in range(1,n+1):
#     print(a[i])

#反向递推
for i in range(n -2,-1,-1):
    for j in range(0,i+1):
        if a[i+1][j] >= a[i+1][j+1]:
            a[i][j] += a[i+1][j]
        else:
            a[i][j] += a[i+1][j+1]

print('max:',a[0][0])

# for i in range(0,n):
#     for j in range(0,i+1):
#         print(a[i][j],sep=',',end=',')
#     print()

