# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 22:42
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 螺旋数组.py
# @Software: PyCharm
'''
对于一个 n 行 m 列的表格，我们可以使用螺旋的方式给表格依次填上正整数，我们称填好的表格为一个螺旋矩阵。
　　例如，一个 4 行 5 列的螺旋矩阵如下：
　　1 2 3 4 5
　　14 15 16 17 6
　　13 20 19 18 7
　　12 11 10 9 8
　　请问，一个 30 行 30 列的螺旋矩阵，第 20 行第 20 列的值是多少？


'''


demand = 30
num = 1
arry = [[0 for j in range(demand)] for i in range(demand)]
#print(arry)
for i in range((demand//2) + 1):
    for j in range(i,demand-i):
        arry[i][j] = num
        num += 1
    for j in range(i+1,demand-i):
        arry[j][demand-i-1] = num
        num += 1
    for j in range(demand-i-2,i,-1):
        arry[demand-i-1][j] = num
        num += 1
    for j in range(demand-i-1,i,-1):
        arry[j][i] = num
        num += 1

for i in arry:
    for j in i:
        print(j,end='\t')
    print()
print('20,20',arry[19][0])