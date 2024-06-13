# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 11:50
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 如何创建地图.py
# @Software: PyCharm
'''

本代码 熟练 二维列表的生成
使用列表生成器
注意，使用映射map函数时，数组map重名导致致命错误！！！
'''
#定义地图
#列表生成器得到二维列表
mp = [[0 for i in range(4)] for i in range(5)] #1表示空地，2表示障碍物
#打印地图
for i in mp:
    for j in i:
        print(j,end=',')
    print()


#m x n 大小的地图
# m,n = int(input().split())
m = 1 #行
n = 4 #列
for i in range(0,m):
    mp[i] = input().split()
    # mp[i] = list( int(ele)  for ele in mp[i] )
    print(type(mp[i]))
    #转变为list类型！否则使用mp时会报错。map返回迭代器
    mp[i] = list(map(int,mp[i]))

for i in range(0,m):
    print('mp[i]',mp[i])
    for j in range(0,n):
        print(mp[i][j],end=',')
    print()

