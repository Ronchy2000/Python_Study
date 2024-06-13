# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 10:03
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 迷宫问题.py
# @Software: PyCharm
import copy
#定义地图
#列表生成器得到二维列表
mp = [[0 for i in range(100)] for i in range(100)] #1表示空地，2表示障碍物

v =  [[0 for i in range(100)] for i in range(100)] #访问数组（别名   ->0表示未访问，1表示访问
#终点坐标 p ,q
p,q = 4,3
# p,q = int(input().split())
min_path = 99999999

#x , y 为当前坐标
def dfs(x,y,step):
    global min_path
    #如果到达终点
    if x == p and y == q:
        if step < min_path:
            min_path = step
            return 0
    #clockwise detect
    #右
    if mp[x][y+1] == 1 and v[x][y+1] == 0:
        v[x][y+1] =1 #标记为已访问
        dfs(x,y+1,step+1)
        v[x][y+1] = 0 #标记为未访问，回溯
    #下
    if mp[x+1][y] == 1 and v[x+1][y] == 0:
        v[x+1][y] = 1
        dfs(x+1,y,step+1)
        v[x + 1][y] = 0
    #左
    if mp[x][y-1] == 1 and v[x][y-1] == 0:
        v[x][y-1] = 1
        dfs(x,y-1,step+1)
        v[x][y-1] = 0
    #上
    if mp[x-1][y] == 1 and v[x-1][y] == 0:
        v[x-1][y] = 1
        dfs(x-1,y,step+1)
        v[x - 1][y] = 0
    return 0

#m x n 大小的地图
# m,n = int(input().split())
m = 5
n = 4
startx ,starty = 1,1
for i in range(1,m+1):
    mp[i][1:n+1] =list( map( int,input().split()) )

#查看输入迷宫
# for i in range(1,m+1):
#     for j in range(1,n+1):
#         print(mp[i][j],end=',')
#     print()

#起点已访问
v[startx][starty] = 1
dfs(startx,starty,0)
print('min_path:',min_path)





