# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 10:03
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 迷宫问题.py
# @Software: PyCharm
import copy
#定义地图
#列表生成器得到二维列表
mp = [[0 for i in range(4)] for i in range(5)] #1表示空地，2表示障碍物

v = copy.deepcopy(mp) #访问数组（别名   ->0表示未访问，1表示访问
#终点坐标 p ,q
p,q = 3,2
# p,q = int(input().split())
min_path = 99999999

#x , y 为当前坐标
def dfs(x:int,y:int,step:int):
    global min_path
    #如果到达终点
    if x == p and y == q:
        if step < min_path:
            min_path = step
            return
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
    return

#m x n 大小的地图
# m,n = int(input().split())
m = 5
n = 4
startx ,starty = 0,0
for i in range(0,m):
    mp[i] = input().split() #此时是str类型
    mp[i] = list(map(int , mp[i])) #list中的类型变为int类型
for i in range(0,m):
    for j in range(0,n):
        print(mp[i][j],end=',')
    print()
#起点已访问
v[startx][starty]  = 1
dfs(startx,starty,0)
print('min_path:',min_path)





