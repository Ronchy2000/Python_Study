# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 14:26
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 迷宫问题优化.py
# @Software: PyCharm
#建图
mp = [[ 0  for i in range(10)] for j in range(10)]
#记录访问点，以及回溯
v = [[ 0  for i in range(10)] for j in range(10)]
# m ,n  = int(input().split())
#迷宫大小 m x n
m ,n= 5,4
#输入地图
for i in range(1,m+1):
    # for j in range(1,n+1):
    mp[i][1:n+1] = list(map(int,input().split()))   #切片赋值，地图大小不变
    ####mp[i][1:n + 1] = [map(int, input().split())] #强制类型转化必须用list函数 ，用该语句出错！！！
for ii in mp:
    for jj in ii:
        print(jj,end=',')
    print()
#起点1,1
startx, starty = 1,1
#终点
p,q = 4,3
#结果:最小路径
min_path = 99999999
'''
优化部分！
方向数组
'''
#tuple 元组，不可被修改
dx = [0,1,0,-1]
dy = [1,0,-1,0]
def dfs(x,y,step):
    global min_path,mp,v,dx,dy
    if x == p and y == q:#到达终点
        if step<min_path:
            min_path = step
        return 0
    for i in range(0,4):#顺时针，四个方向->试探
        tx,ty = x+dx[i],y+dy[i]
        if(mp[tx][ty] == 1 and v[tx][ty] == 0):
            v[tx][ty] = 1 #已经访问
            dfs(tx,ty,step+1)  #递归
            v[tx][ty] = 0 #回溯
    return 0

dfs(startx,starty,0)
print('min_path:',min_path)