# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 13:49
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 如何创建地图2.py
# @Software: PyCharm
mp =[ [i for i in range(10)] for j in range(10)  ]

m ,n = 2,4
print('___________origin____________________-')
for i in range(1,m+1):
    for j in range(1,n+1):
        print(mp[i][j],end=',')
    print()
print('___________changed____________________-')
for i in range(1,m+1):
    #mp[i][1:] = list( map(int,input().split())  )  #被赋值，100个空间变为0 + 4 = 5个空间
    mp[i][1:n+1] = list(map(int, input().split()))  # 被赋值，100个空间还是100个空间，用切片操作，注意是 n 列！
for ii in mp:
    for jj in ii:
        print(jj,end=',')
    print()

print('len(mp[1])',len(mp[1]))
print('len(mp[2])',len(mp[2]))