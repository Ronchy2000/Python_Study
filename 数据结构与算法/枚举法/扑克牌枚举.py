# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 15:50
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 扑克牌枚举.py
# @Software: PyCharm
#创建一定长度的lsit
# a = [[0]*10]
# a = a[0]
a,ans  = [],[]
c = input().split(' ')
print(c)
for i in range(0,6):
    if c[i] == 'A':
        a.append(1)
    elif c[i] == 'J':
        a.append(11)
    elif c[i] == 'Q':
        a.append(12)
    elif c[i] == 'K':
        a.append(13)
#很特殊的10
    elif c[i] == '10':
        a.append(10)
    else:
    #利用ASCII码来 输入 0 - 9，ASCII码里没有10!所以再次加入一个判断
        a.append(ord(c[i]) - ord('0'))

# print(a)

