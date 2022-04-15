# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 19:17
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 排序.py
# @Software: PyCharm
'''
题目描述
给定一个长度为 N 的数组 A，请你先从小到大输出它的每个元素，再从大到小输出它的每个元素。
https://www.lanqiao.cn/courses/3993/learning/?id=250183
输入描述
第一行包含一个整数 N。
第二行包含 N 个整数 a1,...,an
​表示数组 A 的元素。
'''
N = int(input())
a = list(map(int,input().split()))
if len(a) > N:
    a = a[0:N]

a.sort() #单独执行，返回None
#print(a)
#print(a[-1::-1])
for i in a:
  print(i,end=' ')
b = a[-1: :-1]
print()
for ii in b:
  print(ii,end=" ")

