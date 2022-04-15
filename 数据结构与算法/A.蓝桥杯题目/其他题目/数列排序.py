# -*- coding: utf-8 -*-
# @Time    : 2022/2/24 16:29
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 数列排序.py
# @Software: PyCharm
# from typing import List
#
# class Solution():
#     def bubble_sort(self,len:int,li:List[int]):
#         for i in range(len-1):
#             for j in range(0,len-i-1):
#                 if li[j] < li[j+1]:
#                     li[j],li[j+1] = li[j+1],li[j]
#                     print(i,j,li)
#         return li
# sol = Solution()
# print(sol.bubble_sort(5,[1,2,3,4,5]))

len = int(input())
li = input().split() #str
#str->int 很重要
li = list(map(int,li)) #函数映射
print(li)
for i in range(len-1):
	for j in range(0,len-i-1):
		if li[j] > li[j+1]:
		    li[j],li[j+1] = li[j+1],li[j]

print(li)