# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 15:31
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 试题D：数位排序.py
# @Software: PyCharm
n = int(input())
m = int(input())
# n,m = 50,2

#获取数位和
def get_sum(num):
    sum = 0
    while num:
        sum += num%10
        # 递归出错！一定要更新递归值，易错！
        # get_sum(num//10)
        num = num//10 #注意
        get_sum(num)
    return sum
# print(get_sum(5))
#
ls = [i for i in range(1,n+1)]
##冒泡排序
for i in range(0,len(ls)-1):
    for j in range(i+1,len(ls)):
        # if ls[i] > ls[j]:
        #     ls[i], ls[j] = ls[j],ls[i]
        #冒泡排序变种
        if get_sum(ls[i]) > get_sum(ls[j]):
            ls[i], ls[j] = ls[j], ls[i]
        elif get_sum(ls[i]) == get_sum(ls[j]):
            if ls[i] > ls[j]:
                ls[i], ls[j] = ls[j],ls[i]

# print(ls)
print(ls[m-1])
