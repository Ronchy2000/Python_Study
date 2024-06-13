# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 23:49
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 约数问题.py
# @Software: PyCharm

num1,num2,num3 = input().split()

def yueshu(number):
    yueshu_cnt = 0
    yueshu  = []
    for i in range(1,number+1):
        if number%i == 0:
            yueshu_cnt += 1
            yueshu.append(i)
    return yueshu_cnt,yueshu

print(num1,num2,num3)

a1,b1 = yueshu(int(num1))
a2,b2 = yueshu(int(num2))
a3,b3 = yueshu(int(num3))

#利用集合查重！
set1 = set(b1)&set(b2)
set2=  set(b1)&set(b3)
set3=  set(b2)&set(b3)
# print(set1,set2,set3)
# print(set((set1|set2|set3)))
print(len(set((set1|set2|set3))))
