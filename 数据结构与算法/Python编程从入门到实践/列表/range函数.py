e# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 10:59
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : range函数.py
# @Software: PyCharm
'''
r = range(1,10,2)
print(r)
print(list(r))  #查看range对象中的整数序列 -->list是列表
'''

'''
#计算1-100之间的偶数和
a = 1
sum1 = 0
sum2 = 0
while a<=100:
    if a%2 == 0:  #if a%2: 奇数和
        sum1 += a
    if not a%2 == 0:
        sum2 +=a
    a+=1
print('偶数和：',sum1)
print('偶数和：',sum2)
'''
'''
r = range(5)
print(list(r)) #注意结果：[0,1,3,4]  ,没有5
'''
