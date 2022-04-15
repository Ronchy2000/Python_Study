# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 13:01
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 模拟方法.py
# @Software: PyCharm
'''
xxx  +  xxx = xxx
x代表1-9的不同的数字，这个算式有多少种正确的填写方法

111 + 222 = 333  错误写法
满足加法交换的式子算两种不同的答案，答案是偶数

'''
#此函数判断：是否有重复的数字
def check(a,b,c):
    f = []
    f.append(0)
    while a!=0 :
        if a%10 in f:
            return False
        else:
            f.append(a%10)
        if b % 10 in f:
            return False
        else:
            f.append(b % 10)
        if c%10 in f:
            return False
        else:
            f.append(c%10)
        a = int(a/10)
        b = int(b/10)
        c = int(c/10)
    return True

if __name__ == '__main__':
    cnt = 0
    for  a in range(123,987): #不重复数字的最大范围;123 987
        for b in range(123,987-a):
            c = a+b
            if(check(a,b,c)):
                cnt += 1
                print(a,'+',b,'=',c)
    print(cnt)

#python 提交的正式代码
#print(336)