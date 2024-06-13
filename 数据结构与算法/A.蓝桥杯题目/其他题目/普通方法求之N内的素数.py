# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 普通方法求之N内的素数.py
# @Time      : 2022/2/3 下午3:37
# @Author    : Ronchy
'''
用筛法求之N内的素数。
输入
N
输出
0～N的素数
样例输入
100
'''
#素数： N%i == 0
sum = 0

import math
#判断素数
def judge_prime_number(num):

    if num%2 == 0 and num!= 2: #只判断奇数
        return False
    for i in range(2,int(math.sqrt(num)+1)): # sqrt(9) 3,range 前包后不包，所以要加1
        if num % i == 0:
            return False
    return True

def find(n):
    global sum
    prime =[]
    for i in range(2,n+1): #从2开始，2是最小的素数
        if judge_prime_number(i)==True:
            prime.append(i)
    for j in prime:
        print(j)
        sum += j


if __name__ == '__main__':
    n = int(input("请输入数字："))
    print('n=',n)
    #print( judge_prime_number(n) )
    find(n)
    print('sum',sum)

