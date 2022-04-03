# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 15:22
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 快速指数.py
# @Software: PyCharm
cnt = 0
def fast_power(x,n):
    global cnt
    cnt +=1
    print('step',cnt,sep=':')
    if n == 0:
        return 1.0
    elif n<0:
        return 1/fast_power(x,-n)
    elif n>0 and n%2 == 1:
        return fast_power(x*x,n//2)*x
    elif n>0 and n%2 == 0:
        return fast_power(x*x,n//2)
    else:
        print('error!')
        return None

if __name__ =='__main__':
    #非打印科学计数
    print('%f'%fast_power(2,100))