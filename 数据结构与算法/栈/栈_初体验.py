# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 13:33
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 栈_初体验.py
# @Software: PyCharm
from collections import deque
# stack = deque([])
stack = []
def Enstack(name):
    stack.append(name)

def Destack(name):
    global stack
    if len(stack)==0:
        print("Empty")
        return None
    else:
        for i in range((len(stack)-1) , -1,-1):
            print(stack[i])
            if stack[i] == name:
                #注意：如果使用deque进行切片的话会抛出异常
                stack = stack[0:i]
Enstack('ll')
Enstack('ll')
Enstack('ll')
Enstack('lll')
Enstack('ll')
Enstack('ll')
Destack('lll')
print(stack)




if __name__ == "__min__":
    M = int(input())
    while M:
        M -=1
        operation = input().split()
        if operation[0] == 'in':
            pass
        elif operation[0] == 'out':
            pass
        else:
            print("False!")
            continue

