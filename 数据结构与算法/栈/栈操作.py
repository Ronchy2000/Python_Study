# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 18:15
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 栈操作.py
# @Software: PyCharm
data = []

def push(elem):
    global data
    data.append(elem)

def pop():
    global data
    if data == []:
        raise Warning("此栈为空，错误操作");
    data.pop()

if __name__ == '__main__':
    M = int(input())
    while M:
        M -=1
        operation = input().split()
        if operation[0] == 'in':
            push(operation[-1])
        elif operation[0] == 'out':
            #出栈检索
            while data[-1] != operation[-1]:
                pop()
            #检索到相同的值，根据题意，本元素也不要，故再执行一次pop()
            pop()
        else:
            print('错误操作')
            continue

    if data == []:
        print("Empty")
    else:
        print(data[-1])