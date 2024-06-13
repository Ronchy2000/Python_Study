# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 23:06
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 和尚挑水缸问题.py
# @Software: PyCharm
'''
一个和尚要挑水，每次最多能挑 a 千克，水缸最多能装 t 千克，开始时水缸为空。
　　请问这个和尚最少要挑多少次可以将水缸装满？
输入格式
　　输入一行包含两个整数 a, t，用一个空格分隔。
输出格式
　　输出一行包含一个整数，表示答案。
样例输入
20 2021
样例输出
102
'''
a ,t = map(int,input().split())
answer = t//a
if t%a != 0:
    answer += 1
print(answer)

