# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 求圆面积.py
# @Time      : 2022/2/4 下午11:18
# @Author    : Ronchy
'''
输入半径，输出面积。
输入包含一个整数r，表示圆的半径。
输出格式：
输出一行，包含一个实数，四舍五入保留小数点后7位，表示园的面积。
说明：在本题中，输入是一个整数，但是输出是一个实数。
对于实数输出的问题，请一定看清楚实数输出的要求，
比如本题中要求保留小数点后7位，则你的程序必须严格的输出7位小数，输出过多或者过少的小数位数都是不行的
，都会被认为错误。
'''
import math
PI = math.pi
class Solution():
    def cir_area(self,r:int)->float:
        return (PI*r*r)

sol = Solution()
r = int(input())
#回顾 python字符串格式化
print('%.7f' %sol.cir_area(r))

