# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 水仙花数判断.py
# @Time      : 2022/2/4 下午10:55
# @Author    : Ronchy

'''
判断  给定的三位数是  水仙花数：
其值等于本身每位数字立方和的数

153为水仙花数，
153=1+125+27

本题核心：取出一个数中的每一位
'''
class Solution():
    def shuixianhua(self,num:int)->str:
        if num == (num)%10**3  + (num/10)%10**3 + (num/100)%10**3:
            return 'YES'
        else:
            return 'NO'

sol = Solution()
num = int(input())
print(sol.shuixianhua(num))