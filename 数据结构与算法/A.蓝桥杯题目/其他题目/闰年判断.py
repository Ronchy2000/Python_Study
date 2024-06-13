# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 闰年判断.py
# @Time      : 2022/2/4 下午10:47
# @Author    : Ronchy
'''
给定一个年份，判断这一年是不是闰年
是：yes
否：no
'''

class Solution():
    def judge_leap_year(self,year:int)->str:
        if year % 4 ==0 and year %100 != 0:
            return 'yes'
        elif year %400 ==0:
            return 'yes'
        else:
            return 'no'

sol = Solution()
year = int(input())
print(sol.judge_leap_year(year))
