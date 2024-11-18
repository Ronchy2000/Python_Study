# -*- coding: utf-8 -*-
#
# @Time : 2024-11-18 21:29:09
# @Author : Ronchy Lu
# @Email : rongqi1949@gmail.com
# @File : 吃鸡蛋问题.py
# @Software: PyCharm
# @Description: None

def count_ways(total_eggs, min_eggs_per_day, max_eggs_per_day):
    # 初始化一个列表来存储每种蛋数量的解决方案数
    dp = [0] * (total_eggs + 1)
    dp[0] = 1  # 吃掉0个鸡蛋的方案数是1（什么也不吃）

    # 遍历所有的蛋数量
    for eggs in range(1, total_eggs + 1):
        for daily_eggs in range(min_eggs_per_day, max_eggs_per_day + 1):
            if eggs >= daily_eggs:
                dp[eggs] += dp[eggs - daily_eggs]

    return dp[total_eggs]

total_eggs = 8
min_eggs_per_day = 1
max_eggs_per_day = 4

print(f"共有 {count_ways(total_eggs, min_eggs_per_day, max_eggs_per_day)} 种方法吃完这8颗鸡蛋。")
