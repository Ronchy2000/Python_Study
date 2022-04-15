# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 第三大的数.py
# @Time      : 2022/2/8 下午9:59
# @Author    : Ronchy

'''
给你一个非空数组，返回此数组中 第三大的数 。如果不存在，则返回数组中最大的数。

思路:
1.数组里有可能有重复的，要第三大，注意重复
2.要第三大
'''
from sortedcontainers import SortedList
from typing import List  #没有这个，指定参数List类型会报错

class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        s = SortedList()
        for num in nums:
            #s是个有序集合
            if num not in s:
                s.add(num)
                if len(s) > 3:
                    s.pop(0)
        return s[0] if len(s) == 3 else s[-1]

sol = Solution()
print(sol.thirdMax([3,2,2,0]))