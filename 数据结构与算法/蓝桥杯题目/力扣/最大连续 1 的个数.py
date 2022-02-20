# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 最大连续 1 的个数.py
# @Time      : 2022/2/8 下午9:18
# @Author    : Ronchy

'''
https://leetcode-cn.com/problems/max-consecutive-ones/solution/zui-da-lian-xu-1de-ge-shu-by-leetcode-so-252a/
给定一个二进制数组 nums ， 计算其中最大连续 1 的个数。

一次遍历
'''
from typing import List
class Solution:
    def findMaxConsecutiveOnes(self,nums:List[int])->int:
        max_a ,a = 0,0
        for num in nums:
            if num ==1:
                a+=1
            else:
                a = 0
            if a>max_a :
                max_a = a
        return  max_a

sol = Solution()
print(sol.findMaxConsecutiveOnes([1,1,1,0,0,1,1,0,0,0,0]))

