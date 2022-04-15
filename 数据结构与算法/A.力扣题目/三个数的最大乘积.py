# -*- coding: utf-8 -*-
# @Time    : 2022/2/20 19:56
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : 三个数的最大乘积.py
# @Software: PyCharm
'''
给你一个整型数组 nums ，在数组中找出由三个数组成的最大乘积，并输出这个乘积。


'''
#   思路：
#       不用排序，线性查找 最大的三个数
# 三个数的最大乘积，
# 注意：如果是负数，那么负负得正，
# 线性查找时，记录最大的三个数，以及最小的两个数
# 三个数的最大乘积：
#    乘积1 = max1 *  max2  * max 3
#    乘积2 = min1 *  min2 *  max1
#最大的数 max(乘积1,乘积2)
from typing import List
import sys
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        min1 ,min2 = nums[0],nums[0]
        max1 ,max2 ,max3 = nums[0],nums[0],nums[0]
        for num in nums:

            if num <min1:
                min2 = min1 #注意赋值顺序
                min1 = num
            elif num < min2:
                min2 = num


            if num >max1:
                max3 = max2 #注意赋值顺序
                max2 = max1 #注意赋值顺序
                max1 = num
            elif num>max2:
                max3 = max2 #注意赋值顺序
                max2 = num
            elif num > max3:
                max3 = num

        product1 = max1* max2 * max3
        product2 = (min1*min2)*max1
        print('max',max1,max2,max3)
        print('min',min1,min2)
        return max(product1,product2)

sol = Solution()
# print(sol.maximumProduct([-10,-2,3,4,5]))
print(sol.maximumProduct([-1,-2,-3]))

