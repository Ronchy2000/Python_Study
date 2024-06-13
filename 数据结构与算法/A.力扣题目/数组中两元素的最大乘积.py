# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 数组中两元素的最大乘积.py
# @Time      : 2022/2/8 下午9:09
# @Author    : Ronchy

'''给你一个整数数组 nums，请你选择数组的两个不同下标 i 和 j，使 (nums[i]-1)*(nums[j]-1) 取得最大值。

请你计算并返回该式的最大值。

来源：A.力扣题目（LeetCode）
链接：https://leetcode-cn.com/problems/maximum-product-of-two-elements-in-an-array
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
'''


from typing import List
class Solution():
    def maxProduct(self,nums:List[int])->int:

        max_a, a = 0,0
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j:
                    a = (nums[i] - 1) * (nums[j] - 1)
                    if max_a < a :
                        max_a = a
                else:
                    pass
        return max_a,i,j

# sol = Solution()
# print(sol.maxProduct([1,2,3,4,5]))


'''解法二'''
'''
找到最大的两个数
'''
class Solution2():
    def maxProduct(self,nums:List[int] )->int:
        a ,b = 0,0
        a = max(nums)
        nums.remove(a)
        b = max(nums)
        return a,b
sol2 = Solution2()
print(sol2.maxProduct([1,2,3,4,5,6,7,8,9]))