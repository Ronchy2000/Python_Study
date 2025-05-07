"""
https://leetcode.cn/problems/longest-consecutive-sequence/?envType=study-plan-v2&envId=top-100-liked

"""

# class Solution:
#     def longestConsecutive(self, nums: List[int]) -> int:
#         """
#         This converts the input array nums into a set, which provides O(1) average time complexity for lookups. This is crucial for the algorithm's efficiency because it allows the following operations to be performed in constant time:
#         """
#         nums_set = set(nums) # 去掉重复,查找到时间复杂度为O(1)
#         max_length = 0
#         for n in nums_set:
#             if (n-1) not in nums_set: # 找到：头 （开始的地方）
#                 current_num = n
#                 current_length = 1
#             # 往后查找
#             while current_num + 1 in nums_set:
#                 current_length += 1
#                 current_num += 1
#                 # if current_length > max_length:
#                 #     max_length = current_length
#                 max_length = max(max_length, current_length)
#         return max_length


'''
排序法做
'''
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums_set = set(nums)
        nums_list = list(nums_set)
        nums_list.sort()
        max_length = 0

        if len(nums_list) == 0:
            return 0
        for i in range(len(nums_list)):
            current_num = nums_list[i]
            current_length = 1
            while current_num + 1 in nums_list:
                current_length += 1
                current_num += 1
            max_length = max(max_length, current_length)
        return max_length