class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # mp = dict() # 创建空字典
        mp = collections.defaultdict(list)  # 利用工厂函数创建字典，默认值为list
        # print(type(mp))
        for s in strs:
             # 注意排序的方法： sort 是对list的操作；而sorted是对所有可迭代的对象的操作
            key = str(sorted(s))
            # key = "".join(sorted(s))
            mp[key].append(s) # 把相同的key的value放在一起
        return list(mp.values()) # 把所有的value取出来，组成一个list返回