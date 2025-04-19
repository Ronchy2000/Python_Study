'''
题目描述：
在一个大型软件维护系统中，小慕负责选择最适合升级的补丁版本。系统中的补丁版本遵循严格的迭代顺序：
每一个新版本只能基于最多一个已有版本进行更新，且不存在相互依赖的情况。每次升级时，小慕总是希望找到
迭代层数最深的补丁版本，也就是从最初的版本一路演进到当前版本，经过的更新次数最多。

现在，小慕收集到了若干条补丁版本之间的依赖关系。请你帮助他找出当前可升级的补丁版本中，哪些是迭代层数最多的。
如果存在多个符合条件的版本，请按照字典序进行排序输出。

输入格式：
第一行输入一个整数 N，表示系统中记录的版本迭代关系数量。
接下来的N行中，每行输入两个字符串，代表一个版本及其前序版本，用空格隔开。
如果该版本没有前序版本，则其前序版本用"NA"表示。
1≤N≤100000
字符串长度为1到100，由大写字母和数字组成

输出格式：
输出所有拥有最多迭代层数的补丁版本号，多个版本按字典序升序排列，用空格分隔。

=================
输入样例：
6
CN0010 BF0001
BF0001 AZ0001
AZ0001 NA
BF0010 AZ0001
AW0001 NA
BF0011 AZ0001

输出样例：
CN0010

在这组版本依赖中，AZ0001与AW0001没有前序版本，因此它们处于起始状态（迭代次数为0）；
基于AZ0001构建的 BF0001、BF0010、BF0011 各迭代了1次；
而CN0010是在BF0001 的基础上迭代出来的，路径为AZ0001 BF0001CN0010，共计迭代了2次。
因此，CN0010 是迭代次数最多的补丁版本。
=================
'''

'''
分析：
这道题本质上是一个有向无环图(DAG)中寻找最长路径的问题。
数据结构选择：
- 字典：使用字典存储图结构，键为版本号，值为其前序版本  例如： graph = {"CN0010": "BF0001", "BF0001": "AZ0001", ...}
- 集合 ：用于存储所有版本号
- 记忆化存储 ：使用字典存储每个版本的最大迭代层数，避免重复计算

解题思路：
- 首先构建一个图，记录每个版本的前序版本
- 找出所有起始版本（没有前序版本的版本）
- 使用深度优先搜索（DFS）计算每个版本的最大迭代层数
- 找出所有终点版本（没有后继版本的版本）
- 在终点版本中找出迭代层数最大的版本
- 按字典序排序并输出结果
'''


def find_deepest_patches(n):
     # 构建图：记录每个版本的前序版本
    graph = {}  # 键为版本号，值为其前序版本
    all_versions = set()  # 记录所有版本
    has_pre_version = set()  # 记录有前序版本的版本（非起始版本）
    
    # 读取版本依赖关系
    for _ in range(n):
        version, pre_version = list(map(str,input().strip().split(' ')))
        all_versions.add(version)

        if pre_version != 'NA':
            all_versions.add(pre_version) # 添加前序版本到所有版本集合
            has_pre_version.add(version ) # 标记当前版本有前序版本
            graph[version] = pre_version # 记录依赖关系

    # 找出所有起始版本（没有前序版本的版本） 集合运算
    start_versions = all_versions - has_pre_version

    # 计算每个版本的最大迭代层数（使用记忆化搜索）
    max_depths = {}  # 记录每个版本的最大迭代层数


    def dfs(version):
            # 如果已经计算过，直接返回结果
            if version in max_depths:
                return max_depths[version]
            
            # 如果是起始版本或没有前序版本信息，深度为1
            if version not in graph or graph[version] == "NA":
                max_depths[version] = 1
                return 1
            
            # 递归计算前序版本的深度，并加1
            max_depths[version] = dfs(graph[version]) + 1
            return max_depths[version]
            
    # 计算所有版本的最大迭代层数
    for version in all_versions:
        dfs(version)
    
    # 找出所有终点版本（没有后继版本的版本）
    end_versions = set()
    for version in all_versions:
        is_end = True
        for v in graph:
            if graph[v] == version:  # 如果有版本以当前版本为前序，则当前版本不是终点
                is_end = False
                break
        if is_end:
            end_versions.add(version)

    # 找出终点版本中迭代层数最大的版本
    max_depth = 0
    deepest_versions = []
    
    for version in end_versions:
        depth = max_depths[version]
        if depth > max_depth:
            max_depth = depth
            deepest_versions = [version]
        elif depth == max_depth:
            deepest_versions.append(version)
    
    # 按字典序排序
    deepest_versions.sort()
    
    return ' '.join(deepest_versions)


if __name__ == "__main__":
    # 读取版本迭代关系数量
    n = int(input().strip())
    result = find_deepest_patches(n)
    print(result)