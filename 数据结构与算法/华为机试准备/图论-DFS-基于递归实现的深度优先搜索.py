class Solution:
    def dfs_recursive(self, graph, u, visited):
        print(u)                        # 访问节点
        visited.add(u)                  # 节点 u 标记其已访问

        for v in graph[u]:
            if v not in visited:        # 节点 v 未访问过
                # 深度优先搜索遍历节点
                self.dfs_recursive(graph, v, visited)
        

# 哈希表实现邻接表
"""
在 Python 中，通过哈希表（字典）可以轻松的实现邻接表。
哈希表实现邻接表包含两个哈希表：
第一个哈希表主要用来存放顶点的数据信息，哈希表的键是顶点，值是该点所有邻接边构成的另一个哈希表。
第二个哈希表用来存放顶点相连的边信息，哈希表的键是边的终点，值是边的权重。

无权图的话，直接用哈希表去代替就行。
"""
graph = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B", "D", "E"],
    "D": ["B", "C", "E", "F"],
    "E": ["C", "D"],
    "F": ["D", "G"],
    "G": []
}


# 基于递归实现的深度优先搜索
visited = set()
Solution().dfs_recursive(graph, "A", visited)
