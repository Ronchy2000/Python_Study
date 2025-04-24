"""
3.1基于队列实现的广度优先搜索算法步骤
1.定义 graph 为存储无向图的嵌套数组变量，visited 为标记访问节点的集合变量，queue为存放节点的队列，u为开始节点，
定义 def bfs（graph,u）：为队列实现的广度优先搜索方法。
2. 首先将起始节点w 标记为已访问，并将其加入队列中，即 visited.add（u），queue.append（u）。
3. 从队列中取出队头节点 2。访问节点 w，并对节点进行相关操作（看具体题目要求）。
4. 遍历节点w的所有未访问邻接节点（节点 不在 visited 中）。
5.将节点2标记已访问，并加入队列中，即 visited.add（v）， queue.append（v）。
6. 重复步骤3～5，直到队列 queue 为空。
"""

import collections

class Solution:
    def bfs(self, graph, u):
        visited = set()                     # 使用 visited 标记访问过的节点
        queue = collections.deque([])       # 使用 queue 存放临时节点
        
        visited.add(u)                      # 将起始节点 u 标记为已访问
        queue.append(u)                     # 将起始节点 u 加入队列中
        
        while queue:                        # 队列不为空
            u = queue.popleft()             # 取出队头节点 u
            print(u)                        # 访问节点 u
            for v in graph[u]:              # 遍历节点 u 的所有未访问邻接节点 v
                if v not in visited:        # 节点 v 未被访问
                    visited.add(v)          # 将节点 v 标记为已访问
                    queue.append(v)         # 将节点 v 加入队列中
                

graph = {
    "0": ["1", "2"],
    "1": ["0", "2", "3"],
    "2": ["0", "1", "3", "4"],
    "3": ["1", "2", "4", "5"],
    "4": ["2", "3"],
    "5": ["3", "6"],
    "6": []
}

# 基于队列实现的广度优先搜索
Solution().bfs(graph, "0")
