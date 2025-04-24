def optimize_bulb_switches(init_states, goal_states, node_count):

    # 操作总次数计数器
    switch_operations = 0
    
    # 记录每个节点的操作次数(0,1,2)
    node_operations = [0] * node_count
    
    # 节点队列，用于按层处理
    queue = [0]  # 从根节点开始
    visited = set([0])
    
    # 颜色变换查找表 - 避免重复计算
    color_transitions = {
        0: {0: 0, 1: 0, 2: 0},  # 空节点
        1: {0: 1, 1: 2, 2: 3},  # 红色变换
        2: {0: 2, 1: 3, 2: 1},  # 绿色变换
        3: {0: 3, 1: 1, 2: 2}   # 蓝色变换
    }
    
    # bfs
    while queue:
        current = queue.pop(0)
        
        # 跳过空
        if init_states[current] == 0:
            continue
            
        # 计算当前节点被影响后的颜色
        actual_color = init_states[current]
        parent_effect = 0
        
        # 自底向上计算所有祖先的累积影响
        parent_idx = current
        ancestors_visited = set()  # 避免重复
        
        while parent_idx > 0:
            parent_idx = (parent_idx - 1) // 2
            if parent_idx not in ancestors_visited:
                parent_effect = (parent_effect + node_operations[parent_idx]) % 3
                ancestors_visited.add(parent_idx)
        
        # 计算节点实际颜色
        for _ in range(parent_effect):
            actual_color = color_transitions[actual_color][1]
        
        # 计算当前节点需要的操作次数
        target_color = goal_states[current]
        operations_needed = 0
        temp_color = actual_color
        
        while temp_color != target_color:
            temp_color = color_transitions[temp_color][1]
            operations_needed += 1
            if operations_needed >= 3:  # 最多需要2次
                operations_needed %= 3
                break
        
        if operations_needed > 0:
            node_operations[current] = operations_needed
            switch_operations += operations_needed
        
        # 将子节点加入队列
        left_child = 2 * current + 1
        right_child = 2 * current + 2
        
        if left_child < node_count and left_child not in visited:
            queue.append(left_child)
            visited.add(left_child)
            
        if right_child < node_count and right_child not in visited:
            queue.append(right_child)
            visited.add(right_child)
    
    return switch_operations


if __name__ == "__main__":

    n = int(input().strip())
    initial_colors = list(map(int, input().strip().split(" ")))
    target_colors = list(map(int, input().strip().split(" ")))

    result = optimize_bulb_switches(initial_colors, target_colors, n)
    print(result)




