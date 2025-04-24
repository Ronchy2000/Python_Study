"""
假设有个二叉树，树的每个节点代表一个灯泡，每个灯泡有三种颜色状态：
红色（整数1）
绿色（整数2）
蓝色（整数3）

每个节点都有一个开关，当按下某个节点的开关时，以该节点为根节点的子树上所有节点的灯会根据当前的颜色按照“红->绿->蓝->红->...”
的顺序切换一次颜色。
目标：计算将二叉树从初始颜色initial切换到目标颜色状态target所需的最小开关切换次数。

补充：
层序遍历是指从上到下，从左到右的逐层遍历二叉树的节点，并将遍历结果保存在一组数组中，如果某个节点在二叉树中不存在，则在数组中用0表示
切换开关，影响的是“传递性”，即切换一个节点的开关会影响以该节点为根节点的子树上所有节点的灯泡颜色。

样例：


输入：
第一行输入一个整数n。代表initial[]和target[]数组大小
第二行为n个整数，代表initial元素值
第三行输入n个整数，target[]的元素值

输出：一个整数，表示最少开关切换的次数

参数取值范围：
initial.lengeth == targets.length
 0 <initial[i] < =3
  0 <target[i] < =3
  如果initial[i]==0,则targets[i]==0
  

输入：
5
1 2 3 0 1
2 3 1 0 2
输出:
1

输入：
7
1 2 3 1 2 3 1
3 1 2 3 1 2 1

输出：
3

初始状态initial[]为[1,2,3,0,1]
代表
  1
2   3 (分别是左节点，右节点)
1

切换一次根节点颜色:
1 -> 2  # 根节点切换
2->3  3->1  # 子左右节点切换
1->2

切换后变为
  2
3   1
2
即满足[2,3,1,0,2],满足目标状态target，总共切换一次，所以返回1


例子2:
输入：
7
1 2 3 1 2 3 1
3 1 2 3 1 2 1
输出：
3


初始状态[1,2,3,1,2,3,1]
代表树
        1
    2     3
1     2  3  1

切换一次根节点变为
   2
3    1
2 3  1  2

切换一次
  3
1   2
3 1 2  3
切换末尾
   3
1     2
3 1  2 3->1
即[3,1,2,3,1,2,1]满足目标target，共切换三次
输出3
"""
def solve_color_tree(initial, target, length):
    """计算将二叉树从初始颜色转换到目标颜色的最小开关次数"""
    
    # 计算两个颜色状态之间所需的切换次数
    def calc_color_diff(start_color, end_color):
        if start_color == 0 or end_color == 0:  # 节点不存在
            return 0
        # 计算颜色转换需要的步骤数(最多2次)
        diff = (end_color - start_color) % 3
        return diff
    
    # 计算颜色在切换特定次数后的结果
    def color_after_switch(base_color, times):
        if base_color == 0:  # 节点不存在
            return 0
        # 1->2->3->1 循环
        return ((base_color - 1 + times) % 3) + 1
    
    # 总切换次数
    total_switches = [0]  # 使用列表作为可变对象避免使用nonlocal
    
    def dfs(index, parent_switches):
        """深度优先搜索计算切换次数
        index: 当前节点索引
        parent_switches: 父节点累积的切换次数
        """
        # 节点不存在
        if index >= length or initial[index] == 0:
            return 0
        
        # 左右子节点索引
        left = 2 * index + 1
        right = 2 * index + 2
        
        # 先处理左右子树
        if left < length:
            dfs(left, parent_switches)
        
        if right < length:
            dfs(right, parent_switches)
        
        # 计算当前节点在父节点影响下的颜色
        current = color_after_switch(initial[index], parent_switches)
        
        # 计算需要几次切换才能达到目标颜色
        needed = calc_color_diff(current, target[index])
        
        # 累加总切换次数
        if needed > 0:
            total_switches[0] += needed
        
        # 返回当前节点对子树的影响
        return (parent_switches + needed) % 3
    
    # 从根节点开始搜索
    dfs(0, 0)
    return total_switches[0]

# 主程序
if __name__ == "__main__":
    # 读取输入
    length = int(input().strip())
    initial_colors = list(map(int, input().strip().split()))
    target_colors = list(map(int, input().strip().split()))
    
    # 求解并输出
    answer = solve_color_tree(initial_colors, target_colors, length)
    print(answer)