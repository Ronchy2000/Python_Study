
'''
输入：
5 5
10
3
10 2 3 4 5
1 2 3 4 10
1 2 3 10 5
1 10 3 4 5
1 2 3 4 5
输出：
3 2 1 3 4 1

按照距离从小到大
查找到三个坐标为(3,2),(1,3),(4,1)则输出为3 2 1 3 4 1

输入：
3 3
10
6
1 10 1
10 10 10
1 10 1

输出：
1 1 0 1 1 0 1 2 2 1

图中可以看出距中心最近的坐标为(1,1),距离为0
剩下满足条件的坐标有4个，分别为(1,0)(0,1)(2,1)(1,2),距离均为1
在距离相同的情况下，以x小的点优先；当x相同时以y小的点优先。
所以输出为: 1 1 0 1 1 0 1 2 2 1
'''
def find_nearest_light_point(img, width, height, target, num_target):
    center_point_row = (height - 1) // 2
    center_point_col = (width - 1) // 2

    # directions = [(0,1),(1,0),(0,-1),(-1,0)] # 模仿grid world中的动作，离散动作。
    # 
    # row,col = center_point_row, center_point_col
    # found_count = 0 #找够了就不找了。
    # 
    # steps = 1 # 0
    # current_dir = 0 #不同的动作
    # point_visited = set((row,col)) #集合数组都行。
    
    found_target = []
    for row in range(height):
        for col in range(width):
            if img[row][col] == target:
                # 计算曼哈顿距离
                distance = abs(row - center_point_row) + abs(col - center_point_col)
                # 添加点(距离, x坐标, y坐标)
                found_target.append((distance, col, row))
    
    # 按距离从小到大排序，距离相同时按x从小到大排序，x相同时按y从小到大排序
    found_target.sort()
    
    return found_target[:num_target]

if __name__ == "__main__":
    img = []

    w, h = map(int, input().strip().split(' ')) # 宽 x 高
    m = int(input().strip()) #亮度值
    k = int(input().strip())# k < w*h  ->  k:个数 亮度值m
    # 接下来h行每个内为w个亮度（样例输入即可）
    for _ in range(h):
        img.append(list(map(int, input().strip().split(' '))))

    nearest_point = find_nearest_light_point(img, w, h, m, k) #img,width,height, target, num_target
    # print(img,w,h,m,k) #测试输入正确！
    
    # nearest_point.sort()
    
    result = []
    for dis, x, y in nearest_point:
        result.append(x)
        result.append(y)
    print(' '.join(map(str, result)))

# 普通枚举：遍历整个图像
