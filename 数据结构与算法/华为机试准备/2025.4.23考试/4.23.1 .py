"""
给定一张二维图像，图中每个值表示该坐标下的亮度。现给定一个亮度值m，请返回离图像中心最近的k个亮度为m值的坐标(x,y)

提示，
图像中元素的坐标范围x：[0,w-1],y[0,h-1]
图像宽高w,h均为奇数，图像中心坐标(w-1)/2,(h-1)/2
平面上两点之间的距离为|x1-x2|+|y1-y2|
在距离相同的情况下，以x小的点优先；当x相同时以y小的点优先。
题目可保证至少存在一个亮度值为m的点
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

比较小，还是用枚举吧。

根据距离从小到大输出，

x1, y1, x2, y2, x3, y3....


"""


def fing_nearest_light_point(img,width,height, target, num_target):
    center_point_row = (height -1) // 2
    center_point_col = (width -1) // 2

    directions = [(0,1),(1,0),(0,-1),(-1,0)] # 模仿grid world中的动作，离散动作。
    
    row,col = center_point_row, center_point_col
    found_count = 0 #找够了就不找了。
    

    steps = 1 # 0
    current_dir = 0 #不同的动作
    point_visited = set((row,col)) #集合数组都行。
    found_target = []
    while found_count < num_target: #应该再加个判断，先不管了
        for _ in range(steps):
            action_r, action_c  = directions[current_dir]
            # 更新状态
            row = row + action_r
            col = col + action_c
            # 是否超出越界！
            if row>=0 and row < height and col>=0 and col < width: #是不是要等好
                point_visited.add((row, col))
                img_point = img[row][col]
                if img_point == target:
                    distance  = abs(row - center_point_row) + abs(col - center_point_col)  #能用np就好了
                    # found_target.append((row,col,distance)) #保存 行，列，距离   这搞反了
                    found_target.append((distance,col,row)) #保存 距离 ,行，列，   这搞反了
                    found_count = found_count + 1
                    if found_count >= num_target:
                        break
            else:
                row = row - action_r
                col = col - action_c
        current_dir = (current_dir +1) %4
        if current_dir % 2 == 0:
            steps = steps + 1 

    return found_target[:num_target]

if __name__ == "__main__":
    img = []

    w,h = map(int,input().strip().split(' ')) # 宽 x 高
    m = int(input().strip()) #亮度值
    k =  int(input().strip())# k < w*h  ->  k:个数 亮度值m
    # 接下来h行每个内为w个亮度（样例输入即可）
    for _ in range(h):
        img.append(list(map(int,input().strip().split(' '))))

    nearest_point = fing_nearest_light_point(img,w, h, m, k) #img,width,height, target, num_target
    # print(img,w,h,m,k) #测试输入正确！
    nearest_point.sort()
    result = []
    for dis, x, y in nearest_point:
        result.append(x)
        result.append(y)
    print(' '.join(map(str,result)))
