'''
你作为数据中心网络地址规划人员，需要尽可能满足不同业务的网络地址需求。每个业务需要的地址范围为一
个闭区间 ［start_ip,end_ip］ 表示，其中 start_ip是起始IP地址，end_ip 是终止IP地址，end_ip 大于等于
start_ip。
不同业务的IP地址不能重叠，因此你需要将业务地址需求，按照一定规则排序，让数据中心网络地址规划尽可
能满足更多数量的业务需求。当业物多数量相同时，以IP地址占用最少优先。当业务数量和IP地址占用数量相同
时，按照IP范围顺序，比较起始IP地址，起始地址最小者优先。


输入描述
1.第一行为业务个数N，有效范围为［1,1000］
2.输入N行IP地址区间，其中每个区间的格式为 start_ip end_ip（中间用空格分隔），其中 start_ip 和 end_ip
为合法的IPv4地址点分十进制格式，即A,B,C,D，其中A、B、C和D的取值范围为［0,255］。
3.IP地址大小的比较，是按照 A、B、C和D的顺序进行比较。
输出描述
输出排序好的M个IP 区间，每行一个。每个区间的格式为 start_ip end_ip，中间用空格分隔。
'''

'''
-----------------------------------
解法一：贪心算法 - 按照结束时间排序
这是最经典的区间调度方法：

将所有IP区间按结束地址（end_ip）排序
每次选择结束最早且与已选区间不冲突的区间
这种方法通常能获得最多数量的不重叠区间，但题目还有其他条件，需要调整。

解法二：贪心算法 - 按区间长度排序
由于题目要求在业务数量相同时，选择IP地址占用最少的方案：

将所有区间按长度（end_ip - start_ip）排序
从短到长选择不重叠的区间
但这个方法可能无法保证最大化业务数量。

解法三：动态规划
可以使用动态规划来解决这个问题：

将所有区间按照start_ip排序
定义dp[i]表示考虑前i个区间时的最优解
状态转移方程需要考虑是否选择当前区间
'''

def convert_ip_to_decimal(ip_addr):
    """将点分十进制IP地址转换为整数表示"""
    parts = ip_addr.split('.')
    decimal_value = 0
    for part in parts:
        decimal_value = decimal_value * 256 + int(part)
    return decimal_value

def decimal_to_ip(decimal_value):
    """将整数表示转换回点分十进制IP地址"""
    ip_parts = []
    for i in range(4):
        ip_parts.insert(0, str(decimal_value % 256))
        decimal_value //= 256
    return '.'.join(ip_parts)

def solve_ip_planning():
    business_count = int(input().strip())
    ip_ranges = []
    
    # 收集业务需要的IP范围
    for _ in range(business_count):
        first_ip, last_ip = input().strip().split()
        first_value = convert_ip_to_decimal(first_ip)
        last_value = convert_ip_to_decimal(last_ip)
        range_size = last_value - first_value + 1
        ip_ranges.append((first_value, last_value, first_ip, last_ip, range_size))
    
    # 按照结束IP排序 - 贪心选择
    ip_ranges.sort(key=lambda r: (r[1], r[4], r[0]))
    
    # 贪心选择不重叠的区间
    allocated_ranges = []
    current_boundary = -1
    
    for ip_range in ip_ranges:
        start_value = ip_range[0]
        # 检查是否与已分配区间冲突
        if start_value > current_boundary:
            allocated_ranges.append(ip_range)
            current_boundary = ip_range[1]
    
    # 输出结果
    for ip_range in allocated_ranges:
        print(f"{ip_range[2]} {ip_range[3]}")

if __name__ == "__main__":
    solve_ip_planning()