'''
描述
数据表记录包含表索引和数值，请对表索引相同的记录进行合并，
即将相同索引的数值进行求和运算，随后按照索引值的大小从小到大依次输出。

输入描述：
第一行输入一个整数n(1≦n≦500)代表数据表的记录数。
此后n行，第i行输入两个整数xi,yi(0≦xi≦11111111; 1≦yi≦10^5)代表数据表的第i条记录。

输出描述：
输出若干行，第i行输出两个整数，代表合并后数据表中第i条记录的索引和数值。
'''

'''
- 创建一个字典 record_dict 用于存储索引和对应的值
- 读取每条记录，如果索引已经存在于字典中，则将值累加；否则，将索引和值添加到字典中
- 最后，按照索引从小到大的顺序遍历字典，输出合并后的结果
'''

if __name__ == '__main__':
    cnt = int(input().strip())

    dict = {}

    for _ in range(cnt):
        num2 = list( map(int,input().strip().split(' ')) )
        if num2[0] in dict:
            dict[ num2[0] ] += num2[1]
        else:
            dict[ num2[0] ] = num2[1]

    # for key,value in dict.items():
    for key in sorted(dict.keys()):  # 利用sorted函数，按照key排序
        value = dict[key]
        # print(key,value) # 直接使用print的默认空格分隔符
        print(f"{key} {value}")  # 使用f-string格式化,是最标准的