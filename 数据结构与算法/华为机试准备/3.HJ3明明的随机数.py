'''
题目：明明的随机数

对于明明生成的n个1到500之间的随机整数，需要完成以下任务：
1. 删去重复的数字，即相同的数字只保留一个
2. 然后再把这些数从小到大排序，按照排好的顺序输出

输入描述：
第一行先输入随机整数的个数n
接下来的n行每行输入一个整数，代表明明生成的随机数

输出描述：
输出删除重复数字，并排序后的结果，每个数字占一行
'''



def process_random_numbers():
    # 读取随机数的个数
    n = int( input().strip() )
    # print("n:",n)
    # print("type(n):",type(n))
    

    # 使用集合来自动去重  用集合来去重
    numbers = set()
    
    # 读取n个随机数
    for _ in range(n):
        num = int(input().strip())
        numbers.add(num)
    
    # 对去重后的数字进行排序
    sorted_numbers = sorted(numbers)
    
    # 输出结果
    for num in sorted_numbers:
        print(num)

if __name__ == "__main__":
    process_random_numbers()