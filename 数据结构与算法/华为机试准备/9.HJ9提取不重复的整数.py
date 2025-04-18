'''
描述
对于给定的正整数n，按照从右向左的阅读顺序，返回一个不含重复数字的新的整数。

输入描述：
在一行上输入一个正整数n(1≦n≦10^8)代表给定的整数。
保证n的最后一位不为0。

输出描述：
在一行上输出一个整数，代表处理后的数字。
'''


if __name__ == '__main__':
    num = str(input().strip() )
    seen = set() # 利用集合去重,但是不能打印
    result = ''
    for i in range(len(num)-1,-1,-1):
        if num[i] not in seen:
            seen.add(num[i])
            result += num[i]
    print(result)
