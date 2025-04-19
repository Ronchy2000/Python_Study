'''
描述
对于给定的字符串，统计其中的ASCII在0到127范围内的不同字符的个数。

备注：受限于输入，本题实际输入字符集为ASCII码在33到126范围内的可见字符。
您可以参阅下表获得其详细信息（您可能关注的内容是，这其中不包含空格、换行）。

输入描述：
在一行上输入一个长度1≦length(s)≦500的字符串s，代表给定的字符串。

输出描述：
在一行上输出一个整数，代表给定字符串中ASCII在0到127范围内的不同字符的个数。
'''

'''
核心：使用ord()函数获取字符的ASCII码


'''
if __name__ == '__main__':
    s = input().strip()
    unique_chars = set(s)  # 集合去重
    ascii_chars = [char for char in unique_chars if (ord(char)>=0 and ord(char)<=127)]

    print(len(ascii_chars))