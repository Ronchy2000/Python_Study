'''
描述
对于给定的若干个单词组成的句子，每个单词均由大小写字母混合构成，单词间使用单个空格分隔。
输出以单词为单位逆序排放的结果，即仅逆序单词间的相对顺序，不改变单词内部的字母顺序。

输入描述：
在一行上输入若干个字符串，每个字符串代表一个单词，组成给定的句子。
除此之外，保证每个单词非空，由大小写字母混合构成，且总字符长度不超过10^3。

输出描述：
在一行上输出一个句子，代表以单词为单位逆序排放的结果。

输入：
Nowcoder Hello
输出：
Hello Nowcoder

'''

if __name__ == '__main__':
    s = list( map(str,input().strip().split(' ')) )
    output_s = ''
    # print(s)

    # 用空格连接单词列表，形成新句子
    for i in s[::-1]:
        output_s += i
        output_s += ' ' # 注意空格，否则会出现NowcoderHello的情况
    print(output_s)


'''
- 首先读取输入的句子
- 使用 split() 方法按空格分割句子，得到单词列表
- 使用切片操作 [::-1] 将单词列表逆序
- 使用 join() 方法将逆序后的单词列表用空格连接起来，形成新的句子
- 输出结果

if __name__ == '__main__':
    # 读取输入的句子
    sentence = input().strip()
    
    # 按空格分割句子成单词列表
    words = sentence.split()
    
    # 逆序单词列表
    reversed_words = words[::-1]
    
    # 用空格连接单词列表，形成新句子
    reversed_sentence = ' '.join(reversed_words)
    
    # 输出结果
    print(reversed_sentence)
    
'''