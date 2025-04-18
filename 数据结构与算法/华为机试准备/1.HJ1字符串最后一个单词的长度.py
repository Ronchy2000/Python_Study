'''
描述
对于给定的若干个单词组成的句子，每个单词均由大小写字母混合构成，单词间使用单个空格分隔。输出最后一个单词的长度。
输入描述：
在一行上输入若干个字符串，每个字符串代表一个单词，组成给定的句子。
除此之外，保证每个单词非空，由大小写字母混合构成，且总字符长度不超过  10^3

  。
输出描述：
在一行上输出一个整数，代表最后一个单词的长度。

示例1：
输入：HelloNowcoder
输出：13

说明：
在这个样例中，最后一个单词是 "HelloNowcoder"，长度为13

输入：
A B C D
输出：
1

'''

 
in_str = input()
if len(in_str) > 5000 or len(in_str) == 0:
    pass
last_string = in_str.split(" ")[-1]
length = len(last_string)
print(length)