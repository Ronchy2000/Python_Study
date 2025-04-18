'''
描述
对于给定的由小写字母和数字混合构成的字符串s，你需要按每8个字符换一行的方式书写它，具体地：
- 书写前8个字符，换行；
- 书写接下来的8个字符，换行；
- ……
- 重复上述过程，直到字符串被完全书写。
特别地，如果最后一行不满8个字符，则需要在字符串末尾补充0，直到长度为8。

输入描述：
在一行上输入一个长度1≦length(s)≦100，由小写字母和数字混合构成的字符串s。

输出描述：
输出若干行，每行输出8个字符，代表按题意书写的结果。
'''

input_str = input().strip()

remain_str = len(input_str) % 8
padding = 0 if remain_str == 0 else 8 - remain_str # 补几个0

input_str += '0' * padding

for i in range(0, len(input_str), 8):
    print(input_str[i:i+8])

