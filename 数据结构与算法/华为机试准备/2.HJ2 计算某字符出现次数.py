'''
描述
对于给定的由大小写字母、数字和空格混合构成的字符串8，统计字符c在其中出现的次数。具体来说：
- 若c为大写或者小写字母，统计其大小写形态出现的次数和；
- 若c为数字，统计其出现的次数。
保证字符c仅为大小写字母或数字。
输入描述：
第一行输入一个长度1≤length（s）≤ 10^3，由大小写字母、数字和空格混合构成的字符串s。保证首尾不为空格。
第二行输入一个字符c，代表需要统计的字符。
输出描述：
在一行上输出一个整数，代表字符c 在字符串s中出现的次数。

输入：HELLONowcoder123
输出：3
说明： 由于 o为小写字母，因此统计其小写形态出现的次数和，即3。

输入：HE L L O Nowcoder123
1
输出：1
'''

def cnt_letter(in_str_list, to_find_letter):
    cnt = 0
      
    for i_string in in_str_list:
        i_string = i_string.lower()
        to_find_letter = to_find_letter.lower()
        for l in i_string:
            if l == to_find_letter:
                cnt += 1
    return cnt


if __name__ == "__main__":
    in_str_list = list( map(str,input().strip().split(' ')) ) # map() 返回一个迭代器，而不是字符串.  ['h', 'd', 'g', 'h']
    print("in_str_list:",in_str_list)
    to_find_letter = list(map(str,input().strip()))[-1]   # .['h', 'd', 'g', 'h']
    print("to_find_letter:",to_find_letter)

    cnt = cnt_letter(in_str_list, to_find_letter) # 变成小写字母

    print(cnt)