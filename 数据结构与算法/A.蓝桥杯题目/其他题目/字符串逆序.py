# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : 字符串逆序.py
# @Time      : 2022/2/4 下午10:31
# @Author    : Ronchy

'''
将一个字符串str的内容颠倒过来，并输出。str不得超过100个字符
'''
##解法一：
def reverse(str):
    rev_str = ''
    for i in range(1, len(str)+1):
        rev_str = rev_str+str[-i]
    return rev_str


#
# if __name__ == '__main__':
#     str = input('请输入字符串：')
#     print('已颠倒：', reverse(str) )


##解法二：
def reverse2(str):
    return str[-1::-1]

'''
知识点：倒着开始，步长为-1，实现倒着打印
'''
if __name__ == '__main__':
    str = input('请输入字符串：')
    print('已颠倒：', reverse2(str) )
