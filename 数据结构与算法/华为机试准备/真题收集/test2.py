'''
solo从小就对英文字母感兴趣，尤其是元音字母(a.e,i,o,u,A,E,I,O,U)
他写日记时把元音写成大写，辅音写成小写，虽然别人看起来很别扭，但是solo非常熟练，请把一个句子翻译成solo日记的习惯

输入Who Love Solo
输出whO lOvE sOlO
'''

sentence = list(map(str,input().strip()))


# for word in sentence:
#     print(f"{word}",end=' ')

def is_char(char):
    if char>="a" and char<="z":
        return True
    if char>='A' and char <='Z':
        return True
    return False


def convert_to_solo_style(sentence):
    # 定义元音字母集合（包含大小写）
    vowels = set('aeiouAEIOU')
    result = []
    
    # 遍历句子中的每个字符
    for char in sentence:
        # 如果是元音字母，转换为大写
        if char.lower() in vowels:
            result.append(char.upper())
        # 如果是辅音字母，转换为小写
        elif is_char(char):
            result.append(char.lower())
        # 如果不是字母（如空格、标点符号等），保持不变
        else:
            result.append(char)
    
    # 将结果列表转换为字符串并返回
    return ''.join(result)

solo_style = convert_to_solo_style(sentence)

print(solo_style)