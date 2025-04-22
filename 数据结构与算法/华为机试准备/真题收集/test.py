'''
每个句子由多个单词组成，句子中的每个单词长度都可能不一样，我们假设每个单词的长度Ni为该单词的重量，你需要做的就是给出整个句子的平均重量V
输入，输入只有一行，包含一个字符床S，代表整个句子，句子中只包含大小写的英文字母，每个单词之间有一个空格。
输出句子S的平均重量V（四舍五入保留两文小数）

'''



def calculate_average_weight(sentence):
    # 按空格拆分句子为单词列表
    words = sentence.split()
    
    # 如果没有单词，返回0
    if not words:
        return 0.00
    
    # 计算每个单词的长度(重量)
    weights = [len(word) for word in words]
    
    # 计算平均重量
    average_weight = sum(weights) / len(words)
    
    # 四舍五入保留两位小数
    return round(average_weight, 2)

# 从标准输入读取句子
if __name__ == "__main__":
    sentence = input().strip()
    result = calculate_average_weight(sentence)
    print(f"{result:.2f}")