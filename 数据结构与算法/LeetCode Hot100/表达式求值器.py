# 表达式求值器
"""
输入是一个字符串表达式，比如：
"3+(2-7)"
要求：按照正常的算术规则，返回计算结果。
"""

# 栈解法

def calculate(s: str) -> int:
    stack = []
    result = 0
    num = 0
    sign = 1
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch in "+-":
            result += sign * num
            num = 0
            sign = 1 if ch == "+" else -1
        elif ch == "(": # 压栈
            # 把当前层状态压栈，开启新的一层
            stack.append(result) # 保存现有结果
            stack.append(sign) # 保存现有符合
            result, sign = 0, 1 #重置
        elif ch == ")": # 出栈
            # 计算
            result += sign*num 
            num = 0
            result *= stack.pop() # 弹出符号
            result += stack.pop() # 弹出之前的结果
        else:
            # 空格直接跳过（如果题目可能包含空格）
            continue
    result += sign * num

    return result

if __name__ =="__main__":
    string = "5-1+(4+5)"
    print("result:", calculate(string))