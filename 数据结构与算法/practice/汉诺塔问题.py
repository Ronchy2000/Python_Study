#求解汉诺塔问题，很套路的问题！
#该问题有个公式解法：
#汉诺塔如图所示：

# |    |    |
# a    b    c
#
def hanoi(n,a,b,c):
    if n>0:
        hanoi(n-1,a,c,b) #上方（n-1个盘子）整体 从 a->c->b
        print("moving from %s to %s" %(a,c) ) #最下方第n个盘子从a->c
        hanoi(n-1,b,a,c) #上方（n-1个盘子）整体 从 b->a->c

hanoi(2,'A','B','C')