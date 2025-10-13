# 定义链表 (类似结构体)
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val # 节点存储的值
        self.next = next # # 指向下一个节点的指针

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)


# 反转这个链表
class Solution:
    def reverseList(self, head):
        prev = None
        curr = head
        while curr:
            next_temp = curr.next # 保存下一个节点
            curr.next = prev # 当前节点指向前一个节点
            prev = curr # prev前进
            curr = next_temp # curr前进
        return prev
    
if __name__ == "__main__":
    # 打印原链表（用临时变量不动 head）
    curr = head
    while curr:
        print(curr.val, end=" ")
        curr = curr.next
    print()

    sol = Solution()
    new_head = sol.reverseList(head)
    # 打印反转后的链表
    curr = new_head
    while curr:
        print(curr.val, end=" ")
        curr = curr.next