from collections import deque

class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right

"""
       4
     /   \
    2     7
   / \   / \
  1   3 6   9

"""
# 对应的写法
root = TreeNode(4)
root.left = TreeNode(2)
root.right = TreeNode(7)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)
root.right.left = TreeNode(6)
root.right.right = TreeNode(9)

# 题目意思：把一个树的左右子树对掉，例如上面的树翻转后变成
"""
       4
     /   \
    7     2
   / \   / \
  9   6 3   1
"""
# 层序遍历打印树
def printTree(root):
    if not root:
        return
    q = deque([root])
    while q:
        node = q.popleft()
        print(node.val, end=" ")
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    print()

class Solution:
    def invertTree(self, root):
        if not root:
            return None
        
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
    
if __name__ == "__main__":
    print("原始树：")
    printTree(root)
    
    sol = Solution()
    new_root = sol.invertTree(root)

    print("翻转后：")
    printTree(new_root)