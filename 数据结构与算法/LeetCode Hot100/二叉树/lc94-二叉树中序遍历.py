class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
    # 构建示例树
#         1
#        / \
#       2   3
#      / \    \
#     4   5    6
def build_tree():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(6)
    return root

# def preorder(root):
#     if not root: return []
#     return [root.val] + preorder(root.left) + preorder(root.right)
# 中序遍历 (左 -> 根 -> 右)
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)
