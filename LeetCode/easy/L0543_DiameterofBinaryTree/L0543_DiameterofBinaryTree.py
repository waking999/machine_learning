from typing import Optional
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        d = 0

        def get_height(node):
            nonlocal d
            if node is None:
                return 0
            l = get_height(node.left)
            r = get_height(node.right)
            d = max(d, l + r)
            return max(l, r) + 1


        get_height(root)
        return d


