from typing import Optional
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if root is None:
            return True
        l_h = self._height(root.left)
        r_h = self._height(root.right)
        return abs(l_h - r_h) <= 1

    def _height(self, node):
        if node is None:
            return 0
        l_h = self._height(node.left)
        r_h = self._height(node.right)
        return max(l_h, r_h) + 1
