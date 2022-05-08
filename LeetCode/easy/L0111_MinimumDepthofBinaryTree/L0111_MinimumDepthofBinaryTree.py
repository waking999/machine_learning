from typing import Optional
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0

        if root.left is None:
            return self._height(root.right)+1

        if root.right is None:
            return self._height(root.left)+1

        l_h = self._height(root.left)
        r_h = self._height(root.right)
        return min(l_h, r_h) + 1

    def _height(self, node):
        if node is None:
            return 0
        l_h = self._height(node.left)
        r_h = self._height(node.right)
        return max(l_h, r_h) +1
