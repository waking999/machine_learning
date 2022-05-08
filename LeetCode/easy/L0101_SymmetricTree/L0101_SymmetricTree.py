from typing import Optional
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        return self._isSymmetric(root.left, root.right)

    def _isSymmetric(self, n1, n2):
        if n1 is None and n2 is None:
            return True

        if n1 is None or n2 is None:
            return False

        if n1.val == n2.val:
            return self._isSymmetric(n1.left, n2.right) and self._isSymmetric(n1.right, n2.left)
        else:
            return False
