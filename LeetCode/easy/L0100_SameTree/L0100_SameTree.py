from typing import Optional
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p is None and q is None:
            return True

        if (p is not None and q is None) or (p is None and q is not None) or (p is not None and q is not None and p.val != q.val):
            return False

        if not self.isSameTree(p.left, q.left):
            return False

        if not self.isSameTree(p.right, q.right):
            return False

        return True
