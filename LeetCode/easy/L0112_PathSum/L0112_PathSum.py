from typing import Optional
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root is None:
            return False

        global find_flag
        find_flag = False

        self._check(root, 0, targetSum)

        return find_flag

    def _check(self, node, tmpSum, targetSum):
        global find_flag

        if find_flag:
            return
        if node.left is None and node.right is None:
            # it is a leaf
            if tmpSum + node.val == targetSum:
                find_flag = True
                return

        if node.left is not None:
            self._check(node.left, tmpSum + node.val, targetSum)
            if find_flag:
                return

        if node.right is not None:
            self._check(node.right, tmpSum + node.val, targetSum)
            if find_flag:
                return
