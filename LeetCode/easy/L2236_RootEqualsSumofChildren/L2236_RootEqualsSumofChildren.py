from typing import Optional
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def checkTree(self, root: Optional[TreeNode]) -> bool:
        return root.val == root.left.val + root.right.val
