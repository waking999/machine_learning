from typing import Optional
from LeetCode.common.TreeNode import TreeNode

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0

        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1
