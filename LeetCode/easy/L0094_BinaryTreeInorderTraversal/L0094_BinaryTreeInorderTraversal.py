from typing import List
from typing import Optional
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def _inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return ' '
        _str = ''
        if root.left is not None:
            _str += self._inorderTraversal(root.left)
        if root.val is not None:
            _str += str(root.val) + ' '
        if root.right is not None:
            _str += self._inorderTraversal(root.right)
        return _str

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        _str = self._inorderTraversal(root)
        return list(map(int, _str.split()))
