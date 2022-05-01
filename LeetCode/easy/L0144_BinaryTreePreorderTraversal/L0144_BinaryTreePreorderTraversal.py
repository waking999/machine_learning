from typing import Optional
from typing import List
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        _list = []
        self._order(root, _list)
        return _list

    def _order(self, node, _list):
        if node is None:
            return

        _list.append(node.val)
        self._order(node.left, _list)
        self._order(node.right, _list)

