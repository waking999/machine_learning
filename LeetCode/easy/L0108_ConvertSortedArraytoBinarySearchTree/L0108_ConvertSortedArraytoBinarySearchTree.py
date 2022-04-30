import math
from typing import Optional
from typing import List
from LeetCode.common.TreeNode import TreeNode


class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        _len = len(nums)
        _mid = _len // 2
        root = TreeNode(nums[_mid])
        self._insert_node(root, nums, 0, _mid - 1, 'L')
        self._insert_node(root, nums, _mid + 1, _len - 1, 'R')
        return root

    def _insert_node(self, root, nums, left_index, right_index, insert_flag):
        if left_index <= right_index:
            _lm = left_index + (right_index - left_index) // 2
            t_node = TreeNode(nums[_lm])

            if insert_flag == 'L':
                root.left = t_node
            elif insert_flag == 'R':
                root.right = t_node

            self._insert_node(t_node, nums, left_index, _lm - 1, 'L')
            self._insert_node(t_node, nums, _lm + 1, right_index, 'R')
