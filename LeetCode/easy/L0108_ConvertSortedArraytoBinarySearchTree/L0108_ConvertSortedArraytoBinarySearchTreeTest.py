import unittest
import L0108_ConvertSortedArraytoBinarySearchTree as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [-10, -3, 0, 5, 9]
        expect = [0, -3, 5, -10, None, None, 9]
        expect = TreeNode.build_tree_node(expect)
        actual = _s.sortedArrayToBST(nums)
        self.assertEqual(expect, actual)



if __name__ == '__main__':
    unittest.main()
