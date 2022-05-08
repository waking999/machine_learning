import unittest
import L0112_PathSum as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        root = [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1]
        root = TreeNode.build_tree_node(root)
        targetSum = 22
        expect = True
        actual = _s.hasPathSum(root, targetSum)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        root = [1, 2, 3]
        root = TreeNode.build_tree_node(root)
        targetSum = 5
        expect = False
        actual = _s.hasPathSum(root, targetSum)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        root = []
        root = TreeNode.build_tree_node(root)
        targetSum = 0
        expect = False
        actual = _s.hasPathSum(root, targetSum)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
