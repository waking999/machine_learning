import unittest
import L0111_MinimumDepthofBinaryTree as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        root = [3, 9, 20, None, None, 15, 7]
        root = TreeNode.build_tree_node(root)
        expect = 2
        actual = _s.minDepth(root)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        root = [2, None, 3, None, 4, None, 5, None, 6]
        root = TreeNode.build_tree_node(root)
        expect = 5
        actual = _s.minDepth(root)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        root = [-9, -3, 2, None, 4, 4, 0, -6, None, -5]
        root = TreeNode.build_tree_node(root)
        expect = 3
        actual = _s.minDepth(root)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
