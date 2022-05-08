import unittest
import L0110_BalancedBinaryTree as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        root = [3, 9, 20, None, None, 15, 7]
        root = TreeNode.build_tree_node(root)
        expect = True
        actual = _s.isBalanced(root)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        root = [1, 2, 2, 3, 3, None, None, 4, 4]
        root = TreeNode.build_tree_node(root)
        expect = False
        actual = _s.isBalanced(root)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        root = []
        root = TreeNode.build_tree_node(root)
        expect = True
        actual = _s.isBalanced(root)
        self.assertEqual(expect, actual)

    def test_4(self):
        _s = leet.Solution()
        root = [1, 2, 2, 3, None, None, 3, 4, None, None, 4]
        root = TreeNode.build_tree_node(root)
        expect = False
        actual = _s.isBalanced(root)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
