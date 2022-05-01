import unittest
import L0144_BinaryTreePreorderTraversal as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        root = [1, None, 2, 3]
        root = TreeNode.build_tree_node(root)
        expect = [1, 2, 3]
        actual = _s.preorderTraversal(root)
        self.assertEqual(expect, actual)


    def test_2(self):
        _s = leet.Solution()
        root = []
        root = TreeNode.build_tree_node(root)
        expect = []
        actual = _s.preorderTraversal(root)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        root = [1]
        root = TreeNode.build_tree_node(root)
        expect = [1]
        actual = _s.preorderTraversal(root)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
