import unittest
import L0104_MaximumDepthofBinaryTree as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        root = [3, 9, 20, None, None, 15, 7]
        root = TreeNode.build_tree_node(root)
        expect = 3
        actual = _s.maxDepth(root)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        root = [1, None, 2]
        root = TreeNode.build_tree_node(root)
        expect = 2
        actual = _s.maxDepth(root)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
