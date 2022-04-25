import unittest
import L0094_BinaryTreeInorderTraversal as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        list = [1, None, 2, 3]
        root = TreeNode.build_tree_node(list)
        expect = [1, 3, 2]
        actual = _s.inorderTraversal(root)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
