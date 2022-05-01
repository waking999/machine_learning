import unittest
import L0145_BinaryTreePostorderTraversal as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        root = [1, None, 2, 3]
        root = TreeNode.build_tree_node(root)
        expect = [3, 2, 1]
        actual = _s.postorderTraversal(root)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
