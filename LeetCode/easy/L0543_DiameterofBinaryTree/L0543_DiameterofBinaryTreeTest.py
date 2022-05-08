import unittest
from LeetCode.common.TreeNode import TreeNode
import L0543_DiameterofBinaryTree as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        root = [1, 2, 3, 4, 5]
        root = TreeNode.build_tree_node(root)
        expect = 3
        actual = _s.diameterOfBinaryTree(root)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
