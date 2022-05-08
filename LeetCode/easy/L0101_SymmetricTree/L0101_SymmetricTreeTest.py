import unittest
import L0101_SymmetricTree as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        root = [1, 2, 2, 3, 4, 4, 3]
        root = TreeNode.build_tree_node(root)
        expect = True
        actual = _s.isSymmetric(root)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        root = [1, 2, 2, None, 3, None, 3]
        root = TreeNode.build_tree_node(root)
        expect = False
        actual = _s.isSymmetric(root)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
