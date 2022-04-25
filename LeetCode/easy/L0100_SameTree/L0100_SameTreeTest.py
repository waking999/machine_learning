import unittest
import L0100_SameTree as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        p = [1, 2, 3]
        q = [1, 2, 3]
        p = TreeNode.build_tree_node(p)
        q = TreeNode.build_tree_node(q)
        expect = True
        actual = _s.isSameTree(p, q)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        p = [1, 2]
        q = [1, None, 2]
        p = TreeNode.build_tree_node(p)
        q = TreeNode.build_tree_node(q)
        expect = False
        actual = _s.isSameTree(p, q)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        p = [1, 2, 1]
        q = [1, 1, 2]
        p = TreeNode.build_tree_node(p)
        q = TreeNode.build_tree_node(q)
        expect = False
        actual = _s.isSameTree(p, q)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
