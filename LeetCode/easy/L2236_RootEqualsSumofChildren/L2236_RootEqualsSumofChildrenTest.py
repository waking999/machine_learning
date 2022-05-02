import unittest
import L2236_RootEqualsSumofChildren as leet
from LeetCode.common.TreeNode import TreeNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        root = [10, 4, 6]
        root = TreeNode.build_tree_node(root)
        expect = True
        actual = _s.checkTree(root)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        root = [5, 3, 1]
        root = TreeNode.build_tree_node(root)
        expect = False
        actual = _s.checkTree(root)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
