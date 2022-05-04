import unittest
import L0684_RedundantConnection as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        edges = [[1, 2], [1, 3], [2, 3]]
        expect = [2, 3]
        actual = _s.findRedundantConnection(edges)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
