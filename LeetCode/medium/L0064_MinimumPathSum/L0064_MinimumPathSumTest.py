import unittest
import L0064_MinimumPathSum as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        grid = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
        expect = 7
        actual = _s.minPathSum(grid)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        grid = [[1, 2, 3], [4, 5, 6]]
        expect = 12
        actual = _s.minPathSum(grid)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
