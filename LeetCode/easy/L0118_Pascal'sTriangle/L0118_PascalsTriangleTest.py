import unittest
import L0118_PascalsTriangle as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        numRows = 5
        expect = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
        actual = _s.generate(numRows)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        numRows = 1
        expect = [[1]]
        actual = _s.generate(numRows)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
