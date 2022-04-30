import unittest
import L0119_PascalsTriangleII as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        rowIndex = 3
        expect = [1, 3, 3, 1]
        actual = _s.getRow(rowIndex)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        rowIndex = 0
        expect = [1]
        actual = _s.getRow(rowIndex)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        rowIndex = 1
        expect = [1, 1]
        actual = _s.getRow(rowIndex)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
