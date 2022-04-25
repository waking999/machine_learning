import unittest
import L0066_PlusOne as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        digits = [1, 2, 3]
        expect = [1, 2, 4]
        actual = _s.plusOne(digits)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        digits = [4, 3, 2, 1]
        expect = [4, 3, 2, 2]
        actual = _s.plusOne(digits)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        digits = [9]
        expect = [1, 0]
        actual = _s.plusOne(digits)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
