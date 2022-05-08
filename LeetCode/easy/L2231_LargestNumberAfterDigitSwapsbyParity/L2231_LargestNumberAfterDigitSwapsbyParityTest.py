import unittest
import L2231_LargestNumberAfterDigitSwapsbyParity as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        num = 1234
        expect = 3412
        actual = _s.largestInteger(num)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        num = 65875
        expect = 87655
        actual = _s.largestInteger(num)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        num = 8
        expect = 8
        actual = _s.largestInteger(num)
        self.assertEqual(expect, actual)

    def test_4(self):
        _s = leet.Solution()
        num = 60
        expect = 60
        actual = _s.largestInteger(num)
        self.assertEqual(expect, actual)

    def test_5(self):
        _s = leet.Solution()
        num = 247
        expect = 427
        actual = _s.largestInteger(num)
        self.assertEqual(expect, actual)

    def test_6(self):
        _s = leet.Solution()
        num = 266
        expect = 662
        actual = _s.largestInteger(num)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
