import unittest
import L2243_CalculateDigitSumofaString as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        s = "11111222223"
        k = 3
        expect = "135"
        actual = _s.digitSum(s, k)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        s = "00000000"
        k = 3
        expect = "000"
        actual = _s.digitSum(s, k)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
